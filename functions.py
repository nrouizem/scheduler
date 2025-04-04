import numpy as np
import datetime
import zoneinfo
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Dict
import random
import requests
from ortools.sat.python import cp_model
import math
from tqdm import trange
from ortools.sat.python import cp_model
import os

# Example focus level function based on time-of-day.
def get_focus_level(time: datetime.datetime) -> float:
    # A simple heuristic:
    # High focus in the morning, moderate in the mid-day, lower in the late afternoon.
    if 7 <= time.hour < 10:
        return 0.9
    elif 10 <= time.hour < 13:
        return 0.8
    elif 13 <= time.hour < 16:
        return 0.7
    else:
        return 0.5
    
def get_weather(dt: datetime.datetime) -> dict:
    """
    Retrieve the current weather for the given datetime (assuming chicago) using weatherapi.com.

    Parameters:
    - date_time: datetime.datedatetime object

    Returns:
    - A dictionary with weather details (condition, temperature, etc.), or None if an error occurs.
    """
    # convert dt into date and time
    today = datetime.date.today()
    day_offset = (dt.date() - today).days
    hour = dt.hour

    base_url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": os.environ.get("WEATHER_API_KEY"),
        "q": "60637",
        "days": 3,
        "aqi": "no"  # Air Quality Index is optional; set to "yes" if needed.
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        data = data["forecast"]["forecastday"][day_offset]["hour"][hour]
        # Extract relevant weather data
        future_weather = {
            "condition": data["condition"]["text"],
            "feelslike_f": data['feelslike_f'],
            "chance_of_rain": data['chance_of_rain'],
            "wind_mph": data["wind_mph"],
        }
        return future_weather

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

@dataclass
class Task:
    name: str
    estimated_duration: int  # in minutes
    required_focus: float  # e.g., 0.0 (low) to 1.0 (high)
    category: str  # e.g., 'gardening', 'writing'
    flexibility: float # e.g., 0.0 (none/fixed) to 1.0 (very flexible) but what does flexibility mean for a task?? "does it have to happen today" maybe?
    buffer_before: int = 15 # travel/other time beforehand to take into account
    buffer_after: int = 15 # rest/other time after to take into account

    def duration_with_buffer(self) -> int:
        """Return the duration of task (buffer included) in minutes."""
        return self.estimated_duration + self.buffer_before + self.buffer_after

@dataclass
class Event:
    name: str
    start: datetime.datetime
    end: datetime.datetime
    required_focus: float  # e.g., 0.0 (low) to 1.0 (high)
    category: str  # e.g., 'gardening', 'meeting'
    flexibility: float # e.g., 0.0 (none/fixed) to 1.0 (very flexible)
    buffer_before: int = 15 # travel/other time beforehand to take into account
    buffer_after: int = 15 # rest/other time after to take into account

    def duration(self) -> int:
        """Return the duration of the event in minutes."""
        return int((self.end - self.start).total_seconds() / 60)
    
    def duration_with_buffer(self) -> int:
        """Return the duration of task (buffer included) in minutes."""
        return self.duration() + self.buffer_before + self.buffer_after

@dataclass
class TimeSlot:
    start: datetime.datetime
    end: datetime.datetime
    focus_level: float  # e.g., 0.0 (low) to 1.0 (high)
    weather: str        # e.g., 'sunny', 'rainy', 'cloudy'

    def duration(self) -> int:
        """Return the duration of the timeslot in minutes."""
        return int((self.end - self.start).total_seconds() / 60)
    
# Function to generate timeslots dynamically
def generate_timeslots(
    day_start: datetime.datetime,
    day_end: datetime.datetime,
    slot_duration_minutes: int,
    focus_func: Callable[[datetime.datetime], float],
    weather_func: Callable[[datetime.datetime], str]
) -> List[TimeSlot]:
    timeslots = []
    current = day_start
    slot_delta = datetime.timedelta(minutes=slot_duration_minutes)
    
    # Generate timeslots while there is enough time left in the day.
    while current + slot_delta <= day_end:
        # Create a timeslot with dynamic focus level and weather
        focus = focus_func(current)
        weather = weather_func(current)
        slot = TimeSlot(start=current, end=current + slot_delta, focus_level=focus, weather=weather)
        timeslots.append(slot)
        
        # Move to the next slot: add both the duration and the buffer period.
        current += slot_delta
    
    return timeslots

def score_weather(item, timeslot):
    """Returns float between 0 (bad match) and 1 (great match)."""
    weather = timeslot.weather
    if "garden" in item.category.lower():
        rain_score = (100 - weather["chance_of_rain"]) / 100
        temperature_score = max(0, 100 - 2*(np.abs(75 - weather["feelslike_f"]))) / 100
        return (2*rain_score + temperature_score)/3
    return 1

def score_focus(item, timeslot):
    # can focus well enough in that time slot
    if timeslot.focus_level >= item.required_focus:
        return 1
    
    # otherwise, steep descending function
    delta = item.required_focus - timeslot.focus_level
    max_delta = 0.4     # the delta that would return 0
    return max(0, 1 - delta/max_delta)

def score_time(item, timeslot):
    # matching timeless tasks is irrelevant, return 0 to not affect other factors
    if isinstance(item, Task):
        return 0
    
    if item.start == timeslot.start:
        return 1
    
    delta = item.start - timeslot.start
    delta = np.abs(delta.total_seconds())/60    # in minutes
    
    tolerance = 500 * item.flexibility**2
    tolerance = max(tolerance, 1e-6)
    # Use an exponential decay function. When time_delta is 0, exp(0)=1.
    return math.exp(-abs(delta) / tolerance)

def score(item, timeslot):
    weights = [
        2,      # weather
        1,      # focus
        10       # time
    ]
    match_array = [
        score_weather(item, timeslot),
        score_focus(item, timeslot),
        score_time(item, timeslot)
    ]
    return np.dot(match_array, weights)

# ok naive implementation, we'll see how bad it is

# first just try to put each item in its best timeslot, see if it works
def greedy_schedule(items, timeslots):
    timeslots_used = []
    schedule = []
    t_score = 0
    for item in items:
        best_timeslots = []
        best_score = 0
        timeslots_required = math.ceil(item.duration_with_buffer() / timeslots[0].duration())
        for i, timeslot in enumerate(timeslots):
            if i + timeslots_required > len(timeslots):
                break
            sc = score(item, timeslot)
            if sc > best_score:
                best_timeslots = [j for j in range(i, i + timeslots_required)]
                best_score = sc
        for timeslot in best_timeslots:
            if timeslot in timeslots_used:
                return None
        timeslots_used.extend(best_timeslots)
        item.calculated_start = timeslots[best_timeslots[0]].start
        item.calculated_end = timeslots[best_timeslots[0]].start + datetime.timedelta(minutes=item.duration())
        schedule.append(item)
        t_score += best_score
    
    schedule.sort(key=lambda task: task.calculated_start)
    return schedule, t_score

# randomly assign timeslots (really dumb)
def random_schedule(items, timeslots):
    #greedy_attempt = greedy_schedule(items, timeslots)
    #if greedy_attempt:
    #    print("Processed with greedy schedule")
    #    return greedy_attempt
    
    print("Processed with random schedule generator")
    best_score = 0
    best_schedule = []
    for _ in range(10000):
        this_score = 0
        schedule = []
        flag = 0
        timeslots_used_idx = []
        random.shuffle(items)
        for item in items:
            i = random.randint(0, len(timeslots)-1)
            timeslots_required = math.ceil(item.duration_with_buffer() / timeslots[0].duration())

            new_timeslots = [j for j in range(i, i + timeslots_required)]
            for j in new_timeslots:
                if j in timeslots_used_idx:
                    flag = 1
            if flag:
                break

            timeslots_used_idx.extend(new_timeslots)
            this_score += score(item, timeslots[i])
            item.calculated_start = timeslots[i].start + item.buffer_before
            item.calculated_end = item.calculated_start + datetime.timedelta(minutes=item.duration())
            schedule.append(item)
        
        if this_score > best_score and flag == 0:
            best_schedule = schedule.copy()
            best_score = this_score
    
    best_schedule.sort(key=lambda task: task.calculated_start)
    return best_schedule, best_score

def schedule_with_ortools(items, timeslots):
    """
    Schedule items (tasks and events) into contiguous timeslot blocks using OR-Tools.
    Each item is assigned a start index (for a contiguous block of timeslots) such that:
      - The block's total duration covers the item (including buffers)
      - No items overlap
      - The total score (sum of score(item, starting_timeslot) for each item) is maximized.
    
    Returns:
      best_schedule: dictionary mapping item names to the scheduled start time (as string)
      best_score: the objective value.
    """
    print("Processed with or_tools")
    model = cp_model.CpModel()
    n_items = len(items)
    n_slots = len(timeslots)
    
    # Precompute how many timeslots are needed per item.
    # We assume all timeslots have the same duration.
    slot_duration = timeslots[0].duration()
    required_slots = [
        math.ceil(item.duration_with_buffer() / slot_duration) for item in items
    ]
    
    # Create decision variables: start index for each item.
    start_vars = []
    for i, item in enumerate(items):
        # The latest start index is such that the contiguous block fits within the timeslot array.
        max_start = n_slots - required_slots[i]
        var = model.NewIntVar(0, max_start, f'start_{i}')
        start_vars.append(var)
    
    # Create interval variables for each item.
    intervals = []
    for i, item in enumerate(items):
        duration = required_slots[i]
        interval = model.NewIntervalVar(start_vars[i], duration, start_vars[i] + duration, f'interval_{i}')
        intervals.append(interval)
    
    # Ensure that intervals (assigned timeslot blocks) do not overlap.
    model.AddNoOverlap(intervals)
    
    # Precompute a score table for each item and each possible start index.
    # score_table[i][s] is the score for item i if its block starts at timeslot index s.
    score_table = []
    for i, item in enumerate(items):
        max_start = n_slots - required_slots[i]
        scores = []
        for s in range(max_start + 1):
            # Here, for simplicity, we evaluate score using the starting timeslot.
            # You could enhance this by aggregating over the block (including buffer considerations).
            scores.append(int(score(item, timeslots[s]) * 100_000))
        score_table.append(scores)
    
    
    # Create integer variables representing the score for each item,
    # using an element constraint to select the score based on the chosen start index.
    score_vars = []
    for i, item in enumerate(items):
        s_var = model.NewIntVar(min(score_table[i]), max(score_table[i]), f'score_{i}')
        model.AddElement(start_vars[i], score_table[i], s_var)
        score_vars.append(s_var)
    
    # Define the objective: maximize the total score.
    model.Maximize(sum(score_vars))
    
    # Solve the model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        best_schedule = []
        best_score = solver.ObjectiveValue()
        for i, item in enumerate(items):
            start_index = solver.Value(start_vars[i])
            item.calculated_start = timeslots[start_index].start
            item.calculated_end = item.calculated_start + datetime.timedelta(minutes=item.duration())
            best_schedule.append(item)
        best_schedule.sort(key=lambda task: task.calculated_start)
        return best_schedule, best_score/100_000
    else:
        return None, None
    
def schedule(items, timeslots):
    or_tools_schedule = schedule_with_ortools(items, timeslots)
    if or_tools_schedule[0]:
        return or_tools_schedule
    
    return random_schedule(items, timeslots)

def convert_gcal_event(gcal_event: dict) -> Event:
    # Extract event name (use a default if not provided)
    name = gcal_event.get('summary', 'Untitled Event')
    
    # Get start and end times. Use 'dateTime' if available; otherwise 'date' (for all-day events)
    start_str = gcal_event.get('start', {}).get('dateTime') or gcal_event.get('start', {}).get('date')
    end_str = gcal_event.get('end', {}).get('dateTime') or gcal_event.get('end', {}).get('date')
    
    # Parse the datetime strings. If it's an all-day event (date only), assume midnight.
    try:
        start_dt = datetime.datetime.fromisoformat(start_str)
    except Exception:
        start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d")
    
    try:
        end_dt = datetime.datetime.fromisoformat(end_str)
    except Exception:
        end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d")
    
    # Set default values for fields not provided by Google Calendar.
    # You might adjust these defaults or enhance the conversion based on your app's needs.
    required_focus = 0.5   # Default focus level
    category = 'meeting'   # Default category; you could infer this from event details if needed.
    flexibility = 0.0      # Assume events from the calendar are fixed by default.
    
    return Event(
        name=name,
        start=start_dt,
        end=end_dt,
        required_focus=required_focus,
        category=category,
        flexibility=flexibility
    )