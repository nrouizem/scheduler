import numpy as np
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Dict
import random
import requests
from ortools.sat.python import cp_model
import math
from ortools.sat.python import cp_model
import os
from dateutil.parser import isoparse
import json
from collections import defaultdict
from config import MAX_MINUTES_PER_DAY
from zoneinfo import ZoneInfo
import copy

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
        return {
            "condition": "Unknown",
            "feelslike_f": 70,
            "chance_of_rain": 0,
            "wind_mph": 5
        }

@dataclass
class Task:
    name: str
    estimated_duration: int  # in minutes
    required_focus: float  # e.g., 0.0 (low) to 1.0 (high)
    category: str  # e.g., 'gardening', 'writing'
    flexibility: float # e.g., 0.0 (none/fixed) to 1.0 (very flexible) but what does flexibility mean for a task?? "does it have to happen today" maybe?
    priority: int = 3  # 1 (low) to 5 (high), default = 3
    buffer_before: int = 15 # travel/other time beforehand to take into account
    buffer_after: int = 15 # rest/other time after to take into account
    created_at: datetime = None
    recurrence: str = "none"

    def duration(self):
        """Helper function to be consistent with Event class."""
        return self.estimated_duration

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
    weather: dict

    def duration(self) -> int:
        """Return the duration of the timeslot in minutes."""
        return int((self.end - self.start).total_seconds() / 60)
    
# Function to generate timeslots dynamically
def generate_timeslots(
    day_start: datetime.datetime,
    day_end: datetime.datetime,
    slot_duration_minutes: int,
    focus_func: Callable[[datetime.datetime], float],
    weather_func: Callable[[datetime.datetime], dict]
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

def category_preferences(task, timeslot):
    bonus = 0.0
    hour = timeslot.start.hour

    if "writing" in task.category:
        if 7 <= hour <= 11:
            bonus += 0.5  # mornings preferred
        else:
            bonus -= 0.2  # avoid afternoons

    elif "exercise" in task.category:
        if timeslot.weather.get("sunlight", 1.0) < 0.5:
            bonus -= 0.5  # penalize low sunlight
        if 6 <= hour <= 9 or 16 <= hour <= 19:
            bonus += 0.3  # morning or early evening

    elif "email" in task.category:
        bonus += 0.1 if hour >= 12 else -0.1  # slight preference for afternoons

    elif "admin" in task.category:
        if hour >= 17:
            bonus += 0.2  # end-of-day tasks

    # Add more categories here as needed

    return bonus

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

    base_score = np.dot(match_array, weights)
    tz = ZoneInfo("America/Chicago")

    priority_boost = 0
    if isinstance(item, Task):
        priority_boost = (item.priority - 3) * 0.5  # -1.0 to +1.0 scaling
        # Aging boost (max +1 after 5 days)
        if item.created_at:
            now = datetime.datetime.now(tz)
            days_old = (now.date() - item.created_at.date()).days
            aging_boost = min(1.0, days_old * 0.2)
            base_score += aging_boost
        
        #category_bonus = category_preferences(item, timeslot)
        category_bonus = 0      # not using category bonus for now, but good to keep in mind
        base_score += category_bonus
    
    if isinstance(item, Task) and item.flexibility > 0.5:
        # this task is very flexible, so deprioritize it slightly
        base_score *= (1 - 0.2 * item.flexibility)
    
    today = datetime.datetime.now(tz).date()

    if isinstance(item, Task) and item.flexibility < 0.5:
        if timeslot.start.date() == today:
            base_score += 1.0  # boost for today if task is inflexible
    
    days_ahead = (timeslot.start.date() - today).days
    penalty = 0.1 * days_ahead
    base_score -= penalty

    return base_score + priority_boost

# ok naive implementation, we'll see how bad it is

# first just try to put each item in its best timeslot, see if it works
def greedy_schedule(items, timeslots):
    timeslots_used = set()
    schedule = []
    t_score = 0
    used_minutes_by_day = defaultdict(int)

    for item in items:
        best_start = None
        best_score = -1
        timeslots_required = math.ceil(item.duration_with_buffer() / timeslots[0].duration())

        for i in range(len(timeslots) - timeslots_required + 1):
            slot_indices = range(i, i + timeslots_required)

            day = timeslots[i].start.date()
            minutes_scheduled = used_minutes_by_day[day]
            if minutes_scheduled + item.duration() > MAX_MINUTES_PER_DAY:
                continue  # too full

            # Skip if overlap
            if any(idx in timeslots_used for idx in slot_indices):
                continue

            sc = score(item, timeslots[i])
            if sc > best_score:
                best_start = i
                best_score = sc

        if best_start is None:
            # No valid placement found
            return None

        # Assign timeslot indices
        slot_indices = range(best_start, best_start + timeslots_required)
        timeslots_used.update(slot_indices)

        # Calculate actual scheduled time
        start_slot = timeslots[best_start]
        item.calculated_start = start_slot.start + datetime.timedelta(minutes=item.buffer_before)
        item.calculated_end = item.calculated_start + datetime.timedelta(minutes=item.duration())
        schedule.append(item)
        t_score += best_score

    schedule.sort(key=lambda task: task.calculated_start)
    return schedule, t_score

# randomly assign timeslots (really dumb)
def random_schedule(items, timeslots, num_schedules=2):
    return None
    print("Processed with random schedule generator")
    valid_schedules = []

    seen_hashes = set()
    attempts = 0
    max_attempts = 10000  # total tries across all N

    while len(valid_schedules) < num_schedules and attempts < max_attempts:
        attempts += 1
        this_score = 0
        schedule = []
        flag = 0
        timeslots_used_idx = []
        random.shuffle(items)

        for item in items:
            i = random.randint(0, len(timeslots) - 1)
            timeslots_required = math.ceil(item.duration_with_buffer() / timeslots[0].duration())
            if i + timeslots_required > len(timeslots):
                flag = 1
                break
            new_timeslots = [j for j in range(i, i + timeslots_required)]
            if any(j in timeslots_used_idx for j in new_timeslots):
                flag = 1
                break

            timeslots_used_idx.extend(new_timeslots)
            this_score += score(item, timeslots[i])
            item.calculated_start = timeslots[i].start + datetime.timedelta(minutes=item.buffer_before)
            item.calculated_end = item.calculated_start + datetime.timedelta(minutes=item.duration())
            schedule.append(copy.deepcopy(item))

        if flag:
            continue

        # Hash schedule by start times to detect duplicates
        sched_hash = tuple(sorted(item.calculated_start for item in schedule))
        if sched_hash in seen_hashes:
            continue

        seen_hashes.add(sched_hash)
        schedule.sort(key=lambda task: task.calculated_start)
        valid_schedules.append((schedule, this_score))

    return valid_schedules

def smarter_schedule(items, timeslots, num_schedules=2):
    schedules = []
    seen_hashes = set()
    MAX_ATTEMPTS = 200

    # Separate fixed Events from flexible Tasks
    fixed_events = [item for item in items if hasattr(item, "start") and hasattr(item, "end")]
    tasks_only = [item for item in items if not hasattr(item, "start") or not hasattr(item, "end")]

    # Mark which timeslots are blocked by fixed events
    def block_event_slots(event, all_timeslots):
        used_indices = set()
        for idx, slot in enumerate(all_timeslots):
            if not (slot.end <= event.start or slot.start >= event.end):
                used_indices.add(idx)
        return used_indices

    reserved_indices = set()
    for event in fixed_events:
        reserved_indices.update(block_event_slots(event, timeslots))

    for _ in range(MAX_ATTEMPTS):
        scheduled = []
        timeslot_used = set(reserved_indices)
        day_minutes = defaultdict(int)
        total_score = 0

        # Always include fixed events
        for event in fixed_events:
            scheduled.append(copy.deepcopy(event))
            day_minutes[event.start.date()] += int((event.end - event.start).total_seconds() / 60)

        # Sort tasks by priority, flexibility, and age
        random.shuffle(tasks_only)
        tasks_only.sort(key=lambda x: (
            -getattr(x, "priority", 5),
            getattr(x, "flexibility", 0.5),
            -(datetime.datetime.now(ZoneInfo("America/Chicago")).date() -
              (getattr(x, "created_at", None) or datetime.datetime.now(ZoneInfo("America/Chicago"))).date()).days
        ))
        unscheduled = []
        for item in tasks_only:
            slots_per_item = math.ceil(item.duration_with_buffer() / timeslots[0].duration())
            candidate_starts = list(range(len(timeslots) - slots_per_item + 1))
            random.shuffle(candidate_starts)

            placed = False
            for i in candidate_starts:
                block = list(range(i, i + slots_per_item))
                if any(j in timeslot_used for j in block):
                    continue

                start_slot = timeslots[i]
                slot_day = start_slot.start.date()
                total_minutes = day_minutes[slot_day] + item.duration()
                if total_minutes > MAX_MINUTES_PER_DAY:
                    continue

                item_copy = copy.deepcopy(item)
                item_copy.calculated_start = start_slot.start
                item_copy.calculated_end = start_slot.start + datetime.timedelta(minutes=item.duration_with_buffer())

                item_copy.real_start = start_slot.start + datetime.timedelta(minutes=item.buffer_before)
                item_copy.real_end = item_copy.real_start + datetime.timedelta(minutes=item.duration())

                scheduled.append(item_copy)
                for j in block:
                    timeslot_used.add(j)
                day_minutes[slot_day] += item.duration()
                total_score += score(item, start_slot)
                placed = True
                break

            if not placed:
                unscheduled.append(item)
                continue

        if not scheduled:
            continue

        sched_hash = tuple(sorted(
            item.calculated_start if hasattr(item, "calculated_start") else item.start
            for item in scheduled
        ))
        if sched_hash in seen_hashes:
            continue
        seen_hashes.add(sched_hash)

        scheduled.sort(key=lambda t: getattr(t, 'calculated_start', getattr(t, 'start', datetime.datetime.min)))
        schedules.append((scheduled, total_score))
        if len(schedules) >= num_schedules:
            break

    return schedules, unscheduled

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
        start_dt = isoparse(start_str)
    except Exception:
        start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d")
    
    try:
        end_dt = isoparse(end_str)
    except Exception:
        end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d")
    
    # Set default values for fields not provided by Google Calendar.
    # You might adjust these defaults or enhance the conversion based on your app's needs.
    required_focus = 0.5
    category = 'meeting'
    flexibility = 0.0

    # Try to parse metadata from the description field
    description = gcal_event.get('description')
    if description:
        try:
            meta = json.loads(description)
            required_focus = float(meta.get("required_focus", required_focus))
            category = meta.get("category", category)
            flexibility = float(meta.get("flexibility", flexibility))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass  # if parsing fails, fall back to defaults

    
    return Event(
        name=name,
        start=start_dt,
        end=end_dt,
        required_focus=required_focus,
        category=category,
        flexibility=flexibility
    )