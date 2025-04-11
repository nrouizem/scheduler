import os
import json
import datetime
import copy
from zoneinfo import ZoneInfo
from dateutil.parser import isoparse
from config import DAY_START_HOUR, DAY_END_HOUR
from storage import save_task, load_tasks
from models import init_db, SessionLocal, TaskDB
from matplotlib.figure import Figure
import io
from collections import defaultdict
from flask import Flask, redirect, url_for, session, request, render_template, flash, jsonify, Response
from dataclasses import asdict
from google_auth_oauthlib.flow import Flow
import google.oauth2.credentials
import googleapiclient.discovery
from googleapiclient.discovery import build
from functions import *


init_db()

def write_credentials_file():
    google_creds = os.environ.get('GOOGLE_CREDENTIALS')
    if google_creds is None:
        raise Exception("GOOGLE_CREDENTIALS environment variable is not set")
    try:
        creds_data = json.loads(google_creds)
    except json.JSONDecodeError as e:
        raise Exception("Invalid JSON in GOOGLE_CREDENTIALS environment variable") from e
    # Write the JSON data to a file named 'credentials.json'
    with open('credentials.json', 'w') as f:
        json.dump(creds_data, f, indent=4)
    print("credentials.json file created successfully.")

# Call the function immediately when the app starts.
write_credentials_file()


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
app.permanent_session_lifetime = datetime.timedelta(days=30)  # Adjust lifetime as needed

@app.before_request
def make_session_permanent():
    session.permanent = True

# Path to your OAuth 2.0 credentials file downloaded from Google Cloud Console.
CLIENT_SECRETS_FILE = "credentials.json"

# This scope allows read/write access to the authenticated user's calendar.
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid"
]

def credentials_to_dict(credentials):
    """Converts credentials to a serializable dictionary."""
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

@app.route("/")
def index():
    if "credentials" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/authorize")
def authorize():
    # Create an OAuth flow instance and redirect the user to the Google OAuth 2.0 consent screen.
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for('oauth2callback', _external=True)
    )
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true'
    )
    session['state'] = state
    return redirect(authorization_url)

@app.route("/oauth2callback")
def oauth2callback():
    # Exchange the authorization code for a token.
    state = session.get('state')
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        state=state,
        redirect_uri=url_for('oauth2callback', _external=True)
    )
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials
    session['credentials'] = credentials_to_dict(credentials)

    # Create a service using credentials
    people_service = build("people", "v1", credentials=credentials)

    # Fetch the authenticated user's primary email
    profile = people_service.people().get(resourceName="people/me", personFields="emailAddresses").execute()
    email = profile.get("emailAddresses", [{}])[0].get("value", "")
    session["user_email"] = email

    flash("Successfully authenticated with Google Calendar!")
    return redirect(url_for('dashboard'))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/dashboard")
def dashboard():
    if 'credentials' not in session:
        return redirect(url_for('index'))
    
    db = SessionLocal()
    tasks = db.query(TaskDB).all()
    db.close()
    
    # all events
    # Use stored credentials to build the Google Calendar service.
    credentials = google.oauth2.credentials.Credentials(**session['credentials'])
    service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)
    events_result = service.events().list(calendarId='primary', maxResults=10).execute()
    events = events_result.get('items', [])

    # today's events
    tz = ZoneInfo("America/Chicago")

    # Define the specific day (e.g., today) and make it timezone-aware
    now = datetime.datetime.now(tz)
    specific_day = now.replace(hour=0, minute=0, second=0, microsecond=0)


    # Calculate the start and end of the day
    time_min = specific_day.isoformat()
    time_max = (specific_day + datetime.timedelta(days=1)).isoformat()

    # Retrieve events within that time window
    today_events_result = service.events().list(
        calendarId='primary',
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,  # expands recurring events into individual events
        orderBy='startTime'
    ).execute()
    today_events = today_events_result.get('items', [])

    time_min = (specific_day + datetime.timedelta(days=1)).isoformat()

    # Retrieve events within that time window
    not_today_events_result = service.events().list(
        calendarId='primary',
        timeMin=time_min,
        singleEvents=True,  # expands recurring events into individual events
        orderBy='startTime'
    ).execute()
    not_today_events = not_today_events_result.get('items', [])

    # Update the session credentials (in case they were refreshed).
    session['credentials'] = credentials_to_dict(credentials)
    return render_template("dashboard.html", events=events, today_events=today_events, not_today_events=not_today_events, tasks=tasks)

@app.route("/add_event", methods=["GET", "POST"])
def add_event():
    if 'credentials' not in session:
        return redirect(url_for('index'))
    if request.method == "POST":
        # Get event details from the form.
        summary = request.form.get("summary")
        start_time_raw = request.form.get("start_time")  # e.g., "2025-04-10T09:00"
        end_time_raw = request.form.get("end_time")      # e.g., "2025-04-10T10:00"

        # Convert the raw strings to datetime objects
        start_dt = datetime.datetime.strptime(start_time_raw, "%Y-%m-%dT%H:%M").replace(tzinfo=ZoneInfo("America/Chicago"))
        end_dt = datetime.datetime.strptime(end_time_raw, "%Y-%m-%dT%H:%M").replace(tzinfo=ZoneInfo("America/Chicago"))

        # Format datetime to include seconds (RFC3339-compliant, without timezone offset)
        start_time_formatted = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        end_time_formatted = end_dt.strftime("%Y-%m-%dT%H:%M:%S")

        # Get metadata from the form
        required_focus = float(request.form.get("required_focus", 0.5))
        category = request.form.get("category", "meeting")
        flexibility = float(request.form.get("flexibility", 0.0))

        # Create the metadata dict to store in description
        metadata = {
            "required_focus": required_focus,
            "category": category,
            "flexibility": flexibility
        }

        event = {
            'summary': summary,
            'start': {'dateTime': start_time_formatted, 'timeZone': 'America/Chicago'},
            'end': {'dateTime': end_time_formatted, 'timeZone': 'America/Chicago'},
            'description': json.dumps(metadata)  # store as a JSON string
        }

        credentials = google.oauth2.credentials.Credentials(**session['credentials'])
        service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)
        service.events().insert(calendarId='primary', body=event).execute()
        flash("Event added successfully!")
        return redirect(url_for('dashboard'))
    return render_template("add_event.html")

@app.route("/add_task", methods=["GET", "POST"])
def add_task():
    if request.method == "POST":
        task = Task(
            name=request.form["name"],
            estimated_duration=int(request.form["duration"]),
            required_focus=float(request.form.get("required_focus", 0.5)),
            category=request.form.get("category", "general"),
            flexibility=float(request.form.get("flexibility", 0.5)),
            priority=int(request.form.get("priority", 3)),
            recurrence=request.form.get("recurrence", "none")
        )
        save_task(task)
        flash("Task added!")
        return redirect(url_for("dashboard"))
    
    return render_template("add_task.html")

@app.route("/process_schedule")
def process_schedule():
    if 'credentials' not in session:
        return redirect(url_for('index'))
    
    NUM_DAYS = 3  # Look ahead N days

    tz = ZoneInfo("America/Chicago")

    # Define the specific day (e.g., today) and make it timezone-aware
    now = datetime.datetime.now(tz)
    specific_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Calculate the start and end of the day
    time_min = specific_day.isoformat()
    time_max = (specific_day + datetime.timedelta(days=NUM_DAYS)).isoformat()

    # Retrieve events within that time window
    credentials = google.oauth2.credentials.Credentials(**session['credentials'])
    service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)
    events_result = service.events().list(
        calendarId='primary',
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,  # expands recurring events into individual events
        orderBy='startTime'
    ).execute()
    gcal_events = events_result.get('items', [])
        
    # Convert Google Calendar events to internal Event objects.
    internal_events = [convert_gcal_event(event) for event in gcal_events]

    # generate timeslots
    today = datetime.datetime.now(ZoneInfo("America/Chicago"))
    slot_duration = 15  # minutes

    all_timeslots = []
    timeslots_by_day = defaultdict(list)

    for offset in range(NUM_DAYS):
        day = today + datetime.timedelta(days=offset)
        day_start = day.replace(hour=DAY_START_HOUR, minute=0, second=0, microsecond=0)
        day_end = day.replace(hour=DAY_END_HOUR, minute=0, second=0, microsecond=0)
        
        day_slots = generate_timeslots(day_start, day_end, slot_duration, get_focus_level, get_weather)
        for slot in day_slots:
            timeslots_by_day[slot.start.date()].append(slot)
        all_timeslots.extend(day_slots)

    # add recurring tasks
    dates_in_range = [today + datetime.timedelta(days=i) for i in range(NUM_DAYS)]

    all_tasks = load_tasks()
    scheduled_tasks = []

    for task in all_tasks:
        if task.recurrence == "none":
            scheduled_tasks.append(task)
        elif task.recurrence == "daily":
            for day in dates_in_range:
                new_task = copy.deepcopy(task)
                new_task.name += f" ({day.strftime('%a')})"
                scheduled_tasks.append(new_task)
        elif task.recurrence.startswith("weekly:"):
            weekday = int(task.recurrence.split(":")[1])
            for day in dates_in_range:
                if day.weekday() == weekday:
                    new_task = copy.deepcopy(task)
                    new_task.name += f" ({day.strftime('%a')})"
                    scheduled_tasks.append(new_task)

    task_objs = scheduled_tasks

    # Merge calendar events + tasks
    items_to_schedule = internal_events + task_objs

    greedy_attempt = greedy_schedule(copy.deepcopy(items_to_schedule), all_timeslots)
    if greedy_attempt:
        print("Used greedy schedule as fallback")
        items_to_schedule = greedy_attempt[0]

    smart_results, smart_unscheduled = smarter_schedule(copy.deepcopy(items_to_schedule), all_timeslots, num_schedules=2)

    schedules = {
        "ortools": schedule(copy.deepcopy(items_to_schedule), all_timeslots)[0],
        "smart1": smart_results[0][0] if len(smart_results) > 0 else [],
        "smart2": smart_results[1][0] if len(smart_results) > 1 else [],
        "unscheduled": smart_unscheduled
    }
    
    def serialize_item(item):
        d = item.__dict__.copy()
        for key in ["calculated_start", "calculated_end", "real_start", "real_end", "start", "end"]:
            if key in d and isinstance(d[key], datetime.datetime):
                d[key] = d[key].isoformat()
        return d

    session["last_schedule_ortools"] = [item.__dict__ for item in schedules["ortools"]]
    session["last_schedule_smart1"] = [serialize_item(item) for item in schedules["smart1"]]
    session["last_schedule_smart2"] = [serialize_item(item) for item in schedules["smart2"]]
    
    # Render a new page to display the optimal schedule.
    return render_template("schedules.html", schedules=schedules)

@app.template_filter('datetimeformat')
def datetimeformat(value):
    """
    Convert an ISO datetime string to a nicer format.
    If the event is today, return "Today HH:MM AM/PM".
    If it's tomorrow, return "Tomorrow HH:MM AM/PM".
    Otherwise, return "DayOfWeek HH:MM AM/PM" (e.g., "Monday 08:00 AM").
    """
    try:
        dt = datetime.datetime.fromisoformat(value)
    except ValueError:
        # If conversion fails, return the original value.
        return value

    # Use the event's tzinfo if available, else local time.
    tz = ZoneInfo("America/Chicago")
    now = datetime.datetime.now(tz)

    if dt.date() == now.date():
        day_str = ""        # just display the time if it's today
    elif dt.date() == (now + datetime.timedelta(days=1)).date():
        day_str = "Tomorrow"
    else:
        day_str = dt.strftime('%A')  # e.g., "Monday"

    time_str = dt.strftime('%I:%M %p').lstrip('0')
    return f"{day_str} {time_str}"

@app.context_processor
def inject_schedule_bounds():
    return dict(
        SCHEDULE_START=DAY_START_HOUR,
        SCHEDULE_END=DAY_END_HOUR
    )

@app.route("/delete_task/<int:task_id>")
def delete_task(task_id):
    db = SessionLocal()
    task = db.query(TaskDB).filter(TaskDB.id == task_id).first()
    if task:
        db.delete(task)
        db.commit()
        flash(f"Deleted task: {task.name}")
    else:
        flash("Task not found.")
    db.close()
    return redirect(url_for("dashboard"))

@app.route("/timeslot_quality_plot")
def timeslot_quality_plot():
    tz = ZoneInfo("America/Chicago")
    now = datetime.datetime.now(tz)
    start = now.replace(hour=DAY_START_HOUR, minute=0, second=0, microsecond=0)
    end = now.replace(hour=DAY_END_HOUR, minute=0, second=0, microsecond=0)

    timeslots = generate_timeslots(start, end, 30, get_focus_level, get_weather)

    focus_scores = [slot.focus_level for slot in timeslots]
    weather_scores = [score_weather(Task(name="", estimated_duration=30, required_focus=0.5, category="gardening", flexibility=0.5), slot) for slot in timeslots]
    labels = [slot.start.strftime('%I:%M %p') for slot in timeslots]

    fig = Figure(figsize=(10, 4))
    ax = fig.subplots()
    ax.plot(labels, focus_scores, label="Focus")
    ax.plot(labels, weather_scores, label="Weather")
    composite_scores = [(f + w)/2 for f, w in zip(focus_scores, weather_scores)]
    ax.plot(labels, composite_scores, label="Composite", linestyle='--')
    ax.set_xticks(labels[::4])  # reduce label clutter
    ax.set_ylim(0, 1)
    ax.set_title("Focus vs. Weather Score per Timeslot")
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")

@app.route("/commit_schedule/<string:which>", methods=["POST"])
def commit_schedule(which):
    if "credentials" not in session:
        return redirect(url_for("index"))

    credentials = google.oauth2.credentials.Credentials(**session['credentials'])
    service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)

    # Determine schedule to push
    schedule_map = {
        "ortools": session.get("last_schedule_ortools", []),
        "random1": session.get("last_schedule_random1", []),
        "random2": session.get("last_schedule_random2", [])
    }
    selected_schedule = schedule_map.get(which, [])

    # Delete previously generated events
    now = datetime.datetime.now(ZoneInfo("America/Chicago"))
    time_min = now.isoformat()
    time_max = (now + datetime.timedelta(days=3)).isoformat()

    events_result = service.events().list(calendarId='primary', timeMin=time_min, timeMax=time_max).execute()
    existing = events_result.get("items", [])
    for evt in existing:
        if evt.get("description", "").startswith("generated-by:smart-scheduler"):
            service.events().delete(calendarId='primary', eventId=evt["id"]).execute()

    # Add new events
    for item in selected_schedule:
        event = {
            'summary': item["name"],
            'start': {'dateTime': item["calculated_start"], 'timeZone': 'America/Chicago'},
            'end': {'dateTime': item["calculated_end"], 'timeZone': 'America/Chicago'},
            'description': 'generated-by:smart-scheduler'
        }
        service.events().insert(calendarId='primary', body=event).execute()

    flash("Schedule pushed to Google Calendar!")
    return redirect(url_for("dashboard"))

@app.route("/undo_schedule", methods=["POST"])
def undo_schedule():
    if "credentials" not in session:
        return redirect(url_for("index"))

    credentials = google.oauth2.credentials.Credentials(**session['credentials'])
    service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)

    now = datetime.datetime.now(ZoneInfo("America/Chicago"))
    time_min = now.isoformat()
    time_max = (now + datetime.timedelta(days=3)).isoformat()

    # Find all events your app created
    events_result = service.events().list(
        calendarId='primary',
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True
    ).execute()

    for event in events_result.get("items", []):
        if event.get("description", "").startswith("generated-by:smart-scheduler"):
            service.events().delete(calendarId='primary', eventId=event["id"]).execute()

    flash("Schedule removed from Google Calendar.")
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    app.run(debug=True)
