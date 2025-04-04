import os
import json
import datetime
import pytz
import copy

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

from flask import Flask, redirect, url_for, session, request, render_template, flash, jsonify
from dataclasses import asdict
from google_auth_oauthlib.flow import Flow
import google.oauth2.credentials
import googleapiclient.discovery
from functions import *

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
app.permanent_session_lifetime = datetime.timedelta(days=30)  # Adjust lifetime as needed

@app.before_request
def make_session_permanent():
    session.permanent = True

# Path to your OAuth 2.0 credentials file downloaded from Google Cloud Console.
CLIENT_SECRETS_FILE = "credentials.json"

# This scope allows read/write access to the authenticated user's calendar.
SCOPES = ['https://www.googleapis.com/auth/calendar']

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
    flash("Successfully authenticated with Google Calendar!")
    return redirect(url_for('dashboard'))

@app.route("/dashboard")
def dashboard():
    if 'credentials' not in session:
        return redirect(url_for('index'))
    
    # all events
    # Use stored credentials to build the Google Calendar service.
    credentials = google.oauth2.credentials.Credentials(**session['credentials'])
    service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)
    events_result = service.events().list(calendarId='primary', maxResults=10).execute()
    events = events_result.get('items', [])

    # today's events
    tz = pytz.timezone("America/Chicago")

    # Define the specific day (e.g., today) and make it timezone-aware
    specific_day = tz.localize(datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0))

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
    return render_template("dashboard.html", events=events, today_events=today_events, not_today_events=not_today_events)

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
        start_dt = datetime.datetime.fromisoformat(start_time_raw)
        end_dt = datetime.datetime.fromisoformat(end_time_raw)

        # Format datetime to include seconds (RFC3339-compliant, without timezone offset)
        start_time_formatted = start_dt.strftime("%Y-%m-%dT%H:%M:%S")
        end_time_formatted = end_dt.strftime("%Y-%m-%dT%H:%M:%S")

        # Create the event object.
        event = {
            'summary': summary,
            'start': {'dateTime': start_time_formatted, 'timeZone': 'America/Chicago'},
            'end': {'dateTime': end_time_formatted, 'timeZone': 'America/Chicago'},
        }

        credentials = google.oauth2.credentials.Credentials(**session['credentials'])
        service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)
        service.events().insert(calendarId='primary', body=event).execute()
        flash("Event added successfully!")
        return redirect(url_for('dashboard'))
    return render_template("add_event.html")

@app.route("/process_schedule")
def process_schedule():
    if 'credentials' not in session:
        return redirect(url_for('index'))
    
    tz = pytz.timezone("America/Chicago")

    # Define the specific day (e.g., today) and make it timezone-aware
    specific_day = tz.localize(datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0))

    # Calculate the start and end of the day
    time_min = specific_day.isoformat()
    time_max = (specific_day + datetime.timedelta(days=1)).isoformat()

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
    today = datetime.datetime.now(zoneinfo.ZoneInfo("America/Chicago"))
    day_start = today.replace(hour=8, minute=0, second=0, microsecond=0)
    day_end = today.replace(hour=18, minute=0, second=0, microsecond=0)
    slot_duration = 15  # minutes
    timeslots = generate_timeslots(day_start, day_end, slot_duration, get_focus_level, get_weather)
    for event in internal_events:
        print(event)
    optimal_schedule = schedule(copy.deepcopy(internal_events), timeslots)[0]
    schedules = {
        "ortools": optimal_schedule,
        "random1": random_schedule(copy.deepcopy(internal_events), timeslots)[0],
        "random2": random_schedule(copy.deepcopy(internal_events), timeslots)[0]
    }
    
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
    now = datetime.datetime.now(dt.tzinfo) if dt.tzinfo else datetime.datetime.now()

    if dt.date() == now.date():
        day_str = ""        # just display the time if it's today
    elif dt.date() == (now + datetime.timedelta(days=1)).date():
        day_str = "Tomorrow"
    else:
        day_str = dt.strftime('%A')  # e.g., "Monday"

    time_str = dt.strftime('%I:%M %p').lstrip('0')
    return f"{day_str} {time_str}"

if __name__ == "__main__":
    # For local testing
    app.run(debug=True)
