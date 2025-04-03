from flask import Flask, redirect, url_for, session, request, render_template, flash
from google_auth_oauthlib.flow import Flow
import google.oauth2.credentials
import googleapiclient.discovery
import os

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")

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
    # Use stored credentials to build the Google Calendar service.
    credentials = google.oauth2.credentials.Credentials(**session['credentials'])
    service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)
    events_result = service.events().list(calendarId='primary', maxResults=10).execute()
    events = events_result.get('items', [])
    # Update the session credentials (in case they were refreshed).
    session['credentials'] = credentials_to_dict(credentials)
    return render_template("dashboard.html", events=events)

@app.route("/add_event", methods=["GET", "POST"])
def add_event():
    if 'credentials' not in session:
        return redirect(url_for('index'))
    if request.method == "POST":
        # Get event details from the form.
        summary = request.form.get("summary")
        start_time = request.form.get("start_time")
        end_time = request.form.get("end_time")
        # Create the event object.
        event = {
            'summary': summary,
            'start': {'dateTime': start_time, 'timeZone': 'America/Los_Angeles'},
            'end': {'dateTime': end_time, 'timeZone': 'America/Los_Angeles'},
        }
        credentials = google.oauth2.credentials.Credentials(**session['credentials'])
        service = googleapiclient.discovery.build('calendar', 'v3', credentials=credentials)
        service.events().insert(calendarId='primary', body=event).execute()
        flash("Event added successfully!")
        return redirect(url_for('dashboard'))
    return render_template("add_event.html")

if __name__ == "__main__":
    # For local testing
    app.run(debug=True)
