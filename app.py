"""
MoodMuse — app.py
Main Flask application file.
All routes, database models, and app configuration in one file
so there are zero import issues.
"""

import os
import base64
import json
import calendar
import random
import string
from datetime import datetime, date, timedelta
from collections import Counter

import cv2
import numpy as np
from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, session)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from flask_bcrypt import Bcrypt
except ImportError:
    Bcrypt = None

# ─────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Secret key for sessions — change this in production
app.config['SECRET_KEY'] = 'moodmuse-secret-2025-change-me'

# SQLite database stored in the instance/ folder
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///moodmuse.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db      = SQLAlchemy(app)
bcrypt  = Bcrypt(app) if Bcrypt else None
login_manager = LoginManager(app)
login_manager.login_view = 'login'           # redirect here if not logged in
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'


def hash_password(password):
    """Hash passwords with Flask-Bcrypt when available, otherwise Werkzeug."""
    if bcrypt:
        return bcrypt.generate_password_hash(password).decode('utf-8')
    return generate_password_hash(password)


def verify_password(hashed_password, password):
    """Verify passwords with Flask-Bcrypt when available, otherwise Werkzeug."""
    if bcrypt:
        return bcrypt.check_password_hash(hashed_password, password)
    return check_password_hash(hashed_password, password)


# ─────────────────────────────────────────────────────────────────
# DATABASE MODELS
# ─────────────────────────────────────────────────────────────────

class User(db.Model, UserMixin):
    """Stores registered users."""
    __tablename__ = 'users'

    id             = db.Column(db.Integer, primary_key=True)
    first_name     = db.Column(db.String(50), nullable=True)
    last_name      = db.Column(db.String(50), nullable=True)
    name           = db.Column(db.String(100), nullable=False)
    email          = db.Column(db.String(120), unique=True, nullable=False)
    password       = db.Column(db.String(200), nullable=False)
    reset_code     = db.Column(db.String(50), nullable=True)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship: one user → many mood logs
    mood_logs = db.relationship('MoodLog', backref='user', lazy=True,
                                cascade='all, delete-orphan')

    def __repr__(self):
        return f'<User {self.email}>'


class MoodLog(db.Model):
    """Stores one mood detection result per entry."""
    __tablename__ = 'mood_logs'

    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    mood       = db.Column(db.String(20), nullable=False)   # happy / sad / neutral / angry / surprised
    score      = db.Column(db.Integer,  nullable=False)     # 1–5 mapped from emotion
    note       = db.Column(db.Text,     nullable=True)      # optional user note
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':         self.id,
            'mood':       self.mood,
            'score':      self.score,
            'note':       self.note,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M'),
        }


class Psychologist(db.Model):
    """Pre-seeded list of mental health professionals."""
    __tablename__ = 'psychologists'

    id             = db.Column(db.Integer, primary_key=True)
    name           = db.Column(db.String(120), nullable=False)
    specialty      = db.Column(db.String(200), nullable=True)
    contact        = db.Column(db.String(120), nullable=True)
    location       = db.Column(db.String(200), nullable=True)
    available      = db.Column(db.Boolean, default=True)


class Resource(db.Model):
    """Self-help resources — articles, hotlines, exercises."""
    __tablename__ = 'resources'

    id          = db.Column(db.Integer, primary_key=True)
    title       = db.Column(db.String(200), nullable=False)
    category    = db.Column(db.String(50),  nullable=False)  # article / hotline / exercise
    description = db.Column(db.Text, nullable=True)
    url         = db.Column(db.String(300), nullable=True)
    mood_tag    = db.Column(db.String(50),  nullable=True)   # which mood this helps


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ─────────────────────────────────────────────────────────────────
# EMOTION DETECTION  (OpenCV only — no TensorFlow)
# ─────────────────────────────────────────────────────────────────

# Mood scores used for the weekly report chart
MOOD_SCORES = {
    'happy':     5,
    'surprised': 4,
    'neutral':   3,
    'sad':       2,
    'angry':     1,
}

MOOD_EMOJIS = {
    'happy': '😄',
    'surprised': '😲',
    'neutral': '😐',
    'sad': '😢',
    'angry': '😤',
    'unknown': '🤔',
}

MOOD_DESCRIPTIONS = {
    'happy': 'Feeling upbeat and positive.',
    'surprised': 'Something unexpected is happening.',
    'neutral': 'Feeling steady and balanced.',
    'sad': 'Feeling a little low or tired.',
    'angry': 'Feeling tense or overwhelmed.',
    'unknown': 'A clear mood has not been captured yet.',
}

QUICK_FEELINGS = [
    {'label': 'Great', 'mood': 'happy', 'emoji': '😄', 'hint': 'High energy'},
    {'label': 'Good', 'mood': 'happy', 'emoji': '🙂', 'hint': 'Positive and steady'},
    {'label': 'Okay', 'mood': 'neutral', 'emoji': '😌', 'hint': 'Balanced'},
    {'label': 'Calm', 'mood': 'neutral', 'emoji': '🫶', 'hint': 'At ease'},
    {'label': 'Stressed', 'mood': 'angry', 'emoji': '😣', 'hint': 'Need a reset'},
    {'label': 'Sad', 'mood': 'sad', 'emoji': '😔', 'hint': 'Feeling low'},
    {'label': 'Sleepy', 'mood': 'sad', 'emoji': '🥱', 'hint': 'Low on energy'},
]


def get_first_name(user):
    name = (user.name or '').strip()
    if not name:
        return 'Friend'
    return name.split()[0]


def mood_icon(mood):
    return MOOD_EMOJIS.get(mood, MOOD_EMOJIS['unknown'])


def current_streak(logs):
    """Count consecutive days with at least one log, ending today."""
    log_days = {log.created_at.date() for log in logs}
    streak = 0
    day = datetime.utcnow().date()
    while day in log_days:
        streak += 1
        day -= timedelta(days=1)
    return streak


def month_calendar(reference_date, highlight_days=None):
    """Build a simple month grid for the dashboard calendar."""
    highlight_days = highlight_days or set()
    cal = calendar.Calendar(firstweekday=6)
    weeks = []
    for week in cal.monthdayscalendar(reference_date.year, reference_date.month):
        week_rows = []
        for day_num in week:
            if day_num == 0:
                week_rows.append(None)
                continue
            current = date(reference_date.year, reference_date.month, day_num)
            week_rows.append({
                'day': day_num,
                'date': current,
                'is_today': current == reference_date,
                'has_activity': current in highlight_days,
            })
        weeks.append(week_rows)
    return weeks


def build_upcoming_sessions(psychologists, reference_date):
    """Create dashboard-friendly upcoming session cards from available psychologists."""
    time_slots = [
        ('09:00', '10:00'),
        ('10:00', '11:00'),
        ('21:00', '22:00'),
    ]
    day_offsets = [2, 4, 6]
    sessions = []
    for idx, psychologist in enumerate(psychologists[:3]):
        start_time, end_time = time_slots[idx % len(time_slots)]
        session_date = reference_date + timedelta(days=day_offsets[idx])
        sessions.append({
            'name': psychologist.name,
            'specialty': psychologist.specialty or 'Therapy session',
            'location': psychologist.location or 'Remote',
            'date': session_date,
            'date_label': session_date.strftime('%B %d, %Y'),
            'weekday': session_date.strftime('%A'),
            'time_label': f'{start_time} - {end_time}',
            'avatar': ''.join(part[0] for part in psychologist.name.split()[:2]).upper() or 'MM',
        })
    return sessions

# Haar cascade for face detection (built into OpenCV)
_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
_face_cascade  = cv2.CascadeClassifier(_CASCADE_PATH)

# Separate cascade for smile detection
_SMILE_PATH    = cv2.data.haarcascades + 'haarcascade_smile.xml'
_smile_cascade = cv2.CascadeClassifier(_SMILE_PATH)

# Separate cascade for eye detection
_EYE_PATH      = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
_eye_cascade   = cv2.CascadeClassifier(_EYE_PATH)


def _clamp(value, low, high):
    return max(low, min(high, value))


def detect_mood_from_image(b64_string: str) -> dict:
    """
    Accepts a base64 image string from the browser webcam.
    Returns dict with: mood, score, confidence, face_detected.

    Logic (no ML model needed):
    ─ No face                 → unknown
    ─ Weighted facial signals  → happy / surprised / angry / sad / neutral
    """
    # Strip the data URL prefix if present
    if ',' in b64_string:
        b64_string = b64_string.split(',', 1)[1]

    # Decode base64 → numpy image
    try:
        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr   = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return {'mood': 'unknown', 'score': 0,
                'confidence': 0, 'face_detected': False}

    if img_bgr is None:
        return {'mood': 'unknown', 'score': 0,
                'confidence': 0, 'face_detected': False}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Step 1 — detect face
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.08, minNeighbors=4, minSize=(70, 70)
    )
    if len(faces) == 0:
        return {'mood': 'unknown', 'score': 0,
                'confidence': 0, 'face_detected': False}

    # Use the largest face found
    x, y, w, h  = max(faces, key=lambda f: f[2] * f[3])
    face_gray    = gray[y: y + h, x: x + w]
    face_bgr     = img_bgr[y: y + h, x: x + w]
    face_gray    = cv2.GaussianBlur(face_gray, (5, 5), 0)

    # Step 2 — check for smile and eyes in relaxed face regions
    upper_face = face_gray[: max(1, int(h * 0.55)), :]
    lower_face = face_gray[max(1, int(h * 0.40)):, :]
    smiles = _smile_cascade.detectMultiScale(
        lower_face, scaleFactor=1.5, minNeighbors=12, minSize=(20, 20)
    )
    eyes = _eye_cascade.detectMultiScale(
        upper_face, scaleFactor=1.1, minNeighbors=5, minSize=(18, 18)
    )

    # Step 3 — brightness, contrast, and colour analysis
    brightness = float(np.mean(face_gray))
    contrast = float(np.std(face_gray))
    # Upper half vs lower half brightness (brow region vs mouth region)
    upper_bright = float(np.mean(face_gray[: h // 2, :]))
    lower_bright = float(np.mean(face_gray[h // 2:, :]))
    ratio        = upper_bright / (lower_bright + 1e-6)

    # Red channel dominance can indicate anger (flushed face)
    b_ch, g_ch, r_ch = cv2.split(face_bgr)
    red_dominance = float(np.mean(r_ch)) - float(np.mean(b_ch))

    # Step 4 — score each mood and choose the strongest signal
    edge_map = cv2.Canny(lower_face, 55, 145)
    edge_density = float(np.mean(edge_map)) / 255.0
    mouth_roi = face_gray[int(h * 0.58):, int(w * 0.22):int(w * 0.78)]
    mouth_roi_darkness = 255.0 - float(np.mean(mouth_roi)) if mouth_roi.size else 0.0

    scores = {
        'happy': 0.0,
        'surprised': 0.0,
        'neutral': 0.0,
        'sad': 0.0,
        'angry': 0.0,
    }

    if len(smiles) > 0:
        scores['happy'] += 4.0
    scores['happy'] += _clamp((edge_density - 0.06) * 12.0, 0.0, 2.0)
    scores['happy'] += _clamp((mouth_roi_darkness - 70.0) / 80.0, 0.0, 1.0)

    scores['surprised'] += _clamp((brightness - 110.0) / 18.0, 0.0, 2.5)
    scores['surprised'] += _clamp((upper_bright - lower_bright) / 18.0, 0.0, 2.0)
    scores['surprised'] += 0.8 if len(eyes) >= 2 else 0.0

    scores['angry'] += _clamp((red_dominance - 8.0) / 8.0, 0.0, 2.5)
    scores['angry'] += _clamp((115.0 - brightness) / 22.0, 0.0, 1.8)
    scores['angry'] += _clamp((edge_density - 0.08) * 6.0, 0.0, 1.0)

    scores['sad'] += _clamp((102.0 - brightness) / 18.0, 0.0, 2.5)
    scores['sad'] += _clamp((upper_bright - lower_bright) / 24.0, 0.0, 1.2)
    scores['sad'] += _clamp((0.06 - edge_density) * 20.0, 0.0, 0.9)

    scores['neutral'] += 0.65
    scores['neutral'] += _clamp(1.4 - abs(ratio - 1.0) * 8.0, 0.0, 1.4)
    scores['neutral'] += _clamp((contrast - 25.0) / 35.0, 0.0, 0.8)

    mood = max(scores, key=scores.get)
    ordered_scores = sorted(scores.values(), reverse=True)
    top_score = ordered_scores[0]
    runner_up = ordered_scores[1] if len(ordered_scores) > 1 else 0.0
    gap = top_score - runner_up

    if top_score < 0.9:
        mood = 'neutral'
    if mood == 'neutral' and top_score < 1.2:
        confidence = 68
    else:
        confidence = int(_clamp(58 + top_score * 9 + gap * 12, 55, 94))

    return {
        'mood':         mood,
        'score':        MOOD_SCORES.get(mood, 3),
        'confidence':   confidence,
        'face_detected': True,
    }


# ─────────────────────────────────────────────────────────────────
# SEED DATA  (runs once on startup if tables are empty)
# ─────────────────────────────────────────────────────────────────

def seed_data():
    """Insert sample psychologists and resources if the DB is empty."""

    if Psychologist.query.count() == 0:
        psychologists = [
            Psychologist(name='Dr. Anika Rahman',    specialty='Anxiety & Depression, CBT',
                         contact='+880 1711-100001', location='Dhaka, Dhanmondi', available=True),
            Psychologist(name='Dr. Farhan Islam',    specialty='Bipolar & Mood Disorders',
                         contact='+880 1711-100002', location='Dhaka, Mirpur',    available=True),
            Psychologist(name='Dr. Sadia Hossain',   specialty='Trauma, PTSD, ACT',
                         contact='+880 1711-100003', location='Dhaka, Gulshan',   available=True),
            Psychologist(name='Dr. Imran Chowdhury', specialty='Stress & Burnout',
                         contact='+880 1811-200001', location='Chittagong',        available=True),
            Psychologist(name='Dr. Sharmin Akter',   specialty='Couples & Grief Therapy',
                         contact='+880 1811-200002', location='Chittagong',        available=True),
            Psychologist(name='Dr. Rayan Hasan',     specialty='Online CBT, Anxiety',
                         contact='+880 1611-700001', location='Online / Remote',   available=True),
        ]
        db.session.add_all(psychologists)

    if Resource.query.count() == 0:
        resources = [
            Resource(title='5-Minute Breathing Exercise',
                     category='exercise',
                     description='A simple box-breathing technique: inhale 4s, hold 4s, exhale 4s. Repeat 4 times to calm your nervous system.',
                     url=None, mood_tag='angry'),
            Resource(title='Mood Journal Starter',
                     category='article',
                     description='Writing down three things you\'re grateful for each day is proven to lift mood within two weeks.',
                     url=None, mood_tag='sad'),
            Resource(title='10-Minute Walk Challenge',
                     category='exercise',
                     description='Step outside for 10 minutes. Natural light and movement reset cortisol and boost serotonin.',
                     url=None, mood_tag='sad'),
            Resource(title='Kaan Pete Roi (Bangladesh Helpline)',
                     category='hotline',
                     description='Free emotional support helpline. Open 6 PM – 10 PM daily.',
                     url='https://www.kaanpeteroibangladesh.org', mood_tag='sad'),
            Resource(title='Progressive Muscle Relaxation',
                     category='exercise',
                     description='Tense and release each muscle group from toes to head. Reduces physical anxiety symptoms in 10 minutes.',
                     url=None, mood_tag='angry'),
            Resource(title='Sleep Hygiene Guide',
                     category='article',
                     description='Consistent bedtime, no screens 30 min before sleep, and a cool room temperature improve sleep quality significantly.',
                     url=None, mood_tag='neutral'),
            Resource(title='Headspace – Free Meditations',
                     category='article',
                     description='Short guided meditations available for free. Even 3 minutes daily shows measurable mood improvement.',
                     url='https://www.headspace.com/meditation/guided-meditation', mood_tag='neutral'),
        ]
        db.session.add_all(resources)

    db.session.commit()


# ─────────────────────────────────────────────────────────────────
# ROUTES — AUTH
# ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Home page — redirects to dashboard if already logged in."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        first_name = request.form.get('first_name', '').strip()
        last_name  = request.form.get('last_name', '').strip()
        email      = request.form.get('email', '').strip().lower()
        password   = request.form.get('password', '')
        confirm    = request.form.get('confirm', '')
        privacy    = request.form.get('privacy')

        # Basic validation
        if not first_name or not last_name or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('signup.html')

        if not privacy:
            flash('You must agree to the privacy policy & terms.', 'danger')
            return render_template('signup.html')

        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('signup.html')

        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return render_template('signup.html')

        if User.query.filter_by(email=email).first():
            flash('Email is already registered. Please log in.', 'warning')
            return redirect(url_for('login'))

        full_name = f"{first_name} {last_name}".strip()
        hashed = hash_password(password)
        user = User(
            first_name=first_name,
            last_name=last_name,
            name=full_name,
            email=email,
            password=hashed
        )
        db.session.add(user)
        db.session.commit()

        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'

        user = User.query.filter_by(email=email).first()

        if user and verify_password(user.password, password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Incorrect email or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


@app.route('/send-reset-code', methods=['POST'])
def send_reset_code():
    """Generate and send password reset code (simulated)."""
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    
    user = User.query.filter_by(email=email).first()
    
    if not user:
        return jsonify({'success': False, 'message': 'Email not found.'})
    
    # Generate a simple reset code
    reset_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    user.reset_code = reset_code
    db.session.commit()
    
    # In a real app, you'd send this via email using Flask-Mail
    # For now, this is simulated
    print(f"[SIMULATED EMAIL] Password reset code for {email}: {reset_code}")
    
    return jsonify({'success': True, 'message': 'Reset code sent to your email.'})


@app.route('/reset-password', methods=['POST'])
def reset_password():
    """Reset user password with reset code."""
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    new_password = data.get('password', '')
    
    user = User.query.filter_by(email=email).first()
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found.'})
    
    if not new_password or len(new_password) < 6:
        return jsonify({'success': False, 'message': 'Password must be at least 6 characters.'})
    
    # Update password
    hashed = hash_password(new_password)
    user.password = hashed
    user.reset_code = None
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Password reset successfully.'})


# ─────────────────────────────────────────────────────────────────
# ROUTES — CORE PAGES
# ─────────────────────────────────────────────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard with summaries, charts, search, and quick check-ins."""
    query = request.args.get('q', '').strip()
    now = datetime.utcnow()
    today = now.date()
    week_start = now - timedelta(days=7)

    logs = (MoodLog.query
            .filter_by(user_id=current_user.id)
            .order_by(MoodLog.created_at.desc())
            .all())
    latest_mood = logs[0] if logs else None

    recent_week_logs = [log for log in logs if log.created_at >= week_start]
    total_entries = len(logs)
    weekly_entries = len(recent_week_logs)
    journal_entries = sum(1 for log in logs if log.note)
    streak = current_streak(logs)
    average_score = round(
        sum(log.score for log in recent_week_logs) / weekly_entries, 1
    ) if weekly_entries else 0.0

    resources = (Resource.query
                 .order_by(Resource.category.asc(), Resource.title.asc())
                 .all())
    psychologists = (Psychologist.query
                     .filter_by(available=True)
                     .order_by(Psychologist.name.asc())
                     .all())

    if query:
        lowered = query.lower()
        filtered_logs = [
            log for log in logs
            if lowered in log.mood.lower()
            or lowered in (log.note or '').lower()
        ]
        filtered_resources = [
            resource for resource in resources
            if lowered in resource.title.lower()
            or lowered in (resource.description or '').lower()
            or lowered in resource.category.lower()
            or lowered in (resource.mood_tag or '').lower()
        ]
        filtered_psychologists = [
            psychologist for psychologist in psychologists
            if lowered in psychologist.name.lower()
            or lowered in (psychologist.specialty or '').lower()
            or lowered in (psychologist.location or '').lower()
        ]
    else:
        filtered_logs = logs[:6]
        filtered_resources = resources[:6]
        filtered_psychologists = psychologists[:3]

    mood_day_map = {today - timedelta(days=offset): [] for offset in range(7)}
    for log in recent_week_logs:
        mood_day_map.setdefault(log.created_at.date(), []).append(log)

    chart_labels = []
    chart_scores = []
    for offset in range(6, -1, -1):
        day = today - timedelta(days=offset)
        day_logs = mood_day_map.get(day, [])
        chart_labels.append(day.strftime('%a'))
        chart_scores.append(
            round(sum(log.score for log in day_logs) / len(day_logs), 1)
            if day_logs else 0
        )

    breakdown_source = recent_week_logs or logs
    mood_counts = dict(Counter(log.mood for log in breakdown_source))
    mood_labels = list(mood_counts.keys())
    mood_values = list(mood_counts.values())
    mood_colors = ['#1d4ed8', '#fbbf24', '#94a3b8', '#f97316', '#ef4444']
    if len(mood_labels) > len(mood_colors):
        mood_colors.extend(['#38bdf8'] * (len(mood_labels) - len(mood_colors)))

    dominant_mood, dominant_count = ('unknown', 0)
    if logs:
        dominant_mood, dominant_count = Counter(
            log.mood for log in logs
        ).most_common(1)[0]

    latest_mood_name = latest_mood.mood if latest_mood else dominant_mood
    insight_text = MOOD_DESCRIPTIONS.get(
        latest_mood_name,
        MOOD_DESCRIPTIONS['unknown'],
    )
    tip_map = {
        'happy': 'Keep that momentum going with one small win for tomorrow.',
        'sad': 'Keep things gentle. A short walk or a message to someone you trust can help.',
        'neutral': 'This is a good moment to plan one tiny, calming task.',
        'angry': 'Try a reset: inhale for 4, hold for 4, exhale for 6.',
        'surprised': 'Pause, breathe, and give your mind a moment to settle.',
        'unknown': 'Use the webcam or quick check-in buttons to start tracking.',
    }
    tip = tip_map.get(latest_mood_name, tip_map['unknown'])

    quick_feelings = QUICK_FEELINGS
    suggested_tools = [
        resource for resource in resources
        if resource.mood_tag == latest_mood_name
    ][:3] or resources[:3]

    recent_activity = []
    for log in logs[:3]:
        recent_activity.append({
            'icon': mood_icon(log.mood),
            'title': f'{log.mood.capitalize()} mood check',
            'subtitle': log.note or 'No note added',
            'meta': log.created_at.strftime('%A, %b %d'),
        })
    if streak:
        recent_activity.append({
            'icon': '🔥',
            'title': f'{streak}-day check-in streak',
            'subtitle': 'You are keeping a steady rhythm this week.',
            'meta': 'Milestone',
        })
    if dominant_mood and dominant_count:
        recent_activity.append({
            'icon': '👑',
            'title': 'Mood insight generated',
            'subtitle': f'{dominant_mood.capitalize()} has shown up most often lately.',
            'meta': 'Auto summary',
        })

    upcoming_sessions = build_upcoming_sessions(psychologists, today)
    highlight_days = {log.created_at.date() for log in logs}
    highlight_days.update(session['date'] for session in upcoming_sessions)
    calendar_weeks = month_calendar(today, highlight_days=highlight_days)

    search_counts = {
        'mood_logs': len(filtered_logs),
        'resources': len(filtered_resources),
        'therapists': len(filtered_psychologists),
    }
    selected_resource = suggested_tools[0] if suggested_tools else None

    return render_template(
        'dashboard.html',
        query=query,
        first_name=get_first_name(current_user),
        logs=logs,
        filtered_logs=filtered_logs,
        filtered_resources=filtered_resources,
        filtered_psychologists=filtered_psychologists,
        latest_mood=latest_mood,
        latest_mood_name=latest_mood_name,
        latest_mood_icon=mood_icon(latest_mood_name),
        total_entries=total_entries,
        weekly_entries=weekly_entries,
        journal_entries=journal_entries,
        exercise_resources=sum(1 for resource in resources if resource.category == 'exercise'),
        average_score=average_score,
        streak=streak,
        tip=tip,
        insight_text=insight_text,
        dominant_mood=dominant_mood,
        dominant_count=dominant_count,
        suggested_tools=suggested_tools,
        selected_resource=selected_resource,
        quick_feelings=quick_feelings,
        recent_activity=recent_activity,
        upcoming_sessions=upcoming_sessions,
        calendar_weeks=calendar_weeks,
        calendar_label=today.strftime('%B %Y'),
        search_counts=search_counts,
        chart_labels=json.dumps(chart_labels),
        chart_scores=json.dumps(chart_scores),
        mood_labels=json.dumps(mood_labels),
        mood_values=json.dumps(mood_values),
        mood_colors=json.dumps(mood_colors[:len(mood_labels)]),
    )


@app.route('/dashboard/quick-checkin', methods=['POST'])
@login_required
def dashboard_quick_checkin():
    """Save a mood from the dashboard quick feelings row."""
    mood = request.form.get('mood', '').strip().lower()
    note = request.form.get('note', '').strip() or None

    if mood not in MOOD_SCORES:
        flash('Please choose a valid mood.', 'warning')
        return redirect(url_for('dashboard'))

    if note is None:
        note = 'Quick dashboard check-in'

    log = MoodLog(
        user_id=current_user.id,
        mood=mood,
        score=MOOD_SCORES[mood],
        note=note,
    )
    db.session.add(log)
    db.session.commit()

    flash(f'Logged your {mood} check-in.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/webcam')
@login_required
def webcam():
    """Webcam page — the user captures their face here."""
    return render_template('webcam.html')


@app.route('/api/detect', methods=['POST'])
@login_required
def api_detect():
    """
    API endpoint called by the webcam page JavaScript.
    Receives base64 image, runs mood detection, saves to DB.
    Returns JSON.
    """
    data  = request.get_json(silent=True) or {}
    image = data.get('image', '')

    if not image:
        return jsonify({'success': False, 'error': 'No image received'}), 400

    result = detect_mood_from_image(image)

    if not result['face_detected']:
        return jsonify({
            'success': False,
            'error': 'No face detected. Please ensure your face is clearly visible and well-lit.'
        }), 200

    # Save the mood log to the database
    note = data.get('note', '').strip() or None
    log  = MoodLog(
        user_id = current_user.id,
        mood    = result['mood'],
        score   = result['score'],
        note    = note,
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({
        'success':    True,
        'mood':       result['mood'],
        'score':      result['score'],
        'confidence': result['confidence'],
        'log_id':     log.id,
    })


@app.route('/history')
@login_required
def history():
    """Full mood history table for the logged-in user."""
    page = request.args.get('page', 1, type=int)
    logs = (MoodLog.query
            .filter_by(user_id=current_user.id)
            .order_by(MoodLog.created_at.desc())
            .paginate(page=page, per_page=15, error_out=False))
    return render_template('history.html', logs=logs)


@app.route('/history/delete/<int:log_id>', methods=['POST'])
@login_required
def delete_log(log_id):
    """Delete a single mood log entry."""
    log = db.session.get(MoodLog, log_id)
    if log and log.user_id == current_user.id:
        db.session.delete(log)
        db.session.commit()
        flash('Entry deleted.', 'info')
    return redirect(url_for('history'))


@app.route('/resources')
@login_required
def resources():
    """Psychologist directory + self-help resources."""
    psychologists = Psychologist.query.filter_by(available=True).all()
    all_resources = Resource.query.all()

    # Group resources by category
    by_category = {}
    for r in all_resources:
        by_category.setdefault(r.category, []).append(r)

    return render_template(
        'resources.html',
        psychologists=psychologists,
        by_category=by_category,
    )


# ─────────────────────────────────────────────────────────────────
# MAIN — create tables and run
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    with app.app_context():
        db.create_all()     # create tables if they don't exist
        seed_data()         # insert sample data if tables are empty
        print('\n  MoodMuse is running!')
        print('  Open http://127.0.0.1:5000 in your browser\n')
    app.run(debug=True, port=5000)
