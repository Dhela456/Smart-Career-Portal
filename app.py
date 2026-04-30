import datetime
import os
import json
import sqlite3
from collections import defaultdict

import bcrypt
import numpy as np
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
import pdfplumber
import re


# Load environment variables from .env file
load_dotenv()
    
@st.fragment(run_every=30)  # Runs every 30 seconds to keep WebSocket alive
def keep_alive():
    """Prevents sudden logout by maintaining active connection"""
    st.empty()  # Invisible element that keeps the session alive

# ====================== GEMINI CONFIG ======================
GEMINI_API_KEY = os.getenv("GO--OGLE_API_KEY")
gemini_client  = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL   = "gemini-2.5-flash"

# ====================== ML CONFIG ======================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml", "models")
ML_READY  = False
ml_models = {}

try:
    import joblib
    _req = ["xgb_model.pkl","scaler.pkl","le_career.pkl","feature_names.pkl",
            "le_class_level.pkl","le_department.pkl","le_strength_level.pkl",
            "le_performance_trend.pkl","le_best_subject.pkl","le_weak_subject.pkl"]
    if all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in _req):
        ml_models = {
            "xgb":          joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl")),
            "scaler":       joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
            "le_career":    joblib.load(os.path.join(MODEL_DIR, "le_career.pkl")),
            "le_class":     joblib.load(os.path.join(MODEL_DIR, "le_class_level.pkl")),
            "le_dept":      joblib.load(os.path.join(MODEL_DIR, "le_department.pkl")),
            "le_strength":  joblib.load(os.path.join(MODEL_DIR, "le_strength_level.pkl")),
            "le_trend":     joblib.load(os.path.join(MODEL_DIR, "le_performance_trend.pkl")),
            "le_best":      joblib.load(os.path.join(MODEL_DIR, "le_best_subject.pkl")),
            "le_weak":      joblib.load(os.path.join(MODEL_DIR, "le_weak_subject.pkl")),
            "feature_names":joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl")),
        }
        ML_READY = True
except Exception:
    pass

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="Smart Career Portal", page_icon="🎓", layout="wide")

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    body { background-color: #f5f6fa; }
    .container {
        max-width: 420px; margin: auto; padding: 15px;
        background: whhite ; border-radius: 16px;
        box-shadow: 0px 6px 24px rgba(0,0,0,0.10);
    }
    .title   { font-size: 30px; font-weight: 800; text-align: center; margin-bottom: 8px; }
    .subtitle{ font-size: 14px; color: gray; text-align: center; margin-bottom: 20px; }
    .caption { font-size: 12px; text-align: center; color: gray; margin-bottom: 30px; }
    .stTextInput>div>div>input,
    .stSelectbox>div>div,
    .stDateInput>div>div { border-radius: 10px; padding: 10px; }
    button[kind="primary"] {
        border-radius: 10px; background-color: #2d6cdf;
        color: white; font-weight: bold;
    }
    .footer { text-align: center; font-size: 13px; margin-top: 15px; }

    /* ── Test tab ─────────────────────────────────────── */
    .test-progress-bar {
        background:#e0e7ff; border-radius:10px; height:12px;
        margin-bottom:8px; overflow:hidden;
    }
    .test-progress-fill {
        background:linear-gradient(90deg,#2d6cdf,#5b8dee);
        height:100%; border-radius:10px; transition:width 0.4s ease;
    }
    .test-header {
        background:linear-gradient(135deg,#2d6cdf 0%,#1a3c8f 100%);
        color:white; padding:18px 24px; border-radius:14px; margin-bottom:20px;
    }
    .test-header h2 { margin:0; font-size:20px; }
    .test-header p  { margin:4px 0 0; font-size:13px; opacity:0.85; }
    .q-card {
        background:white; border-radius:12px; border-left:5px solid #2d6cdf;
        padding:16px 20px; margin-bottom:14px;
        box-shadow:0 2px 10px rgba(0,0,0,0.06);
    }
    .q-num  { font-size:11px; font-weight:700; color:#2d6cdf;
              text-transform:uppercase; letter-spacing:0.05em; }
    .q-text { font-size:15px; font-weight:600; color:#1a1a2e; margin-top:4px; }
    .completed-badge {
        display:inline-block; background:#d1fae5; color:#065f46;
        border-radius:20px; padding:4px 14px; font-size:13px; font-weight:600; margin:4px;
    }
    .locked-badge {
        display:inline-block; background:#fef3c7; color:#92400e;
        border-radius:20px; padding:4px 14px; font-size:13px; font-weight:600; margin:4px;
    }
    .active-badge {
        display:inline-block; background:#dbeafe; color:#1e40af;
        border-radius:20px; padding:4px 14px; font-size:13px; font-weight:600; margin:4px;
    }

    /* ── Recommendations ─────────────────────────────── */
    .rec-hero {
        background:linear-gradient(135deg,#1a3c8f 0%,#2d6cdf 60%,#5b8dee 100%);
        color:white; padding:28px 32px; border-radius:18px; margin-bottom:24px; text-align:center;
    }
    .rec-hero h1 { margin:0; font-size:26px; }
    .rec-hero p  { margin:8px 0 0; font-size:14px; opacity:0.9; }
    .uni-card {
        background:#f0f7ff; border-radius:12px; padding:14px 18px;
        margin-bottom:10px; border-left:4px solid #10b981;
    }
    .score-bar-bg {
        background:#e0e7ff; border-radius:8px; height:10px;
        overflow:hidden; margin-top:4px;
    }
    .score-bar-fill {
        background:linear-gradient(90deg,#2d6cdf,#5b8dee);
        height:100%; border-radius:8px;
    }

    /* ── Chatbot ─────────────────────────────────────── */
    .chat-user {
        background:#2d6cdf; color:white; border-radius:14px 14px 4px 14px;
        padding:10px 16px; margin:8px 0 8px auto; max-width:72%;
        font-size:14px; width:fit-content; margin-left:auto;
    }
    .chat-ai {
        background:white; color:#1a1a2e; border-radius:14px 14px 14px 4px;
        padding:10px 16px; margin:8px 0; max-width:82%;
        box-shadow:0 2px 8px rgba(0,0,0,0.08); font-size:14px; width:fit-content;
    }
</style>
""", unsafe_allow_html=True)

# ====================== CONSTANTS ======================
DEPARTMENT_SUBJECTS = {
    "Science":    ["English","Mathematics","Physics","Chemistry","Biology",
                   "Further Mathematics","Agricultural Science","Computer Science","Geography"],
    "Arts":       ["English","Mathematics","Literature in English","Government",
                   "CRS/IRK","History","Economics","Yoruba/Hausa/Igbo","Civic Education"],
    "Commercial": ["English","Mathematics","Economics","Accounting","Commerce",
                   "Business Studies","Government","Office Practice","Insurance"],
}

UNIVERSITY_MAP = {
    "Medicine & Health Sciences": [
        {"name":"University of Lagos (UNILAG)","course":"Medicine & Surgery","cutoff":"280+","location":"Lagos","url":"https://unilag.edu.ng"},
        {"name":"University of Ibadan (UI)",   "course":"Medicine & Surgery","cutoff":"300+","location":"Ibadan, Oyo","url":"https://ui.edu.ng"},
        {"name":"Obafemi Awolowo University",  "course":"Medicine & Surgery","cutoff":"275+","location":"Ile-Ife, Osun","url":"https://oauife.edu.ng"},
        {"name":"Ahmadu Bello University (ABU)","course":"Medicine & Surgery","cutoff":"270+","location":"Zaria, Kaduna","url":"https://abu.edu.ng"},
    ],
    "Engineering & Technology": [
        {"name":"University of Lagos (UNILAG)", "course":"Mechanical/Electrical Engineering","cutoff":"240+","location":"Lagos","url":"https://unilag.edu.ng"},
        {"name":"Ahmadu Bello University (ABU)","course":"Civil/Electrical Engineering","cutoff":"230+","location":"Zaria, Kaduna","url":"https://abu.edu.ng"},
        {"name":"University of Nigeria Nsukka", "course":"Engineering","cutoff":"220+","location":"Nsukka, Enugu","url":"https://unn.edu.ng"},
        {"name":"Covenant University",          "course":"Engineering","cutoff":"220+","location":"Ota, Ogun","url":"https://covenantuniversity.edu.ng"},
    ],
    "Computer Science & IT": [
        {"name":"University of Lagos (UNILAG)","course":"Computer Science","cutoff":"230+","location":"Lagos","url":"https://unilag.edu.ng"},
        {"name":"Covenant University",         "course":"Computer Science","cutoff":"220+","location":"Ota, Ogun","url":"https://covenantuniversity.edu.ng"},
        {"name":"Obafemi Awolowo University",  "course":"Computer Science & Eng.","cutoff":"220+","location":"Ile-Ife, Osun","url":"https://oauife.edu.ng"},
        {"name":"Federal University of Technology Akure (FUTA)","course":"Computer Science","cutoff":"200+","location":"Akure, Ondo","url":"https://futa.edu.ng"},
    ],
    "Agriculture & Environmental Sciences": [
        {"name":"FUNAAB","course":"Agricultural Science","cutoff":"180+","location":"Abeokuta, Ogun","url":"https://funaab.edu.ng"},
        {"name":"Ahmadu Bello University (ABU)","course":"Agriculture","cutoff":"180+","location":"Zaria, Kaduna","url":"https://abu.edu.ng"},
        {"name":"University of Nigeria Nsukka","course":"Agriculture","cutoff":"180+","location":"Nsukka, Enugu","url":"https://unn.edu.ng"},
        {"name":"Michael Okpara University","course":"Agriculture","cutoff":"170+","location":"Umudike, Abia","url":"https://mouau.edu.ng"},
    ],
    "Law & Social Sciences": [
        {"name":"University of Lagos (UNILAG)","course":"Law","cutoff":"250+","location":"Lagos","url":"https://unilag.edu.ng"},
        {"name":"University of Nigeria Nsukka","course":"Law","cutoff":"230+","location":"Nsukka, Enugu","url":"https://unn.edu.ng"},
        {"name":"Obafemi Awolowo University",  "course":"Law","cutoff":"240+","location":"Ile-Ife, Osun","url":"https://oauife.edu.ng"},
        {"name":"University of Ibadan (UI)",   "course":"Sociology / Political Science","cutoff":"200+","location":"Ibadan, Oyo","url":"https://ui.edu.ng"},
    ],
    "Mass Communication & Media": [
        {"name":"University of Lagos (UNILAG)","course":"Mass Communication","cutoff":"200+","location":"Lagos","url":"https://unilag.edu.ng"},
        {"name":"University of Nigeria Nsukka","course":"Mass Communication","cutoff":"180+","location":"Nsukka, Enugu","url":"https://unn.edu.ng"},
        {"name":"Bayero University Kano (BUK)","course":"Mass Communication","cutoff":"180+","location":"Kano","url":"https://buk.edu.ng"},
        {"name":"University of Ibadan (UI)",   "course":"Communication & Language Arts","cutoff":"180+","location":"Ibadan, Oyo","url":"https://ui.edu.ng"},
    ],
    "Education & Humanities": [
        {"name":"University of Nigeria Nsukka","course":"Education","cutoff":"160+","location":"Nsukka, Enugu","url":"https://unn.edu.ng"},
        {"name":"University of Ibadan (UI)",   "course":"Education","cutoff":"170+","location":"Ibadan, Oyo","url":"https://ui.edu.ng"},
        {"name":"Ahmadu Bello University (ABU)","course":"Education","cutoff":"160+","location":"Zaria, Kaduna","url":"https://abu.edu.ng"},
        {"name":"Lagos State University (LASU)","course":"Education","cutoff":"150+","location":"Lagos","url":"https://lasu.edu.ng"},
    ],
    "Business & Finance": [
        {"name":"University of Lagos (UNILAG)","course":"Accounting / Finance","cutoff":"220+","location":"Lagos","url":"https://unilag.edu.ng"},
        {"name":"Covenant University",         "course":"Business Administration","cutoff":"200+","location":"Ota, Ogun","url":"https://covenantuniversity.edu.ng"},
        {"name":"Obafemi Awolowo University",  "course":"Accounting","cutoff":"210+","location":"Ile-Ife, Osun","url":"https://oauife.edu.ng"},
        {"name":"Lagos Business School (PAU)", "course":"Business Studies","cutoff":"220+","location":"Lagos","url":"https://lbs.edu.ng"},
    ],
    "Entrepreneurship & Management": [
        {"name":"Covenant University",         "course":"Business Management","cutoff":"200+","location":"Ota, Ogun","url":"https://covenantuniversity.edu.ng"},
        {"name":"Lagos Business School (PAU)", "course":"Entrepreneurship","cutoff":"220+","location":"Lagos","url":"https://lbs.edu.ng"},
        {"name":"University of Lagos (UNILAG)","course":"Business Administration","cutoff":"210+","location":"Lagos","url":"https://unilag.edu.ng"},
        {"name":"Nile University of Nigeria",  "course":"Management Sciences","cutoff":"180+","location":"Abuja","url":"https://nileuniversity.edu.ng"},
    ],
    "Creative Arts & Design": [
        {"name":"Yaba College of Technology",   "course":"Art & Design","cutoff":"160+","location":"Lagos","url":"https://yabatech.edu.ng"},
        {"name":"Obafemi Awolowo University",   "course":"Fine & Applied Arts","cutoff":"180+","location":"Ile-Ife, Osun","url":"https://oauife.edu.ng"},
        {"name":"University of Nigeria Nsukka", "course":"Fine & Applied Arts","cutoff":"170+","location":"Nsukka, Enugu","url":"https://unn.edu.ng"},
        {"name":"Lagos State University (LASU)","course":"Fine Arts","cutoff":"160+","location":"Lagos","url":"https://lasu.edu.ng"},
    ],
}

# ====================== TEST QUESTIONS ======================
COGNITIVE_QUESTIONS = [
    {"id":"cog_1","text":"If 3 pencils cost ₦45, how much do 7 pencils cost?",
     "options":["₦95","₦105","₦100","₦115"],"correct":1},
    {"id":"cog_2","text":"Which number comes next in the sequence: 2, 6, 18, 54, ___?",
     "options":["108","162","72","216"],"correct":1},
    {"id":"cog_3","text":"A rectangle has length 12 cm and width 5 cm. What is its area?",
     "options":["34 cm²","60 cm²","17 cm²","70 cm²"],"correct":1},
    {"id":"cog_4","text":"If today is Wednesday and a test is in 10 days, what day is the test?",
     "options":["Monday","Friday","Saturday","Sunday"],"correct":3},
    {"id":"cog_5","text":"BOOK is to LIBRARY as PAINTING is to:",
     "options":["Canvas","Museum","Artist","Brush"],"correct":1},
    {"id":"cog_6","text":"Which word does NOT belong: Cat, Dog, Eagle, Rabbit?",
     "options":["Cat","Dog","Eagle","Rabbit"],"correct":2},
    {"id":"cog_7","text":"A train travels 240 km in 3 hours. How far does it travel in 5 hours?",
     "options":["360 km","480 km","400 km","300 km"],"correct":2},
    {"id":"cog_8","text":"If ALL doctors are graduates, and Emeka is a doctor, then:",
     "options":["Emeka may not be a graduate","Emeka is definitely a graduate",
                "Emeka is not a graduate","We cannot tell"],"correct":1},
    {"id":"cog_9","text":"What is 15% of 200?",
     "options":["25","30","35","20"],"correct":1},
    {"id":"cog_10","text":"Choose the word with the opposite meaning of ANCIENT:",
     "options":["Old","Historic","Modern","Antique"],"correct":2},
]

APTITUDE_QUESTIONS = [
    {"id":"apt_1","text":"You are given a broken device. Your first instinct is to:",
     "options":["Take it apart to understand how it works","Look up a repair video online",
                "Ask someone with technical knowledge","Buy a new one"],
     "correct":None,"weights":{"Science":[3,2,1,0],"Arts":[1,2,3,0],"Commercial":[1,2,2,1]}},
    {"id":"apt_2","text":"Which activity would you enjoy MOST on a free Saturday?",
     "options":["Conducting a science experiment at home","Writing a short story or poem",
                "Organising a small business selling snacks","Playing a musical instrument"],
     "correct":None,"weights":{"Science":[3,1,1,1],"Arts":[1,3,0,2],"Commercial":[1,0,3,1]}},
    {"id":"apt_3","text":"Your school wants to raise funds. You suggest:",
     "options":["Build a website to accept donations","Write persuasive letters to sponsors",
                "Create a business plan and sell products","Organise a school debate competition"],
     "correct":None,"weights":{"Science":[3,1,2,1],"Arts":[1,3,1,2],"Commercial":[2,1,3,1]}},
    {"id":"apt_4","text":"Which subject do you find most interesting?",
     "options":["Physics / Biology / Chemistry","Literature / History / Government",
                "Economics / Accounting / Commerce","Music / Fine Art / Drama"],
     "correct":None,"weights":{"Science":[3,0,1,0],"Arts":[0,3,1,2],"Commercial":[1,0,3,1]}},
    {"id":"apt_5","text":"When reading a long text, you prefer to:",
     "options":["Draw diagrams and charts to summarise it","Write bullet points and key arguments",
                "Create a table or spreadsheet of key data","Discuss it with classmates"],
     "correct":None,"weights":{"Science":[3,1,2,1],"Arts":[1,3,1,2],"Commercial":[1,1,3,2]}},
    {"id":"apt_6","text":"Which of these jobs sounds most exciting to you?",
     "options":["Engineer or Doctor","Lawyer or Journalist","Banker or Entrepreneur","Teacher or Counsellor"],
     "correct":None,"weights":{"Science":[3,1,1,1],"Arts":[1,3,0,2],"Commercial":[0,1,3,2]}},
    {"id":"apt_7","text":"A classmate is struggling with a problem. You:",
     "options":["Help them figure out the logic step-by-step","Listen and offer emotional support",
                "Help them plan a practical solution","Encourage them with motivational words"],
     "correct":None,"weights":{"Science":[3,1,2,1],"Arts":[1,3,1,2],"Commercial":[1,1,3,2]}},
    {"id":"apt_8","text":"You are strongest at:",
     "options":["Solving maths and science problems quickly","Expressing ideas through words and debate",
                "Managing money and identifying business opportunities","Creating art, music or stories"],
     "correct":None,"weights":{"Science":[3,1,1,1],"Arts":[0,3,0,2],"Commercial":[1,1,3,0]}},
]

PSYCHOMETRIC_QUESTIONS = [
    {"id":"psy_1","text":"I enjoy working in groups and collaborating with others.",
     "options":["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],"trait":"Extraversion"},
    {"id":"psy_2","text":"I like to plan and organise things well in advance.",
     "options":["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],"trait":"Conscientiousness"},
    {"id":"psy_3","text":"I stay calm and focused even when things get difficult.",
     "options":["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],"trait":"Emotional Stability"},
    {"id":"psy_4","text":"I enjoy trying out new ideas and thinking creatively.",
     "options":["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],"trait":"Openness"},
    {"id":"psy_5","text":"I consider how my decisions will affect other people.",
     "options":["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],"trait":"Agreeableness"},
    {"id":"psy_6","text":"I prefer working on one task at a time until it is completed.",
     "options":["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],"trait":"Conscientiousness"},
    {"id":"psy_7","text":"I feel energised when I am around people.",
     "options":["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],"trait":"Extraversion"},
    {"id":"psy_8","text":"I am comfortable leading a group or project.",
     "options":["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"],"trait":"Leadership"},
]

SENTIMENT_QUESTIONS = [
    {"id":"sen_1","text":"How excited are you about your future career?",
     "options":["Not at all","A little","Moderately","Very excited","Extremely excited"]},
    {"id":"sen_2","text":"How confident are you that you will succeed in your chosen career?",
     "options":["Not confident","Slightly confident","Moderately confident","Confident","Very confident"]},
    {"id":"sen_3","text":"How much does your family support your career aspirations?",
     "options":["No support","Little support","Some support","Good support","Full support"]},
    {"id":"sen_4","text":"How would you describe your current academic motivation?",
     "options":["Very low","Low","Average","High","Very high"]},
    {"id":"sen_5","text":"How do you feel when you face a very difficult academic challenge?",
     "options":["Give up easily","Feel discouraged","Push through with difficulty",
                "Stay motivated","Thrive on the challenge"]},
    {"id":"sen_6","text":"How clear is your vision of what career you want?",
     "options":["Completely unclear","Very unclear","Somewhat clear","Mostly clear","Completely clear"]},
]

TEST_META = [
    {"key":"cognitive",    "label":"Cognitive Test",    "icon":"🧩",
     "desc":"Logic, reasoning & problem-solving",  "questions":COGNITIVE_QUESTIONS},
    {"key":"aptitude",     "label":"Aptitude Test",     "icon":"🎯",
     "desc":"Natural talents & subject strengths", "questions":APTITUDE_QUESTIONS},
    {"key":"psychometric", "label":"Psychometric Test", "icon":"🧠",
     "desc":"Personality traits & working style",  "questions":PSYCHOMETRIC_QUESTIONS},
    {"key":"sentiment",    "label":"Sentiment Test",    "icon":"💬",
     "desc":"Attitudes, motivation & mindset",     "questions":SENTIMENT_QUESTIONS},
]

# ====================== DATABASE ======================
def init_db():
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT, dob TEXT, class_level TEXT,
        department TEXT, email TEXT UNIQUE, password TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS academic_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, result_type TEXT, subject TEXT,
        score REAL, exam_date TEXT, uploaded_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS test_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, test_type TEXT, question_id TEXT,
        answer TEXT, score REAL, submitted_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, career_path TEXT, confidence REAL,
        universities TEXT, linkedin_mentors TEXT,
        narrative TEXT, top3 TEXT, generated_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, role TEXT, message TEXT, created_at TEXT)""")
    for col, typ in [("narrative","TEXT"),("top3","TEXT")]:
        try:
            c.execute(f"ALTER TABLE recommendations ADD COLUMN {col} {typ}")
        except Exception:
            pass
    conn.commit()
    conn.close()

def extract_pdf_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text


def parse_results(text):
    results = []
    lines = text.split("\n")

    for line in lines:
        match = re.search(r"([A-Za-z ]+)\s+(\d{1,3})", line)
        if match:
            subject = match.group(1).strip()
            score = float(match.group(2))

            if 0 <= score <= 100:
                results.append((subject, score))

    return results

def hash_password(p):       return bcrypt.hashpw(p.encode(), bcrypt.gensalt())
def check_password(p, h):   return bcrypt.checkpw(p.encode(), h)

def create_admin_user():
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email='Admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users (full_name,dob,class_level,department,email,password) VALUES (?,?,?,?,?,?)",
                  ("Administrator","2000-01-01","Admin",None,"Admin",hash_password("Admin")))
        conn.commit()
        st.toast("✅ Admin account created  (email: Admin / password: Admin)", icon="🔑")
    conn.close()

def create_user(full_name, dob, class_level, department, email, password):
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (full_name,dob,class_level,department,email,password) VALUES (?,?,?,?,?,?)",
                  (full_name, str(dob), class_level, department, email, hash_password(password)))
        conn.commit(); return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(email, password):
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    c.execute("SELECT id,full_name,class_level,department,password FROM users WHERE email=?", (email,))
    user = c.fetchone(); conn.close()
    return user if (user and check_password(password, user[4])) else None


def save_academic_result(user_id, result_type, subject, score, exam_date):
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    c.execute("INSERT INTO academic_results (user_id,result_type,subject,score,exam_date,uploaded_at) VALUES (?,?,?,?,?,?)",
              (user_id, result_type, subject, score, str(exam_date), datetime.datetime.now().isoformat()))
    conn.commit(); conn.close()

def get_user_results(user_id):
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    c.execute("SELECT id,result_type,subject,score,exam_date,uploaded_at FROM academic_results WHERE user_id=? ORDER BY uploaded_at DESC", (user_id,))
    rows = c.fetchall(); conn.close(); return rows

def delete_academic_result(result_id):
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    c.execute("DELETE FROM academic_results WHERE id=?", (result_id,))
    conn.commit(); conn.close()

def save_test_responses(user_id, test_type, answers_dict, score):
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    c.execute("DELETE FROM test_responses WHERE user_id=? AND test_type=?", (user_id, test_type))
    now = datetime.datetime.now().isoformat()
    for q_id, ans in answers_dict.items():
        c.execute("INSERT INTO test_responses (user_id,test_type,question_id,answer,score,submitted_at) VALUES (?,?,?,?,?,?)",
                  (user_id, test_type, q_id, str(ans), score, now))
    conn.commit(); conn.close()

def get_completed_tests(user_id):
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    c.execute("SELECT DISTINCT test_type FROM test_responses WHERE user_id=?", (user_id,))
    rows = c.fetchall(); conn.close()
    return {r[0] for r in rows}

def save_recommendation(user_id, career_path, confidence, universities, mentors, narrative, top3):
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    c.execute("DELETE FROM recommendations WHERE user_id=?", (user_id,))
    c.execute("""INSERT INTO recommendations
        (user_id,career_path,confidence,universities,linkedin_mentors,narrative,top3,generated_at)
        VALUES (?,?,?,?,?,?,?,?)""",
        (user_id, career_path, confidence,
         json.dumps(universities), json.dumps(mentors),
         narrative, json.dumps(top3),
         datetime.datetime.now().isoformat()))
    conn.commit(); conn.close()

def get_recommendation(user_id):
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    c.execute("""SELECT career_path,confidence,universities,linkedin_mentors,
                        narrative,top3,generated_at
                 FROM recommendations WHERE user_id=? ORDER BY generated_at DESC LIMIT 1""", (user_id,))
    row = c.fetchone(); conn.close()
    if not row: return None
    return {"career_path":row[0],"confidence":row[1],
            "universities":json.loads(row[2] or "[]"),
            "mentors":     json.loads(row[3] or "[]"),
            "narrative":   row[4] or "",
            "top3":        json.loads(row[5] or "[]"),
            "generated_at":row[6]}

def save_chat_message(user_id, role, message):
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (user_id,role,message,created_at) VALUES (?,?,?,?)",
              (user_id, role, message, datetime.datetime.now().isoformat()))
    conn.commit(); conn.close()

def get_chat_history(user_id, limit=40):
    conn = sqlite3.connect("career_portal.db")
    c = conn.cursor()
    c.execute("SELECT role,message FROM chat_history WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
              (user_id, limit))
    rows = c.fetchall(); conn.close()
    return list(reversed(rows))

# ====================== HELPERS ======================
def get_result_types(class_level):  return ["First Term","Second Term","Third Term"]

def get_subjects(class_level, department=None):
    if class_level in ["JSS 2","JSS 3"]:
        return ["Mathematics","English Language","Basic Science",
                "Social Studies","Civic Education","Physical Education"]
    return DEPARTMENT_SUBJECTS.get(department, ["English","Mathematics"])

def score_cognitive(answers, questions):
    return round(sum(1 for q in questions
                     if answers.get(q["id"])==q["correct"] and q["correct"] is not None
                     ) / len(questions) * 100, 1)

def score_aptitude(answers, questions, department):
    dept = department if department in ("Science","Arts","Commercial") else "Science"
    total, maxp = 0, 0
    for q in questions:
        w = q.get("weights",{}).get(dept,[0,0,0,0])
        maxp += max(w) if w else 3
        ans = answers.get(q["id"])
        if ans is not None: total += w[ans] if ans < len(w) else 0
    return round(total / max(maxp,1) * 100, 1)

def score_likert(answers, questions):
    scores = [answers[q["id"]]+1 for q in questions if answers.get(q["id"]) is not None]
    return round(sum(scores)/(len(scores)*5)*100, 1) if scores else 0.0


# ====================== ML INFERENCE ======================
def build_feature_vector(results, profile):
    TERM_MAP = {"First Term":"t1","Second Term":"t2","Third Term":"t3"}
    bucket = defaultdict(list)
    for r in results:
        pfx = TERM_MAP.get(r[1],"t3")
        key = r[2].replace(" ","_").replace("/","_").replace("&","and")
        bucket[(pfx,key)].append(float(r[3]))

    subj_avg = {f"{p}_{k}": round(sum(v)/len(v),2) for (p,k),v in bucket.items()}

    def tavg(pfx):
        vs=[v for k,v in subj_avg.items() if k.startswith(pfx+"_")]
        return round(sum(vs)/len(vs),2) if vs else 0.0

    t3a=tavg("t3"); t2a=tavg("t2") or t3a; t1a=tavg("t1") or t2a
    sess=round((t1a+t2a+t3a)/3,2)
    cons=round(max(0,100-np.std([t1a,t2a,t3a])*2),2)
    trend="Improving" if t3a>t1a+4 else ("Declining" if t3a<t1a-4 else "Stable")
    strength="high" if sess>=65 else ("average" if sess>=50 else "low")

    SCI  = ["t3_Mathematics","t3_Basic_Science","t3_Physics","t3_Chemistry",
            "t3_Biology","t3_Computer_Studies","t3_Computer_Science","t3_Further_Mathematics"]
    ARTS = ["t3_English_Language","t3_Literature_in_English","t3_Government",
            "t3_History","t3_Social_Studies","t3_Cultural_and_Creative_Arts","t3_CRS_IRK"]
    COM  = ["t3_Economics","t3_Accounting","t3_Commerce","t3_Business_Studies","t3_Office_Practice"]

    def davg(ks):
        vs=[subj_avg[k] for k in ks if k in subj_avg]; return round(sum(vs)/len(vs),2) if vs else 0.0

    t3s={k:v for k,v in subj_avg.items() if k.startswith("t3_")}
    best_k=max(t3s,key=t3s.get) if t3s else "t3_Mathematics"
    weak_k=min(t3s,key=t3s.get) if t3s else "t3_Mathematics"

    def grade(s):
        return "A" if s>=75 else "B" if s>=65 else "C" if s>=50 else "D" if s>=45 else "E" if s>=40 else "F"
    gc={"A":0,"B":0,"C":0,"D":0,"E":0,"F":0}
    for v in t3s.values(): gc[grade(v)]+=1

    CLASS_MAP={"JSS 2":"JSS2","JSS 3":"JSS3","SSS 1":"SSS1","SSS 2":"SSS2","Admin":"JSS2"}
    DEPT_MAP ={"Science":"Science","Arts":"Arts","Commercial":"Commercial",
               None:"N_A","":"N_A","Select Department":"N_A"}

    def se(le_key, val):
        le=ml_models[le_key]; cls=list(le.classes_)
        return int(le.transform([val])[0]) if val in cls else 0

    row = {
        "term1_avg":t1a,"term2_avg":t2a,"term3_avg":t3a,
        "session_avg":sess,"consistency_score":cons,
        "science_aptitude_score":davg(SCI),"arts_aptitude_score":davg(ARTS),
        "commercial_aptitude_score":davg(COM),
        "best_subject_score":t3s.get(best_k,0),"weak_subject_score":t3s.get(weak_k,0),
        "grade_A_count":gc["A"],"grade_B_count":gc["B"],"grade_C_count":gc["C"],
        "grade_D_count":gc["D"],"grade_E_count":gc["E"],"grade_F_count":gc["F"],
        "class_level_enc":    se("le_class",   CLASS_MAP.get(profile.get("class_level","JSS2"),"JSS2")),
        "department_enc":     se("le_dept",    DEPT_MAP.get(profile.get("department",""),"N_A")),
        "strength_level_enc": se("le_strength",strength),
        "performance_trend_enc":se("le_trend", trend),
        "best_subject_enc":   se("le_best", best_k.replace("t3_","").replace("_"," ")),
        "weak_subject_enc":   se("le_weak", weak_k.replace("t3_","").replace("_"," ")),
    }
    row.update(subj_avg)
    vec = np.array([row.get(f,0.0) for f in ml_models["feature_names"]], dtype=np.float32)
    meta = {"session_avg":sess,"trend":trend,"strength":strength,
            "best_subject":best_k.replace("t3_","").replace("_"," "),"best_score":t3s.get(best_k,0),
            "weak_subject":weak_k.replace("t3_","").replace("_"," "),"weak_score":t3s.get(weak_k,0),
            "t1_avg":t1a,"t2_avg":t2a,"t3_avg":t3a,"consistency":cons,
            "sci_apt":davg(SCI),"arts_apt":davg(ARTS),"com_apt":davg(COM),"grade_counts":gc}
    return vec, meta

def ml_predict(results, profile):
    vec, meta = build_feature_vector(results, profile)
    vs  = ml_models["scaler"].transform(vec.reshape(1,-1))
    idx = ml_models["xgb"].predict(vs)[0]
    proba = ml_models["xgb"].predict_proba(vs)[0]
    career = ml_models["le_career"].inverse_transform([idx])[0]
    conf   = round(float(proba[idx])*100,1)
    t3i    = np.argsort(proba)[::-1][:3]
    top3   = [(ml_models["le_career"].inverse_transform([i])[0], round(float(proba[i])*100,1))
              for i in t3i]
    return {"career_path":career,"confidence":conf,"top3":top3,"meta":meta}

# ====================== GEMINI: RECOMMENDATION NARRATIVE ======================
def _fallback_narrative(name, career, confidence, top3, meta):
    t2name = top3[1][0] if len(top3)>1 else "an alternative"
    t3name = top3[2][0] if len(top3)>2 else "another option"
    return f"""#### 🌟 Your Career Recommendation Summary
Hi {name}! Based on a comprehensive analysis of your academic performance and all four assessments, your strongest career match is {career}, with a confidence level of {confidence}%. 
Your overall session average of {meta['session_avg']}%, combined with your outstanding performance in {meta['best_subject']} ({meta['best_score']}%), highlights both your capability and natural alignment with this path. 
These results don’t just reflect what you’ve achieved—they reveal where your strengths are most likely to thrive and succeed in the real world.

#### 🎯 Recommended Career Path: {career}
This is one of the most in-demand and impactful career fields in Nigeria today. Professionals here work across the private sector, federal agencies, and international organisations. Your assessment results — particularly your aptitude score and academic performance — show exactly the potential this field requires.

#### 💼 Five Nigerian Career Roles to Explore
- **Core Specialist** — practise your craft at a federal agency or major private company
- **Consultant / Adviser** — work across multiple organisations solving problems
- **Research & Analysis** — contribute to academia, think-tanks, or government policy
- **Entrepreneurship** — start your own practice, firm, or business
- **NGO / Development Sector** — tackle national challenges with international organisations

#### 📈 Your Competitive Strengths
- Strong academic performance — **{meta['best_subject']}** is your top subject at **{meta['best_score']}%**
- {'Improving trend across terms — shows commitment and growth capacity.' if meta['trend']=='Improving' else 'Consistent performance across all three terms — shows reliability and focus.'}
- Assessment scores confirm real aptitude for this career path

#### ⚠️ Areas to Strengthen
Focus extra effort on **{meta['weak_subject']}** ({meta['weak_score']}%) — it appears in WAEC and will matter for admission. Also build reading and comprehension skills through daily practice.

#### 🔄 Your Two Backup Career Options
**{t2name}** is your second-best match — great if your interests evolve. **{t3name}** is also a strong fit and may suit a slightly different academic path.

#### 🚀 Action Steps for Right Now
1. Research what professionals in **{career}** actually do in Nigeria — YouTube and LinkedIn are great starting points
2. Talk to your school counsellor about the right subjects for your SSS class combination
3. Start practising JAMB past questions in your core subjects — aim for consistency
4. Join a relevant school club, science fair, or business competition to start building real experience"""

def generate_recommendation_gemini(name, class_level, department, ml_result, test_scores, results):
    meta  = ml_result["meta"]
    top3  = ml_result["top3"]
    dept_txt = f" ({department} dept.)" if department else ""

    subj_avgs = defaultdict(list)
    for r in results:
        subj_avgs[r[2]].append(float(r[3]))
    summary = " | ".join(f"{s}: {round(sum(v)/len(v),1)}%" for s,v in list(subj_avgs.items())[:6])

    prompt = f"""You are a warm, expert career guidance counsellor at a top Nigerian secondary school.
Write a personalised career recommendation report for a student.

STUDENT PROFILE:
- Name: {name}
- Class: {class_level}{dept_txt}
- Session Average: {meta['session_avg']}% (T1={meta['t1_avg']}%, T2={meta['t2_avg']}%, T3={meta['t3_avg']}%)
- Trend: {meta['trend']} | Academic standing: {meta['strength'].upper()}
- Best Subject: {meta['best_subject']} ({meta['best_score']}%)
- Weakest Subject: {meta['weak_subject']} ({meta['weak_score']}%)
- Grade-A count: {meta['grade_counts']['A']} | Grade-F count: {meta['grade_counts']['F']}
- Science aptitude: {meta['sci_apt']}% | Arts: {meta['arts_apt']}% | Commercial: {meta['com_apt']}%
- Subject summary: {summary}

4-TEST SCORES:
- Cognitive (logic & reasoning): {test_scores.get('cognitive', 50)}%
- Aptitude (natural talents): {test_scores.get('aptitude', 50)}%
- Psychometric (personality): {test_scores.get('psychometric', 50)}%
- Sentiment (motivation & mindset): {test_scores.get('sentiment', 50)}%

ML MODEL OUTPUT:
- Primary career: {top3[0][0]} (confidence {top3[0][1]}%)
- 2nd option: {top3[1][0] if len(top3)>1 else 'N/A'} ({top3[1][1] if len(top3)>1 else 0}%)
- 3rd option: {top3[2][0] if len(top3)>2 else 'N/A'} ({top3[2][1] if len(top3)>2 else 0}%)

Write the report using EXACTLY these section headers:

## 🌟 Your Career Recommendation Summary
2–3 sentences directly addressing {name}, referencing their strongest results.

## 🎯 Recommended Career Path: {top3[0][0]}
Two paragraphs: (1) What this career involves in Nigeria — real sectors, agencies (NNPC, CBN, NAFDAC, NTA, MTN, etc.) (2) Exactly why this matches {name}'s data — mention actual scores.

## 💼 Five Nigerian Career Roles to Explore
5 specific job roles in demand in Nigeria, one line each with a Nigerian employer or context.

## 📈 Your Competitive Strengths
Three bullet points rooted in the actual data. Reference real scores and subjects.

## ⚠️ Areas to Strengthen
Two specific, encouraging, actionable suggestions. Reference {meta['weak_subject']} directly.

## 🔄 Your Two Backup Career Options
Short paragraph each on {top3[1][0] if len(top3)>1 else 'Alternative A'} and {top3[2][0] if len(top3)>2 else 'Alternative B'}.

## 🚀 Action Steps for Right Now
Four numbered, concrete steps {name} can take TODAY as a {class_level} student in Nigeria. Include JAMB subject choices, WAEC prep, and free resources.

Tone: warm, direct, encouraging — like a trusted school counsellor talking to a Nigerian teenager.
Do NOT include university suggestions (handled separately). Total: ~650–800 words."""

    try:
        resp = gemini_client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=1200, temperature=0.7))
        return resp.text
    except Exception:
        return _fallback_narrative(name, ml_result["career_path"],
                                   ml_result["confidence"], ml_result["top3"], meta)

# ====================== GEMINI: CHATBOT ======================
def get_chatbot_response(user_message, student_context, history_list):
    msg = user_message.lower()

    if "jamb" in msg:
        extra_instruction = "Focus on JAMB subjects, cutoff marks, and preparation strategy."
    elif "university" in msg:
        extra_instruction = "Recommend Nigerian universities and admission strategy."
    elif "career" in msg:
        extra_instruction = "Explain career paths and real-world roles in Nigeria."
    else:
        extra_instruction = "Give practical career guidance."

    # ------------------ Strong System Prompt ------------------
    system_ctx = f"""
You are a highly experienced Nigerian career counsellor.

Your job is to give SPECIFIC, PRACTICAL, PERSONALISED advice.

You MUST:
- Use the student's data (scores, strengths, career recommendation)
- Mention Nigerian context (JAMB, WAEC, universities, companies)
- Give actionable steps (bullet points)

Response Structure:
1. Direct answer (1–2 sentences)
2. Personalised explanation using their data
3. Practical steps (bullet points)
4. End with one short encouraging sentence

Avoid:
- Generic advice
- Long storytelling

FOCUS:
{extra_instruction}

STUDENT DATA:
{student_context}
"""

    # ------------------ History (last 6 messages only) ------------------
    history = []
    for role, msg in history_list[-6:]:
        history.append(types.Content(
            role="user" if role == "user" else "model",
            parts=[types.Part(text=msg)]
        ))

    # ------------------ Gemini Call ------------------
    try:
        chat = gemini_client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_ctx,
                max_output_tokens=350,
                temperature=0.7
            ),
            history=history[:-1] if len(history) > 1 else []
        )

        resp = chat.send_message(user_message)
        response = resp.text.strip()

        # ------------------ Post-processing ------------------
        if len(response.split()) > 220:
            response = " ".join(response.split()[:220]) + "..."

        response = response.replace("•", "\n•")

        return response

    except Exception:
        return "Sorry, something went wrong. Please try again."
    
# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role


init_db()
create_admin_user()

# ====================== RENDER SINGLE TEST ======================
def render_test(meta, completed_tests):
    test_key  = meta["key"]
    questions = meta["questions"]

    st.markdown(f"""
    <div class="test-header">
        <h2>{meta['icon']} {meta['label']}</h2>
        <p>{meta['desc']}</p>
    </div>""", unsafe_allow_html=True)

    if test_key in completed_tests:
        score = st.session_state.test_scores.get(test_key, "N/A")
        st.success(f"✅ Already completed — Score: **{score}%**")
        if st.button("🔄 Retake This Test", key=f"retake_{test_key}"):
            save_test_responses(st.session_state.user_id, test_key, {}, 0)
            st.session_state.test_answers.pop(test_key, None)
            st.session_state.test_scores.pop(test_key, None)
            st.session_state.all_tests_done = False
            st.session_state.rec_cache = None
            st.rerun()
        return

    if test_key not in st.session_state.test_answers:
        st.session_state.test_answers[test_key] = {}

    for idx, q in enumerate(questions):
        st.markdown(f"""
        <div class="q-card">
            <div class="q-num">Question {idx+1} of {len(questions)}</div>
            <div class="q-text">{q['text']}</div>
        </div>""", unsafe_allow_html=True)
        cur = st.session_state.test_answers[test_key].get(q["id"], 0)
        chosen = st.radio(f"q_{q['id']}", q["options"], index=cur,
                          key=f"radio_{test_key}_{q['id']}", label_visibility="collapsed")
        st.session_state.test_answers[test_key][q["id"]] = q["options"].index(chosen)

    st.divider()
    _, col2 = st.columns([3,1])
    with col2:
        if st.button(f"✅ Submit {meta['label']}", type="primary", key=f"submit_{test_key}"):
            answers = st.session_state.test_answers[test_key]
            if len(answers) < len(questions):
                st.error("Please answer all questions before submitting.")
                return
            dept = st.session_state.department or "Science"
            if test_key == "cognitive":
                score = score_cognitive(answers, questions)
            elif test_key == "aptitude":
                score = score_aptitude(answers, questions, dept)
            else:
                score = score_likert(answers, questions)
            save_test_responses(st.session_state.user_id, test_key, answers, score)
            st.session_state.test_scores[test_key] = score
            st.session_state.active_test = None
            st.session_state.rec_cache   = None
            st.success(f"🎉 {meta['label']} submitted! Score: **{score}%**")
            st.rerun()

# ====================== CLEAR USER DATA ON LOGOUT ======================
def clear_user_data():
    """Safely clear all user-specific data on logout"""
    keys_to_clear = [
        "active_test", "test_answers", "test_scores", "all_tests_done",
        "rec_cache", "chat_cache", "upload_dir"
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    
    # Reset login state
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.full_name = None
    st.session_state.class_level = None
    st.session_state.department = None

# ====================== SESSION STATE ======================
for k, v in [("logged_in",False),("user_id",None),("full_name",None),
              ("class_level",None),("department",None),("active_test",None),
              ("test_answers",{}),("test_scores",{}),("all_tests_done",False),
              ("rec_cache",None),("chat_cache",None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ====================== MAIN APP ======================
def app():
    if st.session_state.get("logged_in", False):
        keep_alive()
    if st.session_state.get("logged_in", False):
        st.markdown("""
        <div class="container">

        <div class="title">🎓 Student Career Portal</div>

        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="subtitle">Smart Career Path Recommendation for Nigerian Secondary Students</div>', unsafe_allow_html=True)

        # ── LOGGED IN ────────────────────────────────────────────────────────────
        # Sidebar
        dept_txt = f" — {st.session_state.department}" if st.session_state.department else ""
        st.sidebar.success(f"👋 {st.session_state.full_name}\n{st.session_state.class_level}{dept_txt}")
        # Logout Button
        if st.sidebar.button("🚪 Logout", type="secondary"):
                clear_user_data()
                st.success("👋 You have been logged out successfully.")
                st.rerun()
        if not ML_READY:
            st.sidebar.warning("⚠️ ML models not found. Run `train_model.ipynb` first.")
            
        tab_dashboard, tab_upload, tab_test, tab_rec = st.tabs([
            "🏠 Dashboard","📤 Upload Results","🧠 Take 4 Tests","📊 My Recommendations"
        ])

        # ----------------------- DASHBOARD ----------------------------

        with tab_dashboard:
            st.header('🎓 Welcome to Your Smart Career Journey', text_alignment='center')
            #st.markdown('<div class="title">🎓 Welcome to Your Smart Career Journey</div>', unsafe_allow_html=True)
            st.subheader(f"Hello, {st.session_state.full_name}! 👋")
            st.caption(f"Class: **{st.session_state.class_level}**" +
                    (f" | Department: **{st.session_state.department}**" if st.session_state.department else ""))

            n_tests   = len(get_completed_tests(st.session_state.user_id))
            n_results = len(get_user_results(st.session_state.user_id))
            rec_ready = get_recommendation(st.session_state.user_id) is not None

            c1,c2,c3 = st.columns(3)
            c1.metric("📋 Results Uploaded", n_results,       help="Upload First, Second & Third Term scores")
            c2.metric("🧠 Tests Completed",  f"{n_tests}/4",  help="Complete all 4 assessments")
            c3.metric("📊 Recommendation",   "✅ Ready" if rec_ready else ("Generate ↗" if n_tests==4 else "Pending"))

            st.info("Complete all steps to unlock your personalised AI-powered career path!")

            st.divider()
            st.markdown("### 📋 How It Works")
            for icon, title, desc in [
                ("1️⃣","Upload Academic Results","Add your First, Second & Third Term scores for all subjects."),
                ("2️⃣","Take 4 Tests","Complete Cognitive, Aptitude, Psychometric & Sentiment assessments."),
                ("3️⃣","Get AI Recommendation","Our XGBoost ML model + Gemini AI analyses your full profile."),
                ("4️⃣","Explore & Chat","View Nigerian university options and chat with your AI career counsellor."),
            ]:
                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;gap:14px;padding:10px 0;border-bottom:1px solid #f0f0f0;">
                    <div style="font-size:22px;line-height:1.2;">{icon}</div>
                    <div><strong style="font-size:14px;">{title}</strong>
                        <div style="font-size:13px;color:gray;margin-top:2px;">{desc}</div></div>
                </div>""", unsafe_allow_html=True)
            pass

        # -------------------- UPLOAD RESULTS --------------------

        with tab_upload:
            st.markdown('<div class="subtitle">Step 1 of 3</div>', unsafe_allow_html=True)
            st.markdown('<div class="title">📤 Academic Results Upload</div>', unsafe_allow_html=True)
            st.caption(f"**{st.session_state.class_level}** preference page" +
                    (f" | Dept: **{st.session_state.department}**" if st.session_state.department else ""))

            results = get_user_results(st.session_state.user_id)
            if results:
                st.subheader("📋 Your Uploaded Results")
                for row in results:
                    rid, rtype, subj, score, edate, uploaded = row
                    c1,c2,c3 = st.columns([6,2,1])
                    with c1: st.write(f"**{rtype}** — {subj} | Score: **{score}** | {edate}")
                    with c2: st.caption(uploaded[:10] if uploaded else "")
                    with c3:
                        if st.button("🗑️", key=f"del_{rid}", help="Delete result"):
                            delete_academic_result(rid)
                            st.session_state.rec_cache = None
                            st.success("Deleted!")
                            st.rerun()
            else:
                st.info("No results uploaded yet. Use the form below to add your term scores.")

            st.subheader("➕ Add New Result")

            st.subheader("📄 Upload Result PDF")
            uploaded_file = st.file_uploader(
                "Upload your result sheet (PDF only)",
                type=["pdf"]
            )
            if uploaded_file:
                text = extract_pdf_text(uploaded_file)
                parsed_results = parse_results(text)

                if parsed_results:
                    st.success("✅ Results detected from PDF")

                    for subj, score in parsed_results:
                        st.write(f"📘 {subj} — {score}")

                    if st.button("💾 Save All Results from PDF"):
                        for subj, score in parsed_results:
                            save_academic_result(
                                st.session_state.user_id,
                                "Third Term",  # you can later make this selectable
                                subj,
                                score,
                                datetime.date.today()
                            )

                        st.success("🎉 Results saved successfully!")
                        st.rerun()

                else:
                    st.error("❌ Could not detect results. Try manual input.")

            st.divider()
            st.subheader("✍️ Or Enter Results Manually")

            with st.form("add_result_form", clear_on_submit=True):
                c1,c2 = st.columns([3,3])
                with c1:
                    rtype = st.segmented_control("Select Result Type", options=get_result_types(st.session_state.class_level), default=None)
                    edate = st.date_input("Exam Date", value=datetime.date.today())
                with c2:
                    slist = get_subjects(st.session_state.class_level, st.session_state.department)
                    subj  = st.segmented_control("Select Subject", options=slist, default=None)
                    score_str = st.text_input("Score (0–100)", value="", placeholder="Enter score, e.g. 85.5")
                

                if st.form_submit_button("Add This Result", type="primary"):
                    if rtype == "Select Result Type" or subj == "Select Subject":
                        st.error("Please select both Result Type and Subject.")
                    elif not score_str.strip():
                        st.error("Please enter a score.")
                    else:
                        try:
                            score = float(score_str.strip())
                            if not (0 <= score <= 100):
                                st.error("Score must be between 0 and 100.")
                            else:
                                save_academic_result(st.session_state.user_id, rtype, subj, score, edate)
                                st.session_state.rec_cache = None
                                st.success(f"✅ {rtype} — {subj} ({score}) added successfully!")
                                st.rerun()
                        except ValueError:
                            st.error("Please enter a valid number for the score.")

            st.info("✅ Tip: Add all three terms across your main subjects for the most accurate recommendation.")
            pass  

        # -------------------- TEST TAB --------------------

        with tab_test:
            st.markdown('<div class="title">🧠 Take 3 Tests</div>', unsafe_allow_html=True)
            st.markdown('<div class="subtitle">Step 2 of 3 — Complete all three tests to unlock your personalised career recommendation.</div>', unsafe_allow_html=True)

            completed_tests = get_completed_tests(st.session_state.user_id)
            n_done = len(completed_tests)
            pct    = int(n_done/3*100)

            st.markdown(f"""
            <div style="margin-bottom:6px;"><b>Overall Progress: {n_done} / 3 tests completed</b></div>
            <div class="test-progress-bar">
                <div class="test-progress-fill" style="width:{pct}%;"></div>
            </div>""", unsafe_allow_html=True)

            st.markdown("### 🗂️ Select a Test to Begin")
            cols = st.columns(4)
            for i, meta in enumerate(TEST_META):
                with cols[i]:
                    done   = meta["key"] in completed_tests
                    badge  = "completed-badge" if done else "active-badge"
                    status = "✅ Done" if done else "⬇ Click the button below to start"
                    stxt   = f"Score: {st.session_state.test_scores.get(meta['key'],'—')}%" if done else ""
                    bclr   = "#10b981" if done else "#2d6cdf"
                    st.markdown(f"""
                    <div style="background:white;border-radius:12px;padding:16px;text-align:center;
                        box-shadow:0 2px 10px rgba(0,0,0,0.07);border-top:4px solid {bclr};min-height:165px;">
                        <div style="font-size:28px;">{meta['icon']}</div>
                        <div style="font-weight:700;font-size:14px;margin:6px 0;">{meta['label']}</div>
                        <div style="font-size:12px;color:gray;margin-bottom:8px;">{meta['desc']}</div>
                        <span class="{badge}">{status}</span>
                        <div style="font-size:12px;color:#065f46;margin-top:4px;">{stxt}</div>
                    </div>""", unsafe_allow_html=True)
                    if st.button("🔄 Retake" if done else "▶ Open",
                                key=f"open_{meta['key']}", use_container_width=True):
                        st.session_state.active_test = meta["key"]
                        st.rerun()

            st.divider()

            if st.session_state.active_test:
                active = next((m for m in TEST_META if m["key"]==st.session_state.active_test), None)
                if active:
                    if st.button("← Back to Test List"):
                        st.session_state.active_test = None; st.rerun()
                    render_test(active, completed_tests)
            else:
                if n_done == 4:
                    st.success("🎉 **All 4 tests completed!** Go to **📊 My Recommendations** to see your career path.")
                    st.session_state.all_tests_done = True
                elif n_done == 0:
                    st.info("👆 Click **▶ Open** on any test above to begin. Start with the **Cognitive Test**.")
                else:
                    remaining = [m["label"] for m in TEST_META if m["key"] not in completed_tests]
                    st.info(f"👍 Good progress! Still needed: **{', '.join(remaining)}**")
            pass
        
        # -------------------- RECOMMENDATIONN TAB --------------------

        with tab_rec:
            st.markdown('<div class="title">📊 My Personalised Career Recommendations</div>', unsafe_allow_html=True)

            completed_tests = get_completed_tests(st.session_state.user_id)
            results         = get_user_results(st.session_state.user_id)

            # Guards
            if len(completed_tests) < 4:
                missing = [m["label"] for m in TEST_META if m["key"] not in completed_tests]
                st.warning(f"⚠️ Complete all 4 tests first. Pending: **{', '.join(missing)}**")
                st.info("👉 Go to the **🧠 Take 4 Tests** tab.")
                return

            if not results:
                st.warning("⚠️ Please upload at least one academic result first.")
                st.info("👉 Go to the **📤 Upload Results** tab.")
                return

            if not ML_READY:
                st.error("⚠️ ML models not found. Place `ml/models/` folder next to `app.py` and restart.")
                st.info("Run `train_model.ipynb` to generate the model files.")
                return

            existing = st.session_state.rec_cache or get_recommendation(st.session_state.user_id)

            # Generate / Regenerate
            btn_label = "🔄 Regenerate My Recommendations" if existing else "🚀 Generate My Career Recommendations"
            if st.button(btn_label, type="primary"):
                with st.spinner("🤖 Analysing your full profile with XGBoost ML + Gemini AI..."):
                    profile = {"class_level":st.session_state.class_level,
                                "department": st.session_state.department}
                    test_scores = {k: st.session_state.test_scores.get(k, 50.0)
                                    for k in ["cognitive","aptitude","psychometric","sentiment"]}

                    ml_result = ml_predict(results, profile)

                    narrative = generate_recommendation_gemini(
                        st.session_state.full_name, st.session_state.class_level,
                        st.session_state.department, ml_result, test_scores, results)

                    unis = UNIVERSITY_MAP.get(ml_result["career_path"], [])

                    mentor_prompt = f"""List 4 realistic Nigerian professionals in {ml_result['career_path']}.
        Return a JSON array only. Each item is a string:
        "[Full Name] — [Job Title] at [Nigerian Organisation] — [One sentence: why they are a good mentor]"
        Return ONLY the JSON array. No markdown, no extra text."""
                    try:
                        mr  = gemini_client.models.generate_content(
                            model=GEMINI_MODEL, contents=mentor_prompt,
                            config=types.GenerateContentConfig(max_output_tokens=300, temperature=0.6))
                        raw = mr.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                        mentors = json.loads(raw)
                    except Exception:
                        mentors = [
                            f"Dr. Chukwuemeka Eze — Senior Professional at NNPC — Dedicated mentor with 15+ years experience",
                            f"Mrs. Ngozi Adeyemi — Director, Federal Ministry of Nigeria — Passionate about youth career development",
                            f"Mr. Oluwaseun Bello — Lead Consultant, Lagos — Known for mentoring secondary school students",
                            f"Prof. Amina Suleiman — University of Abuja — Active researcher and student career advocate",
                        ]

                    save_recommendation(st.session_state.user_id, ml_result["career_path"],
                                        ml_result["confidence"], unis, mentors,
                                        narrative, ml_result["top3"])
                    conn2 = sqlite3.connect("career_portal.db")
                    conn2.execute("DELETE FROM chat_history WHERE user_id=?", (st.session_state.user_id,))
                    conn2.commit(); conn2.close()

                    st.session_state.rec_cache  = get_recommendation(st.session_state.user_id)
                    st.session_state.chat_cache = None
                    st.success("✅ Recommendation generated!")
                    st.rerun()

            if not existing:
                st.info("👆 Click **🚀 Generate My Career Recommendations** above to see your personalised career path.")
                return

            rec    = existing
            career = rec["career_path"]
            conf   = rec["confidence"]
            top3   = rec["top3"]
            unis   = rec["universities"]
            mentors= rec["mentors"]
            narr   = rec["narrative"]
            gen_at = rec["generated_at"][:16].replace("T"," ")

            # Hero card
            st.markdown(f"""
            <div class="rec-hero">
                <h1>🎯 {career}</h1>
                <p>ML Confidence: <strong>{conf}%</strong> &nbsp;|&nbsp; Generated: {gen_at}
                &nbsp;|&nbsp; ✨ Powered by Gemini AI</p>
            </div>""", unsafe_allow_html=True)

            # Scores + Test results
            col_l, col_r = st.columns([3,2])
            with col_l:
                st.markdown("#### 📊 Career Match Confidence")
                medals = ["🥇","🥈","🥉"]
                for idx2, (cname, cprob) in enumerate(top3[:3]):
                    st.markdown(f"""
                    <div style="margin-bottom:14px;">
                        <div style="font-size:14px;font-weight:600;">{medals[idx2]} {cname}</div>
                        <div style="font-size:12px;color:gray;margin-bottom:3px;">{cprob}% match</div>
                        <div class="score-bar-bg">
                            <div class="score-bar-fill" style="width:{int(cprob)}%;"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            with col_r:
                st.markdown("#### 📝 Your Test Scores")
                for tlabel, tkey in [("🧩 Cognitive","cognitive"),("🎯 Aptitude","aptitude"),
                                        ("🧠 Psychometric","psychometric"),("💬 Sentiment","sentiment")]:
                    sc = st.session_state.test_scores.get(tkey, "—")
                    sv = f"{sc}%" if isinstance(sc,(int,float)) else sc
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;padding:6px 0;
                            border-bottom:1px solid #f0f0f0;font-size:14px;">
                        <span>{tlabel}</span><strong>{sv}</strong>
                    </div>""", unsafe_allow_html=True)

            st.divider()

            # Narrative
            st.markdown("## 📋 **Your Personalised Career Report**")
            st.markdown(narr)
            st.divider()

            # Universities
            st.markdown("#### 🏛️ Recommended Nigerian Universities")
            st.caption(f"Top institutions offering programmes in **{career}**")
            if unis:
                uc1, uc2 = st.columns(2)
                for idx2, u in enumerate(unis[:4]):
                    with (uc1 if idx2%2==0 else uc2):
                        st.markdown(f"""
                        <div class="uni-card">
                            <div style="font-weight:700;font-size:15px;">🏛️ {u['name']}</div>
                            <div style="font-size:13px;color:#1a3c8f;margin:3px 0;">📚 {u['course']}</div>
                            <div style="font-size:12px;color:#444;">🎯 JAMB Cutoff: <strong>{u.get('cutoff','200+')}</strong></div>
                            <div style="font-size:12px;color:gray;margin-top:2px;">📍 {u.get('location','Nigeria')}</div>
                            <a href="{u.get('url','#')}" target="_blank"
                                style="font-size:12px;color:#2d6cdf;text-decoration:none;">
                                🌐 Visit Website ↗</a>
                        </div>""", unsafe_allow_html=True)
            else:
                st.info("University data not available for this career path.")

            st.divider()

            # LinkedIn Mentors
            st.markdown("#### 👥 Suggested LinkedIn Mentors")
            st.caption("Nigerian professionals in your recommended career field — search them on LinkedIn")
            if mentors:
                for m in mentors[:4]:
                    st.markdown(f"""
                    <div style="background:white;border-radius:10px;padding:12px 16px;
                            margin-bottom:10px;box-shadow:0 2px 8px rgba(0,0,0,0.06);
                            border-left:4px solid #2d6cdf;font-size:13px;">
                        👤 {m}
                    </div>""", unsafe_allow_html=True)

            st.divider()

            # Chatbot
            st.markdown("#### 💬 Ask Your AI Career Counsellor")
            st.caption("Questions about your recommendation, JAMB scores, university options, or career paths? Ask below!")

            rec = st.session_state.rec_cache or get_recommendation(st.session_state.user_id)
            
            student_ctx = f"""
            Name: {st.session_state.full_name}
            Class: {st.session_state.class_level}
            Department: {st.session_state.department}

            Career Recommendation: {rec['career_path']}
            Confidence: {rec['confidence']}%

            Top 3 Careers: {rec['top3']}

            """

            if st.session_state.chat_cache is None:
                st.session_state.chat_cache = get_chat_history(st.session_state.user_id)
            chat_history = st.session_state.chat_cache

            if st.session_state.chat_cache is None:
                st.session_state.chat_cache = get_chat_history(st.session_state.user_id)
            chat_history = st.session_state.chat_cache

            if not chat_history:
                st.markdown(f"""
                <div class="chat-ai">
                    Hi {st.session_state.full_name}! 👋 I'm your AI career counsellor, powered by Gemini.
                    I've reviewed your full profile and recommended <strong>{career}</strong> for you.
                    Do you have questions about this career, JAMB subject choices, university cut-offs,
                    or how to prepare? I'm here to help! 😊
                </div>""", unsafe_allow_html=True)
            else:
                for role, msg in chat_history:
                    css = "chat-user" if role=="user" else "chat-ai"
                    st.markdown(f'<div class="{css}">{msg}</div>', unsafe_allow_html=True)

            with st.form("chat_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Your message",
                    placeholder="e.g. What JAMB score do I need for UNILAG Medicine?",
                    label_visibility="collapsed")
                if st.form_submit_button("Send 💬", type="primary") and user_input.strip():
                    with st.spinner("Thinking..."):
                        reply = get_chatbot_response(user_input.strip(), student_ctx, chat_history)
                    save_chat_message(st.session_state.user_id, "user",      user_input.strip())
                    save_chat_message(st.session_state.user_id, "assistant", reply)
                    st.session_state.chat_cache = get_chat_history(st.session_state.user_id)
                    st.rerun()
            pass


        return

    # ====================== AUTHENTICATION PAGE (Not Logged In) ======================
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<div class="title">🎓 Smart Career Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Smart Career Path Recommendation for Nigerian Secondary Students</div>', unsafe_allow_html=True)


    auth_tab = st.tabs(["Login", "Sign Up"])

    # ====================== LOGIN TAB ======================
    with auth_tab[0]:
        st.subheader("Login to your account")
        with st.form("login_form", clear_on_submit=True):
            email = st.text_input("Email Address", placeholder="student@example.com")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login", type="primary"):
                if not email or not password:
                    st.error("Please enter both email and password.")
                else:
                    user = login_user(email, password)
                    if user:
                        st.session_state.update({
                            "logged_in": True,
                            "user_id": user[0],
                            "full_name": user[1],
                            "class_level": user[2],
                            "department": user[3],
                            "active_test": None,
                            "test_answers": {},
                            "test_scores": {},
                            "all_tests_done": False,
                            "rec_cache": None,
                            "chat_cache": None
                        })
                        st.success(f"Welcome back, {user[1]}! 🎓")
                        st.rerun()
                    else:
                        st.error("Invalid email or password. Please try again.")

    # ====================== SIGN UP TAB ======================
    with auth_tab[1]:
        st.subheader("Create New Account")
        with st.form("signup_form", clear_on_submit=True):
            full_name = st.text_input("Full Name *")
            dob = st.date_input("Date of Birth", 
                                max_value=datetime.date.today(), 
                                min_value=datetime.date(1990, 1, 1))
            
            class_level = st.segmented_control(
                "Class Level *", 
                options=["JSS 2", "JSS 3", "SSS 1", "SSS 2"],
                default=None
            )
            st.caption('Select Department if you are in SSS 1 or SSS 2')

            department = st.segmented_control(
                "Department *", 
                options=["Science", "Arts", "Commercial"],
                default=None
            )

            email = st.text_input("Email Address *")
            password = st.text_input("Password *", type="password")
            confirm = st.text_input("Confirm Password *", type="password")
            agree = st.checkbox("I agree to the Terms of Service and Privacy Policy")

            if st.form_submit_button("Create Account", type="primary"):
                if not agree:
                    st.warning("You must agree to the terms.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.warning("Password must be at least 6 characters long.")
                elif not full_name or not email:
                    st.warning("Please fill all required fields.")
                elif class_level in ["SSS 1", "SSS 2"] and not department:
                    st.warning("Please select your department.")
                else:
                    dept_to_save = department if class_level in ["SSS 1", "SSS 2"] else None
                    
                    if create_user(full_name, dob, class_level, dept_to_save, email, password):
                        st.success("🎉 Account created successfully!")
                        st.balloons()
                        st.info("Please go to the **Login** tab and sign in.")
                    else:
                        st.error("An account with this email already exists.")

    

    st.markdown('</div>', unsafe_allow_html=True)


# Call the app
if __name__ == "__main__":
    app()