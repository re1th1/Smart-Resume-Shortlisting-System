import streamlit as st
import pdfplumber
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Resume Matcher", page_icon="🎯", layout="wide")

# --- ULTRA-MODERN DARK MODE CSS ---
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background - Deep Midnight Gradient */
    .stApp {
        background: radial-gradient(circle at top left, #1a1a2e, #16213e, #0f3460);
        color: #ffffff;
    }

    /* Glassmorphism Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 25px;
    }

    /* Neon Titles */
    .main-title {
        font-size: 60px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-shadow: 0px 10px 20px rgba(0, 210, 255, 0.3);
    }
    
    .sub-title {
        font-size: 22px;
        text-align: center;
        color: #adb5bd;
        margin-bottom: 40px;
        font-weight: 400;
    }

    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 45px;
        color: #00d2ff !important;
        font-weight: 700;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-size: 18px;
    }

    /* Styling the File Uploader */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.07);
        border-radius: 20px;
        padding: 20px;
        border: 2px dashed #00d2ff;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 52, 96, 0.5) !important;
        backdrop-filter: blur(10px);
    }

    /* Table Styling */
    .stDataFrame {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- AI MODEL LOADING ---
@st.cache_resource
def load_ai_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_ai_model()

# --- HELPER FUNCTIONS ---
def load_jobs():
    try:
        with open('jobs.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ 'jobs.json' not found!")
        return []

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color: #00d2ff;'>⚙️ Control Panel</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <p><b>AI Model:</b> SBERT MiniLM</p>
        <p><b>Logic:</b> Cosine Similarity</p>
        <p><b>Status:</b> <span style='color: #00ff00;'>Active ✅</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📖 How to use")
    st.write("1. Upload Resume (PDF)")
    st.write("2. Let the AI compute vectors")
    st.write("3. View the top-ranked job match")
    st.divider()
    st.markdown("<p style='text-align: center;'>🚀 Hackathon Edition</p>", unsafe_allow_html=True)

# --- MAIN UI ---
st.markdown('<p class="main-title">🎯 AI Resume Matcher</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Next-Gen Talent Acquisition using Semantic AI</p>', unsafe_allow_html=True)

# File Uploader inside a custom container
col_1, col_2, col_3 = st.columns([1, 2, 1])
with col_2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Drop Candidate Resume Here", type="pdf")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    with st.spinner("🤖 AI is analyzing the resume..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        jobs_data = load_jobs()
        
        if jobs_data:
            job_roles = [j['role'] for j in jobs_data]
            job_descriptions = [j['description'] for j in jobs_data]

            # AI Calculation
            resume_vec = model.encode(resume_text, convert_to_tensor=True)
            jd_vecs = model.encode(job_descriptions, convert_to_tensor=True)
            cosine_scores = util.cos_sim(resume_vec, jd_vecs)[0]

            results = []
            for i in range(len(job_roles)):
                results.append({"Job Role": job_roles[i], "Score": round(float(cosine_scores[i]) * 100, 2)})

            df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

            # --- RESULTS SECTION ---
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 1. TOP MATCH SECTION (Glass Card)
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("<h2 style='text-align: center; color: #00d2ff;'>🏆 Best Fit Identified</h2>", unsafe_allow_html=True)
            
            top_role = df.iloc[0]['Job Role']
            top_score = df.iloc[0]['Score']
            
            m_col_left, m_col_right = st.columns([1, 2])
            with m_col_left:
                st.metric(label="Top Role", value=top_role, delta=f"{top_score}%")
            with m_col_right:
                st.markdown(f"<div style='padding-top: 20px;'>The AI has determined that the candidate is a strong match for the <b>{top_role}</b> role based on their professional experience and skill set.</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 2. DETAILED RANKING SECTION
            st.markdown("<h3 style='color: #adb5bd;'>📊 Comparative Analysis</h3>", unsafe_allow_html=True)
            res_col_1, res_col_2 = st.columns([1, 1])
            
            with res_col_1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with res_col_2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.bar_chart(df.set_index('Job Role')['Score'], color="#00d2ff")
                st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: #adb5bd; font-style: italic;'>
            Waiting for resume upload to begin AI analysis...
        </div>
    """, unsafe_allow_html=True)