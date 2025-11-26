# main_full.py
"""
Combined Streamlit app:
- Login system
- Dashboard (visualization)
- Prediction page
- Resume scoring
- Admin panel (retrain model)
- PDF report generation
- Hosting guide page
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import hashlib
from fpdf import FPDF
from io import BytesIO
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------
# Config / Globals
# ----------------------------
MODEL_PATH = "placement_model.pkl"
DATA_PATH = "Placement_Data_Full_Class.csv"  # default dataset filename, if present
DEFAULT_USERS = {
    # username: sha256(password)
    "admin": hashlib.sha256("adminpass".encode()).hexdigest(),
    "user": hashlib.sha256("userpass".encode()).hexdigest(),
}

st.set_page_config(page_title="Placement System (Full)", layout="wide")

# ----------------------------
# Authentication helpers
# ----------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def check_credentials(username: str, password: str) -> bool:
    # In production, replace with DB call. Here we use DEFAULT_USERS dict.
    hp = hash_password(password)
    return DEFAULT_USERS.get(username) == hp

def signup(username: str, password: str):
    DEFAULT_USERS[username] = hash_password(password)

# ----------------------------
# Model helpers
# ----------------------------
def train_model_from_df(df: pd.DataFrame, save_path=MODEL_PATH):
    # Basic preprocessing: expect numeric columns: ssc_p,hsc_p,degree_p,etest_p,mba_p; target 'status' (Placed/Not Placed)
    df = df.copy()
    # Try to map status if present
    if 'status' in df.columns:
        df['status_num'] = df['status'].map({'Placed':1,'Not Placed':0})
    else:
        # create synthetic target if missing
        df['status_num'] = (df.select_dtypes(include=[np.number]).mean(axis=1) > df.select_dtypes(include=[np.number]).mean().mean()).astype(int)

    features = []
    for f in ['ssc_p','hsc_p','degree_p','etest_p','mba_p']:
        if f in df.columns:
            features.append(f)
    if not features:
        # choose numeric columns if expected ones missing
        features = df.select_dtypes(include=[np.number]).columns.drop('status_num').tolist()[:5]
    X = df[features].fillna(df[features].median())
    y = df['status_num']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump({"model":clf, "features":features}, save_path)
    return clf, features

def load_model(save_path=MODEL_PATH):
    if os.path.exists(save_path):
        data = joblib.load(save_path)
        return data.get("model"), data.get("features")
    return None, None

# ----------------------------
# PDF generation helper
# ----------------------------
def generate_pdf_report(student_info: dict, prediction: int, prob: float, filename="report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(0, 10, "Student Placement Prediction Report", ln=True, align='C')
    pdf.ln(6)
    pdf.set_font("Arial", size=12)
    for k, v in student_info.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)
    pdf.ln(4)
    pdf.cell(0, 8, f"Prediction: {'Placed' if prediction==1 else 'Not Placed'}", ln=True)
    pdf.cell(0, 8, f"Confidence: {round(prob*100,2)}%", ln=True)
    # Save to bytes
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# ----------------------------
# Resume scoring helper (simple)
# ----------------------------
RESUME_KEYWORDS = {
    "python": 5, "machine learning": 8, "data analysis": 6, "sql": 4,
    "communication": 3, "leadership": 3, "internship": 4, "project": 3
}
def score_resume_text(text: str):
    text_l = text.lower()
    score = 0
    hits = {}
    for k, v in RESUME_KEYWORDS.items():
        if k in text_l:
            score += v
            hits[k] = v
    # Normalize to 100
    max_possible = sum(RESUME_KEYWORDS.values())
    normalized = min(100, round((score / max_possible) * 100, 2))
    return normalized, hits

# ----------------------------
# UI: Authentication flow
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

# Simple login/signup box
if not st.session_state.logged_in:
    st.sidebar.header("🔐 Login / Signup")
    mode = st.sidebar.radio("Mode", ["Login", "Signup"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Submit"):
        if mode == "Login":
            if check_credentials(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.sidebar.success("Logged in")
            else:
                st.sidebar.error("Invalid username or password")
        else:
            # signup
            if username in DEFAULT_USERS:
                st.sidebar.error("User exists. Choose a different username.")
            else:
                signup(username, password)
                st.sidebar.success("Account created. Please login.")
    st.markdown("""
        <small>Default admin: <b>admin</b> / adminpass · demo user: <b>user</b> / userpass</small>
    """, unsafe_allow_html=True)
    st.stop()

# When logged in, show main UI
user = st.session_state.user
st.sidebar.write(f"Logged in as: **{user}**")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.experimental_rerun()

# ----------------------------
# Navigation
# ----------------------------
page = st.sidebar.selectbox("Select Page", ["Dashboard", "Prediction", "Resume Scoring", "Admin", "Hosting Guide"])

# ----------------------------
# Load model on start if exists
# ----------------------------
model, features = load_model()

# ----------------------------
# Dashboard page
# ----------------------------
if page == "Dashboard":
    st.title("📊 Placement Dashboard (Visualization)")
    st.markdown("Upload dataset (CSV) or use default dataset file in project folder.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif os.path.exists(DATA_PATH):
        if st.checkbox("Load dataset from project folder"):
            df = pd.read_csv(DATA_PATH)
        else:
            df = None
    else:
        df = None

    if df is None:
        st.info("Please upload a CSV file or place Placement_Data_Full_Class.csv in project folder and check the box.")
    else:
        st.success("Dataset loaded")
        st.dataframe(df.head(100))
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.warning("No numeric columns detected for charts.")
        else:
            c1, c2 = st.columns([1,2])
            with c1:
                col = st.selectbox("Select numeric column", num_cols)
                bins = st.slider("Bins", 5, 100, 20)
                fig = px.histogram(df, x=col, nbins=bins)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("Correlation heatmap")
                fig, ax = plt.subplots(figsize=(6,5))
                sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

# ----------------------------
# Prediction page
# ----------------------------
if page == "Prediction":
    st.title("🎯 Predict Placement")
    st.markdown("Fill input fields (or upload CSV to batch predict). Model will use features saved with the model if available.")
    # If model exists, use saved features; else provide defaults
    sample_features = features if features else ['ssc_p','hsc_p','degree_p','etest_p','mba_p']
    st.write("Using features:", sample_features)

    # Single record inputs
    st.subheader("Single Student Prediction")
    # Simple inputs: adapt if gender/workex available in features
    inputs = {}
    for f in sample_features:
        inputs[f] = st.number_input(f"{f}", min_value=0.0, max_value=100.0, value=60.0)

    if st.button("Predict Single"):
        if model is None:
            st.warning("No trained model found. Please go to Admin -> Retrain model (upload dataset).")
        else:
            X = pd.DataFrame([inputs])
            # Keep only model features
            X = X[ sample_features ]
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
            st.success(f"Prediction: {'Placed' if pred==1 else 'Not Placed'}")
            if prob is not None:
                st.info(f"Confidence: {round(prob*100,2)}%")

            # PDF report
            if st.button("Generate PDF report"):
                buf = generate_pdf_report(inputs, int(pred), float(prob if prob is not None else 0.0))
                st.download_button("Download PDF report", buf, file_name="placement_report.pdf", mime="application/pdf")

    # Batch prediction via CSV
    st.subheader("Batch prediction (CSV)")
    up = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch")
    if up is not None:
        dfp = pd.read_csv(up)
        if model is None:
            st.warning("No model to predict. Retrain from Admin first.")
        else:
            # use available features intersection
            use_feats = [f for f in sample_features if f in dfp.columns]
            if not use_feats:
                st.error("Uploaded CSV doesn't contain required features.")
            else:
                Xb = dfp[use_feats].fillna(0)
                preds = model.predict(Xb)
                dfp['pred_status'] = preds
                st.dataframe(dfp.head(50))
                csv = dfp.to_csv(index=False).encode()
                st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")

# ----------------------------
# Resume scoring page
# ----------------------------
if page == "Resume Scoring":
    st.title("📝 Resume Scoring")
    st.markdown("Paste your resume text or upload a .txt/.pdf (pdf upload: text extraction not built-in). We'll score by keyword matching.")
    rtext = st.text_area("Paste resume text here", height=300)
    if st.button("Score Resume"):
        if not rtext.strip():
            st.error("Please paste resume text.")
        else:
            score, hits = score_resume_text(rtext)
            st.success(f"Resume score: {score} / 100")
            st.write("Matched keywords:", hits)
            st.write("Suggestions:")
            if score < 50:
                st.write("- Add technical keywords related to the job role (python, machine learning, sql).")
            else:
                st.write("- Good coverage. Add measurable outcomes and more projects.")

# ----------------------------
# Admin page
# ----------------------------
if page == "Admin":
    st.title("🔧 Admin Panel")
    st.markdown("Only admin users should use this. You can upload dataset and retrain the model here.")
    st.markdown("Current model: " + (MODEL_PATH if os.path.exists(MODEL_PATH) else "No model found"))
    up = st.file_uploader("Upload dataset CSV (to retrain)", type=["csv"], key="admin_upload")
    if up is not None:
        st.write("Dataset uploaded (preview):")
        dfr = pd.read_csv(up)
        st.dataframe(dfr.head())
        if st.button("Retrain model from uploaded dataset"):
            with st.spinner("Training model..."):
                clf, feats = train_model_from_df(dfr, save_path=MODEL_PATH)
                st.success("Model trained and saved.")
                st.write("Features used:", feats)
    # Allow admin to download current model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_bytes = f.read()
        st.download_button("Download current model file", model_bytes, file_name="placement_model.pkl")

# ----------------------------
# Hosting Guide
# ----------------------------
if page == "Hosting Guide":
    st.title("🚀 Hosting Guide")
    st.markdown("""
    **Options to host your Streamlit app**

    1. **Streamlit Community Cloud (share.streamlit.io)**  
       - Push your repo to GitHub (include requirements.txt).  
       - Create a new app on share.streamlit.io and link your repo.  
       - Pros: easiest for Streamlit apps. Free for public repos.

    2. **Render.com**  
       - Create a new Web Service, connect your GitHub repo, set start command: `streamlit run main_full.py --server.port $PORT`  
       - Pros: easy, supports private repos.

    3. **Heroku (deprecated complexity)**  
       - Use a Procfile with: `web: streamlit run main_full.py --server.port $PORT`  
       - Add runtime, requirements.txt.

    4. **Docker**  
       - Create Dockerfile, build image, then deploy to any container service (AWS ECS, GCP Cloud Run).

    **Important**: Add a `requirements.txt` with dependencies:
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    joblib
    fpdf2
    plotly
    seaborn
    matplotlib
    ```
    """)
    st.info("Tip: Ensure your model file (placement_model.pkl) is in the repo or retrain on first deploy.")

# ----------------------------
# End
# ----------------------------
st.markdown("---")
st.markdown("Made with ❤️ — Streamlit full app demo. Modify usernames/passwords and secure properly for production.")

