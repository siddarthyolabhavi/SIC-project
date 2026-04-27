# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load
model = joblib.load("job_model.pkl")
le_target = joblib.load("target_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="AI Career Intelligence", layout="wide")

# HEADER
st.title("💼 AI Career Intelligence Dashboard")
st.caption("AI-powered job prediction, insights, and career guidance")

# SIDEBAR
st.sidebar.header("⚙️ Input Profile")

company_size = st.sidebar.selectbox("Company Size", ["Startup", "MNC"])
industry = st.sidebar.selectbox("Industry", ["Technology", "Healthcare", "Finance"])
country = st.sidebar.selectbox("Country", ["India", "Germany", "USA"])
remote_type = st.sidebar.selectbox("Work Type", ["Remote", "Hybrid", "Onsite"])
experience_level = st.sidebar.selectbox("Experience Level", ["Junior", "Mid", "Senior"])
education = st.sidebar.selectbox("Education", ["Bachelor", "Master", "PhD"])

years_exp = st.sidebar.slider("Years of Experience", 0, 20, 5)

st.sidebar.markdown("### 🧠 Skills")
skills = {
    "python": st.sidebar.checkbox("Python"),
    "sql": st.sidebar.checkbox("SQL"),
    "ml": st.sidebar.checkbox("Machine Learning"),
    "dl": st.sidebar.checkbox("Deep Learning"),
    "cloud": st.sidebar.checkbox("Cloud")
}

predict_btn = st.sidebar.button("🚀 Predict Role")

# MAIN TABS
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Insights", "📈 Analytics"])

if predict_btn:

    # Input
    input_data = pd.DataFrame([{
        "company_size": company_size,
        "company_industry": industry,
        "country": country,
        "remote_type": remote_type,
        "experience_level": experience_level,
        "years_experience": years_exp,
        "education_level": education,
        "skills_python": int(skills["python"]),
        "skills_sql": int(skills["sql"]),
        "skills_ml": int(skills["ml"]),
        "skills_deep_learning": int(skills["dl"]),
        "skills_cloud": int(skills["cloud"]),
        "salary": 100000,
        "job_posting_month": 6,
        "job_posting_year": 2025,
        "hiring_urgency": "Medium",
        "job_openings": 5
    }])

    # Encode
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Predict
    probs = model.predict_proba(input_data)[0]
    pred = np.argmax(probs)
    confidence = probs[pred]
    role = le_target.inverse_transform([pred])[0]

    # ===== TAB 1 =====
    with tab1:
        col1, col2 = st.columns(2)
        col1.metric("🎯 Role", role)
        col2.metric("📊 Confidence", f"{confidence*100:.2f}%")

        st.progress(int(confidence * 100))

        if confidence < 0.5:
            st.warning("⚠️ Low confidence prediction")

        st.markdown("### 🔝 Top Matches")
        top_idx = probs.argsort()[-3:][::-1]

        for i in top_idx:
            r = le_target.inverse_transform([i])[0]
            st.write(f"{r} → {probs[i]*100:.2f}%")

    # ===== TAB 2 =====
    with tab2:
        st.markdown("### 📚 Skill Recommendations")

        if role == "Data":
            st.success("Python, SQL, Machine Learning")
        elif role == "Engineer":
            st.success("System Design, Cloud, Backend")

        st.info(f"💰 Estimated Salary: ${80000 + confidence*70000:.0f}")

    # ===== TAB 3 =====
    with tab3:
        st.markdown("### 📊 Feature Importance")

        try:
            importances = model.estimators_[0].feature_importances_
            features = model_columns[:len(importances)]

            fig, ax = plt.subplots()
            ax.barh(features[:10], importances[:10])
            st.pyplot(fig)
        except:
            st.write("Feature importance not available")