# 💼 AI Career Intelligence Dashboard

An end-to-end **Machine Learning + Dashboard project** that predicts job roles based on user profile, skills, and experience.  
It also provides **confidence scores, career insights, and skill recommendations**, making it a decision-support tool rather than just a predictor.

---

## 🚀 Features

- 🎯 **Job Role Prediction**
- 📊 **Confidence Score with Probability**
- 🔝 **Top 3 Matching Roles**
- 📚 **Skill Recommendations**
- 💰 **Estimated Salary Insight**
- 📈 **Feature Importance Visualization**
- 🧠 **Ensemble ML Model (XGBoost + Random Forest)**
- 🎨 **Modern Dashboard UI using Streamlit**

---

## 🧠 Machine Learning Approach

### ✔ Model
- Ensemble model combining:
  - XGBoost
  - Random Forest

### ✔ Techniques Used
- Data preprocessing & cleaning  
- Label Encoding (target variable)  
- One-Hot Encoding (features)  
- Feature engineering  
- Ensemble learning (Voting Classifier)  

### ✔ Output
- Predicted Job Role  
- Confidence Score  
- Top 3 Role Probabilities  

---

## 📂 Dataset

- **AI Job Market Dataset**
- Contains:
  - Job Title
  - Company Info
  - Skills (Python, SQL, ML, etc.)
  - Experience Level
  - Salary
  - Location & Work Type

---

## 🖥️ Dashboard Preview

### 🎯 Prediction View
- Displays predicted role + confidence  
- Shows top matching roles  

### 📊 Insights View
- Skill recommendations  
- Salary estimation  

### 📈 Analytics View
- Feature importance visualization  

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-career-dashboard.git
cd ai-career-dashboard
pip install -r requirements.txt
python model.py
streamlit run app.py
├── model.py                 # ML model training
├── app.py                   # Streamlit dashboard
├── requirements.txt         # Dependencies
├── AI Job Market Dataset.csv
├── job_model.pkl            # Saved model
├── target_encoder.pkl       # Label encoder
├── model_columns.pkl        # Feature columns
└── README.md
