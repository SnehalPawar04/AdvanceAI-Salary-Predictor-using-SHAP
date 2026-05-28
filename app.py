import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Page setup
st.set_page_config(page_title="AI Salary Predictor", layout="centered")

st.title("💼 AI Salary Predictor (Real World Project)")

# ---------------- LOAD DATA ----------------
data = pd.read_csv("Salary Data.csv")

# Show dataset
st.subheader("📊 Dataset Preview")
st.write(data.head())

# Remove missing values
data = data.dropna()

# ---------------- ENCODING ----------------
# Gender encoding
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Education encoding
data['Education Level'] = data['Education Level'].astype('category')
edu_categories = data['Education Level'].cat.categories
data['Education Level'] = data['Education Level'].cat.codes

# Job Title encoding
data['Job Title'] = data['Job Title'].astype('category')
job_categories = data['Job Title'].cat.categories
data['Job Title'] = data['Job Title'].cat.codes

# ---------------- MODEL ----------------
X = data[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']]
y = data['Salary']

model = RandomForestRegressor()
model.fit(X, y)

# ---------------- USER INPUT ----------------
st.sidebar.header("Enter Details")

age = st.sidebar.slider("Age", 18, 60, 25)
exp = st.sidebar.slider("Years of Experience", 0, 40, 1)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox("Education Level", list(edu_categories))
job = st.sidebar.selectbox("Job Title", list(job_categories))

# Convert inputs
gender_val = 0 if gender == "Male" else 1
edu_val = list(edu_categories).index(education)
job_val = list(job_categories).index(job)

# Input dataframe
input_data = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_val],
    'Education Level': [edu_val],
    'Job Title': [job_val],
    'Years of Experience': [exp]
})

# ---------------- PREDICTION ----------------
prediction = model.predict(input_data)

st.subheader("💰 Predicted Salary")
st.success(f"{prediction[0]:.2f}")

# ---------------- ACCURACY ----------------
y_pred = model.predict(X)
accuracy = r2_score(y, y_pred)

st.subheader("📈 Model Accuracy")
st.write(f"R² Score: {accuracy:.2f}")

# ---------------- GRAPH ----------------
st.subheader("📊 Actual vs Predicted Salary")

fig, ax = plt.subplots()
ax.scatter(y, y_pred)
ax.set_xlabel("Actual Salary")
ax.set_ylabel("Predicted Salary")
ax.set_title("Actual vs Predicted")

st.pyplot(fig)
plt.close(fig)

# ---------------- SHAP ----------------
explainer = shap.Explainer(model, X)
shap_values = explainer(input_data)

st.subheader("🧠 Why this prediction?")

fig, ax = plt.subplots()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)
plt.close(fig)