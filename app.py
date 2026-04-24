import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="ZenithMind AI", page_icon="🧠", layout="centered")


# -------------------------------------------------
# Load trained model and feature list
# -------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("xgb_model.pkl")
        features = joblib.load("features.pkl")
        return model, features
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None


model, features = load_assets()


# -------------------------------------------------
# App Title
# -------------------------------------------------
st.title("🧠 ZenithMind AI")
st.subheader("Student Burnout Prediction System")
st.write("Enter your academic and lifestyle metrics to estimate burnout risk.")

st.divider()


# -------------------------------------------------
# Input Section
# -------------------------------------------------
st.markdown("### 📊 Student Metrics")

col1, col2 = st.columns(2)

# Left side inputs
with col1:
    study_hours = st.number_input("Study Hours / Day", 0.0, 16.0, 7.0)
    sleep_hours = st.number_input("Sleep Hours / Day", 0.0, 12.0, 7.0)
    physical_activity = st.number_input("Exercise Hours / Day", 0.0, 5.0, 1.0)
    screen_time = st.number_input("Screen Time / Day", 0.0, 16.0, 4.0)
    academic_performance = st.number_input("Academic Performance (1-10)", 1.0, 10.0, 6.0)

# Right side inputs
with col2:
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 4)
    exam_pressure = st.slider("Exam Pressure (1-10)", 1, 10, 5)
    anxiety_score = st.slider("Anxiety Score (1-10)", 1, 10, 3)
    depression_score = st.slider("Depression Score (1-10)", 1, 10, 2)
    social_support = st.slider("Social Support (1-3)", 1, 3, 2)
    financial_stress = st.slider("Financial Stress (1-10)", 1, 10, 3)
    family_expectation = st.slider("Family Expectation (1-10)", 1, 10, 5)


st.divider()


# -------------------------------------------------
# Prediction Button
# -------------------------------------------------
predict_btn = st.button("Generate Wellness Report")


if predict_btn:

    if model is None:
        st.warning("Model files not loaded properly.")
    
    else:

        # ------------------------------------------
        # Derived features (same as used in training)
        # ------------------------------------------
        stress_sleep_ratio = round(stress_level / sleep_hours if sleep_hours > 0 else 0, 2)
        pressure_gap = exam_pressure - stress_level
        academic_overload = round(study_hours * exam_pressure, 2)


        # ------------------------------------------
        # Create input dictionary
        # ------------------------------------------
        user_input = {
            "study_hours_per_day": study_hours,
            "exam_pressure": exam_pressure,
            "academic_performance": academic_performance,
            "stress_level": stress_level,
            "anxiety_score": anxiety_score,
            "depression_score": depression_score,
            "sleep_hours": sleep_hours,
            "physical_activity": physical_activity,
            "social_support": social_support,
            "screen_time": screen_time,
            "financial_stress": financial_stress,
            "family_expectation": family_expectation,
            "stress_sleep_ratio": stress_sleep_ratio,
            "pressure_gap": pressure_gap,
            "academic_overload": academic_overload
        }


        # Convert input to dataframe
        data = pd.DataFrame([user_input])


        # Ensure same feature order as training
        data = data[features]


        # ------------------------------------------
        # Model prediction
        # ------------------------------------------
        prediction = model.predict(data)[0]
        score = round(float(prediction), 2)


        st.divider()
        st.subheader("📋 Wellness Report")


        # ------------------------------------------
        # Show burnout risk level
        # ------------------------------------------
        if score < 1.5:
            st.success(f"Low Burnout Risk (Score: {score})")
        elif score < 2.5:
            st.warning(f"Moderate Burnout Risk (Score: {score})")
        else:
            st.error(f"High Burnout Risk (Score: {score})")


        # ------------------------------------------
        # Burnout Risk Meter (Gauge Chart)
        # ------------------------------------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Burnout Risk Meter"},

            gauge={
                'axis': {'range': [0, 4]},

                'steps': [
                    {'range': [0, 1.5], 'color': "lightgreen"},
                    {'range': [1.5, 2.5], 'color': "yellow"},
                    {'range': [2.5, 4], 'color': "red"}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)


        # ------------------------------------------
        # Simple Lifestyle Visualization
        # ------------------------------------------
        st.subheader("Student Lifestyle Overview")

        lifestyle_data = {
            "Study Hours": study_hours,
            "Sleep Hours": sleep_hours,
            "Exercise": physical_activity,
            "Screen Time": screen_time
        }

        lifestyle_df = pd.DataFrame(
            list(lifestyle_data.items()),
            columns=["Metric", "Hours"]
        )

        st.bar_chart(lifestyle_df.set_index("Metric"))


        # ------------------------------------------
        # Basic wellness recommendation
        # ------------------------------------------
        st.subheader("Wellness Advice")

        if sleep_hours < 6:
            st.write("• Try to increase your sleep duration for better recovery.")

        if screen_time > 8:
            st.write("• High screen time detected. Consider reducing device usage.")

        if physical_activity < 1:
            st.write("• Regular physical activity can help reduce stress levels.")

        if stress_level > 7:
            st.write("• Stress level is high. Consider relaxation techniques or counseling.")

        if score < 1.5:
            st.write("• Keep maintaining your current lifestyle balance.")


st.divider()

st.caption("ZenithMind AI | B.Tech CSE Project | Student Burnout Prediction")