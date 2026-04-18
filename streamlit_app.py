import streamlit as st
import requests
import os

# Get API URL from environment or use Hugging Face Space
API_URL = os.getenv(
    "API_URL", 
    "https://hafsaimranattaria7115-student-performance-api.hf.space/analyze"
)

st.set_page_config(page_title="Student Analyzer", layout="wide")

st.title("🎓 Student Performance Analyzer")
st.write("Predict exam score and get optimization strategies.")

# Show current API endpoint
with st.expander("ℹ️ API Configuration"):
    st.info(f"**API Endpoint:** {API_URL}")
# ------------------ NUMERIC INPUTS ------------------
st.subheader("📊 Academic Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    hours_studied = st.slider("Avg. no. of hours studied per week", 0.0, 50.0, 20.0)
    attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0)

with col2:
    sleep_hours = st.slider("Avg. sleep hours per night", 0.0, 14.0, 7.0)
    previous_scores = st.slider("Previous Scores", 40.0, 100.0, 75.0)

with col3:
    tutoring_sessions = st.slider("Tutoring Sessions attended per month", 0.0, 10.0, 2.0)
    physical_activity = st.slider("Avg hrs. of Physical Activity per week", 0.0, 15.0, 3.0)

# ------------------ CATEGORICAL INPUTS ------------------
st.subheader("🧩 Lifestyle & Environment")

col4, col5, col6 = st.columns(3)

with col4:
    parental_involvement = st.selectbox("Parental Involvement in your Education", ["Low", "Medium", "High"])
    access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
    extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

with col5:
    motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])

with col6:
    teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
    school_type = st.selectbox("School Type", ["Public", "Private"])
    peer_influence = st.selectbox("Peer Influence on Academic Performance", ["Negative", "Neutral", "Positive"])

col7, col8 = st.columns(2)

with col7:
    learning_disabilities = st.selectbox("Learning Disabilities", ["Yes", "No"])
    parental_education = st.selectbox("Parental Education (Highest Level)", ["High School", "College", "Postgraduate"])

with col8:
    gender = st.selectbox("Gender", ["Male", "Female"])
    distance_from_home = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"])

# ------------------ BUTTON ------------------
if st.button("🚀 Analyze Performance"):

    payload = {
        "hours_studied": hours_studied,
        "attendance": attendance,
        "sleep_hours": sleep_hours,
        "previous_scores": previous_scores,
        "tutoring_sessions": tutoring_sessions,
        "physical_activity": physical_activity,
        "parental_involvement": parental_involvement,
        "access_to_resources": access_to_resources,
        "extracurricular": extracurricular,
        "motivation_level": motivation_level,
        "internet_access": internet_access,
        "family_income": family_income,
        "teacher_quality": teacher_quality,
        "school_type": school_type,
        "peer_influence": peer_influence,
        "learning_disabilities": learning_disabilities,
        "parental_education": parental_education,
        "gender": gender,
        "distance_from_home": distance_from_home
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()

            # ------------------ RESULT ------------------
            st.success("✅ Analysis Complete")

            score = data["predicted_exam_score"]

            st.subheader("📈 Predicted Score")
            st.metric(label="Exam Score", value=f"{score:.2f}")

            # Progress bar
            progress_value = min(int(score), 100) / 100
            st.progress(progress_value)

            # Display contextual message
            st.info(data.get("message", ""))

            # ------------------ STRATEGIES ------------------
            st.subheader("🛠️ Optimization Strategies")

            if len(data["optimization_steps"]) == 0:
                st.info("No major improvements needed. You're already optimized 🎯")
            else:
                for i, step in enumerate(data["optimization_steps"], 1):
                    # Create expandable section for each step
                    with st.expander(f"{i}. {step['strategy']} (Impact: +{step['impact']})"):
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            st.write(f"**Current Value:** {step['from_value']}")
                        
                        with col_right:
                            st.write(f"**Recommended:** {step['to_value']}")
                        
                        st.write(f"**Estimated Score Improvement:** +{step['impact']} points")

        else:
            st.error(f"❌ API Error: {response.text}")

    except requests.exceptions.Timeout:
        st.error("⏱️ Request Timeout: API took too long to respond. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error(f"🔌 Connection Error: Cannot reach API at {API_URL}")
        st.info("Make sure the backend API is running and the URL is correct.")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
        