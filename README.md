# 🎓 Student Performance Predictor

ML-powered app to predict student exam scores and get personalized improvement recommendations.

## 🚀 Quick Start

### Option 1: Use Deployed App (Recommended) ✨

**No installation needed!** Just visit:
- 🌐 **Streamlit App**: https://student-performance-predictor-n3tbjn34cx4jthemsvyhqt.streamlit.app/
- 📚 **API Docs**: https://hafsaimranattaria7115-student-performance-api.hf.space/docs

---

### Option 2: Run Locally

### Install
```bash
git clone https://github.com/hafsaimranattaria71125/student-performance-predictor.git
cd student-performance-predictor
pip install -r requirements.txt
```

### Run
**Terminal 1** - Start API:
```bash
uvicorn main:app --reload --port 8000
```

**Terminal 2** - Start UI:
```bash
streamlit run streamlit_app.py
```

- API: http://127.0.0.1:8000
- UI: http://localhost:8501

## ✨ Features

- 🎯 **Predict** exam scores based on 19 student factors
- 💡 **Smart Recommendations** tailored to each student
- 📈 **Impact Analysis** showing score improvement (+X points)
- 📊 **Interactive UI** with sliders and dropdowns

## 📊 What It Analyzes

**Numeric Features**: Hours studied, attendance, sleep, previous scores, tutoring, physical activity

**Categorical Features**: Motivation level, extracurricular activities, internet access, learning disabilities, parental involvement, family income, school type, etc.

## 📤 API Example

**POST** `/analyze`

```json
{
  "hours_studied": 20.0,
  "attendance": 85.0,
  "sleep_hours": 7.0,
  "previous_scores": 75.0,
  "tutoring_sessions": 2.0,
  "physical_activity": 3.0,
  "parental_involvement": "Medium",
  "motivation_level": "Medium",
  "extracurricular": "No",
  "internet_access": "Yes",
  "family_income": "Medium",
  "teacher_quality": "Medium",
  "school_type": "Public",
  "peer_influence": "Neutral",
  "learning_disabilities": "No",
  "parental_education": "College",
  "gender": "Male",
  "access_to_resources": "Medium",
  "distance_from_home": "Near"
}
```

**Response**:
```json
{
  "predicted_exam_score": 67.23,
  "message": "📈 You have potential to improve significantly!",
  "optimization_steps": [
    {
      "feature": "attendance",
      "strategy": "Improve Attendance",
      "from_value": 85,
      "to_value": 100,
      "impact": 5.2
    },
    {
      "feature": "hours_studied",
      "strategy": "Increase Study Time",
      "from_value": 20,
      "to_value": 35,
      "impact": 3.8
    }
  ]
}
```

## 🛠 Tech Stack

- **Backend**: FastAPI, scikit-learn, pandas
- **Frontend**: Streamlit
- **Validation**: Pydantic

## 📁 Project Structure

```
student_grade_api/
├── main.py                          # FastAPI application
├── streamlit_app.py                 # Streamlit web UI
├── model.py                         # ML model & recommendation logic
├── schemas.py                       # Pydantic data models
├── train_model.py                   # Model training script
├── StudentPerformanceFactors.csv    # Training dataset
├── model.pkl                        # Trained ML model
├── scaler.pkl                       # Feature scaler
├── model_meta.pkl                   # Model metadata
├── test_indices.npy                 # Test set indices
├── requirements.txt                 # Python dependencies
├── runtime.txt                      # Python version info
├── README.md                        # This file
└── __pycache__/                     # Compiled Python cache
```

## 🎯 How It Works

1. Student inputs 19 factors
2. ML model predicts exam score
3. Smart recommendation engine finds best improvements
4. Returns score + actionable steps with impact estimates

## 🤝 Retrain Model

```bash
python train_model.py
```

## 📝 License

Open source - feel free to use and modify!

## 👤 Author

Hafsa Imran - [@hafsaimranattaria71125](https://github.com/hafsaimranattaria71125)

---

**Questions?** Check `/docs` endpoint or open an issue!
