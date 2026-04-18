"""
predictor.py — Loads trained model artifacts and exposes predict + recommend functions.
"""
import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any

_model = None
_scaler = None
_meta = None

NUMERIC_FEATURES = None
CATEGORICAL_FEATURES = None
ALL_FEATURES = None
ENCODERS = None
# Load from main folder (where train_model.py saves)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "model.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "scaler.pkl")
META_PATH = os.path.join(ROOT_DIR, "model_meta.pkl")
def load_artifacts():
    global _model, _scaler, _meta
    global NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES, ENCODERS

    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    if _meta is None:
        _meta = joblib.load(META_PATH)

        NUMERIC_FEATURES = _meta["numeric_features"]
        CATEGORICAL_FEATURES = _meta["categorical_features"]
        ALL_FEATURES = _meta["features"]
        ENCODERS = _meta["encoders"]

    return _model, _scaler, _meta


# ── 2. PREPROCESS ────────────────────────────────────────────────────────
def preprocess(data: Dict[str, Any]) -> np.ndarray:
    """Convert raw input into scaled feature vector."""

    row = {
        # Numeric
        "Hours_Studied": float(data.get("hours_studied", 20)),
        "Attendance": float(data.get("attendance", 85)),
        "Sleep_Hours": float(data.get("sleep_hours", 7)),
        "Previous_Scores": float(data.get("previous_scores", 75)),
        "Tutoring_Sessions": float(data.get("tutoring_sessions", 2)),
        "Physical_Activity": float(data.get("physical_activity", 3)),

        # Categorical
        "Parental_Involvement": str(data.get("parental_involvement", "Medium")),
        "Access_to_Resources": str(data.get("access_to_resources", "Medium")),
        "Extracurricular_Activities": str(data.get("extracurricular", "No")),
        "Motivation_Level": str(data.get("motivation_level", "Medium")),
        "Internet_Access": str(data.get("internet_access", "Yes")),
        "Family_Income": str(data.get("family_income", "Medium")),
        "Teacher_Quality": str(data.get("teacher_quality", "Medium")),
        "School_Type": str(data.get("school_type", "Public")),
        "Peer_Influence": str(data.get("peer_influence", "Neutral")),
        "Learning_Disabilities": str(data.get("learning_disabilities", "No")),
        "Parental_Education_Level": str(data.get("parental_education", "College")),
        "Gender": str(data.get("gender", "Male")),
        "Distance_from_Home": str(data.get("distance_from_home", "Near"))
    }

    vector = []

    for feat in ALL_FEATURES:
        val = row[feat]

        if feat in CATEGORICAL_FEATURES:
            le = ENCODERS[feat]
            val = str(val)

            if val not in le.classes_:
                val = le.classes_[0]

            val = int(le.transform([val])[0])

        vector.append(float(val))

    df = pd.DataFrame([dict(zip(ALL_FEATURES, vector))], columns=ALL_FEATURES)
    return _scaler.transform(df)



# ── 3. PREDICT ───────────────────────────────────────────────────────────
def predict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict exam score."""
    X = preprocess(data)
    score = float(_model.predict(X)[0])

    score = round(np.clip(score, 40.0, 100.0), 2)

    return {
        "predicted_exam_score": score
    }

def _get_strategy_label(feature: str) -> str:
    """Helper to get human-readable strategy labels."""
    labels = {
        "hours_studied": "Increase Study Time",
        "attendance": "Improve Attendance",
        "previous_scores": "Strengthen Basics",
        "tutoring_sessions": "Increase Tutoring",
        "sleep_hours": "Improve Sleep Quality",
        "extracurricular": "Join Extracurricular Activities",
        "motivation_level": "Boost Motivation Level",
        "internet_access": "Secure Internet Access",
        "learning_disabilities": "Seek Help for Learning Disabilities"
    }
    return labels.get(feature, feature)


def recommend(data: Dict[str, Any]) -> Dict[str, Any]:
    """Multi-feature optimization with smart recommendations and contextual messages."""
    
    base_score = predict(data)["predicted_exam_score"]
    
    # Determine max steps and initial message based on score
    if base_score >= 85:
        max_steps = 1
        initial_message = "🌟 Excellent Performance! You're doing great! Here's one small optimization if you want to push even further:"
    elif base_score >= 75:
        max_steps = 4
        initial_message = "✅ Good Job! You're on the right track. Here are a few tweaks to push your score even higher:"
    else:
        max_steps = 10
        initial_message = "📈 You have potential to improve significantly! Here's a roadmap to boost your performance:"
    
    best_data = dict(data)
    best_score = base_score
    
    numeric_tweaks = {
        "hours_studied": (5, 35),
        "attendance": (10, 100),
        "previous_scores": (10, 95),
        "tutoring_sessions": (2, 5),
        "sleep_hours": (1, 8),
    }
    
    categorical_tweaks = {
        "extracurricular": ["No", "Yes"],
        "motivation_level": ["Low", "Medium", "High"],
        "internet_access": ["No", "Yes"],
        "learning_disabilities": ["Yes", "No"],
    }
    
    steps_taken = []
    improved_features = set()  # Track which features have been improved
    
    # Iterate to find best improvements
    for iteration in range(max_steps):
        best_improvement = None
        best_improvement_key = None
        best_trial_data = None
        best_new_value = None
        
        # Test ALL numeric tweaks
        for key, (increment, max_val) in numeric_tweaks.items():
            if key in improved_features:  # ← SKIP if already improved
                continue
                
            current_val = float(best_data.get(key, 0))
            
            if current_val >= max_val:
                continue
            
            trial = dict(best_data)
            trial[key] = min(current_val + increment, max_val)
            trial_score = predict(trial)["predicted_exam_score"]
            
            improvement = trial_score - best_score
            
            if improvement > 0:
                if best_improvement is None or improvement > best_improvement:
                    best_improvement = improvement
                    best_improvement_key = key
                    best_trial_data = trial
                    best_new_value = trial[key]
        
        # Test ALL categorical tweaks
        for key, options in categorical_tweaks.items():
            if key in improved_features:  # ← SKIP if already improved
                continue
                
            current_val = best_data.get(key, options[0])
            
            for new_val in options:
                if new_val == current_val:
                    continue
                
                trial = dict(best_data)
                trial[key] = new_val
                trial_score = predict(trial)["predicted_exam_score"]
                
                improvement = trial_score - best_score
                
                if improvement > 0:
                    if best_improvement is None or improvement > best_improvement:
                        best_improvement = improvement
                        best_improvement_key = key
                        best_trial_data = trial
                        best_new_value = new_val
        
        # Apply best improvement if found
        if best_improvement_key:
            old_val = best_data.get(best_improvement_key)
            delta = round(best_improvement, 2)
            
            steps_taken.append({
                "feature": best_improvement_key,
                "from_value": old_val,
                "to_value": best_new_value,
                "impact": delta,
                "strategy": _get_strategy_label(best_improvement_key)
            })
            
            best_data = best_trial_data
            best_score = best_score + best_improvement
            improved_features.add(best_improvement_key)  # ← MARK as improved
        else:
            # No improvement found
            break
    
    # Build closing message
    if steps_taken:
        total_potential_improvement = round(best_score - base_score, 2)
        if base_score >= 85:
            closing_message = f"💡 This small adjustment could help you reach {round(best_score, 2)} (+{total_potential_improvement} points)!"
        elif base_score >= 75:
            closing_message = f"💡 By implementing these {len(steps_taken)} changes, you could potentially reach a score of {round(best_score, 2)} (+{total_potential_improvement} points)!"
        else:
            closing_message = f"💪 Focus on these {len(steps_taken)} areas. By implementing all recommendations, you could potentially reach {round(best_score, 2)} (+{total_potential_improvement} points)!"
    else:
        closing_message = "You're already doing well! No further improvements identified at this moment."
    
    # Combine messages
    full_message = initial_message + "\n\n" + closing_message
    
    return {
        "predicted_exam_score": base_score,
        "optimization_steps": steps_taken,
        "message": full_message
    }