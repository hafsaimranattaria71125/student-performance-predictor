"""
train_model.py - Trains Linear Regression to predict Exam Score from Student Performance Dataset
"""

import os, pandas as pd, numpy as np, joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "StudentPerformanceFactors.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
META_PATH   = os.path.join(BASE_DIR, "model_meta.pkl")
TEST_INDICES_PATH = os.path.join(BASE_DIR, "test_indices.npy")

# Numeric features
NUMERIC_FEATURES = [
    "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores",
    "Tutoring_Sessions", "Physical_Activity"
]

# Categorical features
CATEGORICAL_FEATURES = [
    "Parental_Involvement", "Access_to_Resources", "Extracurricular_Activities",
    "Motivation_Level", "Internet_Access", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Learning_Disabilities",
    "Parental_Education_Level", "Gender", "Distance_from_Home"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def load_and_clean(path):
    df = pd.read_csv(path)
    
    # Clean numeric columns
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                print(f"      Filling {df[col].isna().sum()} NaN values in {col} with median")
                df[col] = df[col].fillna(df[col].median())
    
    # Clean categorical columns
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            if col == "Distance_from_Home":
                valid_cats = ["Near", "Moderate", "Far"]
                df[col] = df[col].apply(lambda x: x if x in valid_cats else "Near")
    
    df = df.dropna(subset=["Exam_Score"])
    print(f"      Final rows: {len(df)}")
    return df

def encode_categoricals(df, encoders=None):
    fit = encoders is None
    if fit:
        encoders = {}
    
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            unique_vals = df[col].astype(str).unique()
            le.fit(unique_vals)
            df[col] = le.transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str)
            known_classes = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            df[col] = le.transform(df[col])
    
    return df, encoders

def train():
    print("="*50)
    print("  Student Performance Prediction — Training")
    print("="*50)
    
    print("\n[1/5] Loading & cleaning data …")
    df = load_and_clean(DATA_PATH)
    
    print("[2/5] Encoding categoricals …")
    X = df[ALL_FEATURES].copy()
    y = df["Exam_Score"]
    
    X, encoders = encode_categoricals(X)
    
    # Final check for NaN values
    if X.isna().any().any():
        print("      Warning: NaN found in features, applying imputer...")
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    print("[3/5] Scaling features …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("[4/5] Splitting & training (80/20) …")
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, range(len(y)), test_size=0.2, random_state=42
    )
    
    # Save test indices for later evaluation
    np.save(TEST_INDICES_PATH, idx_test)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("[5/5] Evaluating on test set …")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    within_1 = np.mean(np.abs(y_test - y_pred) <= 1) * 100
    within_2 = np.mean(np.abs(y_test - y_pred) <= 2) * 100
    within_5 = np.mean(np.abs(y_test - y_pred) <= 5) * 100
    
    print(f"\n  📊 Test Set Performance:")
    print(f"  MSE   : {mse:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  R²    : {r2:.4f}")
    print(f"\n  🎯 Prediction Accuracy:")
    print(f"    ±1 point:  {within_1:.1f}%")
    print(f"    ±2 points: {within_2:.1f}%")
    print(f"    ±5 points: {within_5:.1f}%")
    
    # Store metadata
    meta = {
        "features": ALL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "encoders": encoders,
        "feature_importance": dict(zip(ALL_FEATURES, np.abs(model.coef_))),
        "metrics": {
            "mse": round(mse,4), 
            "rmse": round(rmse,4), 
            "r2": round(r2,4),
            "mae": round(mae,4),
            "within_1_percent": round(within_1,1),
            "within_2_percent": round(within_2,1),
            "within_5_percent": round(within_5,1)
        },
        "target": "Exam_Score",
        "test_size": len(y_test),
        "train_size": len(y_train)
    }
    
    # Save with compression for compatibility
    joblib.dump(model, MODEL_PATH, compress=3)
    joblib.dump(scaler, SCALER_PATH, compress=3)
    joblib.dump(meta, META_PATH, compress=3)
    
    print(f"\n  💾 Saved model, scaler, meta to {BASE_DIR}")
    print(f"  📁 Test indices saved to {TEST_INDICES_PATH}")
    print(f"  🔢 NumPy version used: {np.__version__}")
    print("\nTraining complete ✓")
    
    return model, scaler, meta

def evaluate_on_test():
    """Evaluate using the exact same test data from training"""
    print("\n" + "="*50)
    print("  Re-evaluating Model on Original Test Data")
    print("="*50)
    
    # Check if test indices exist
    if not os.path.exists(TEST_INDICES_PATH):
        print("  ⚠️ No saved test indices found. Run train() first.")
        return
    
    # Load model and preprocessors
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    meta = joblib.load(META_PATH)
    encoders = meta["encoders"]
    
    # Load original data
    df = load_and_clean(DATA_PATH)
    X = df[ALL_FEATURES].copy()
    y = df["Exam_Score"]
    
    # Encode and scale
    X, _ = encode_categoricals(X, encoders)
    X_scaled = scaler.transform(X)
    
    # Load test indices from training
    test_indices = np.load(TEST_INDICES_PATH)
    
    # Extract exact test data
    X_test = X_scaled[test_indices]
    y_test = y.iloc[test_indices]
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    within_1 = np.mean(np.abs(y_test - y_pred) <= 1) * 100
    within_2 = np.mean(np.abs(y_test - y_pred) <= 2) * 100
    within_5 = np.mean(np.abs(y_test - y_pred) <= 5) * 100
    
    print(f"\n  Test set size: {len(y_test)} samples")
    print(f"\n  📊 Metrics:")
    print(f"  MSE   : {mse:.4f}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  R²    : {r2:.4f}")
    print(f"\n  🎯 Accuracy:")
    print(f"    ±1 point:  {within_1:.1f}%")
    print(f"    ±2 points: {within_2:.1f}%")
    print(f"    ±5 points: {within_5:.1f}%")
    
    print(f"\n  📈 Sample predictions (first 15 test samples):")
    print(f"  {'Actual':>8} {'Predicted':>10} {'Error':>10}")
    print(f"  {'-'*32}")
    for i in range(min(15, len(y_test))):
        error = y_test.iloc[i] - y_pred[i]
        print(f"  {y_test.iloc[i]:>8.1f} {y_pred[i]:>10.2f} {error:>10.2f}")
    
    print(f"\n  ✅ These are the EXACT same test samples used during training.")

if __name__ == "__main__":
    train()
    evaluate_on_test()