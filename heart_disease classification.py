import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, accuracy_score,
    roc_curve, auc,
    precision_score, recall_score
)
import warnings
warnings.filterwarnings("ignore")

#  Feature metadata for chatbot 
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# Correct thalassemia mapping for Cleveland dataset
FEATURE_PROMPTS = [
    ("Patient Age (years)",                                float, 1,   120),
    ("Biological Sex (1=Male, 0=Female)",                int,   0,   1  ),
    ("Chest Pain Type (0=Typical 1=Atypical 2=Non-anginal 3=Asymptomatic)", int, 0, 3),
    ("Resting Blood Pressure mm/Hg",                    float, 50,  250),
    ("Serum Cholesterol mg/dl",                         float, 100, 600),
    ("Fasting Blood Sugar >120 mg/dl (1=Yes 0=No)",     int,   0,   1  ),
    ("Resting ECG (0=Normal 1=ST-T abnormality 2=LVH)", int,   0,   2  ),
    ("Max Heart Rate Achieved",                         float, 50,  250),
    ("Exercise Induced Angina (1=Yes 0=No)",            int,   0,   1  ),
    ("ST Depression / Oldpeak (e.g. 1.5)",              float, 0,   10 ),
    ("Slope of Peak ST Segment (0=Up 1=Flat 2=Down)",   int,   0,   2  ),
    ("Major Vessels Colored by Fluoroscopy (0-4)",      int,   0,   4  ),
    ("Thalassemia (3=Normal 6=Fixed Defect 7=Reversible Defect)", int, 3, 7),
]

# BANNER

def print_banner() -> None:
    print("\n" + "=" * 70)
    print("   DecodeLabs   |   Project 2   |   Heart Disease Classifier")
    print("   Dataset     :   Cleveland UCI — 303 Real Patient Records")
    print("   Algorithm   :   Random Forest + Clinical Rules")
    print("=" * 70)

# STEP 1 – LOAD DATASET
def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV from local path or fallback to GitHub mirror."""
    try:
        df = pd.read_csv(path)
        print(f"\n[1] Dataset loaded from local file: {path}")
    except FileNotFoundError:
        FALLBACK = (
            "https://raw.githubusercontent.com/sharmaroshan/"
            "Heart-UCI-Dataset/master/heart.csv"
        )
        print(f"\n[1] Local file not found. Fetching from GitHub mirror...")
        df = pd.read_csv(FALLBACK)
        print(f"    Loaded from: {FALLBACK}")
    
    # Handle missing values
    print(f"\n    Before cleaning: {df.isnull().sum().sum()} missing values")
    df = df.dropna()
    print(f"    After cleaning: {len(df)} samples remaining")
    
    return df

# STEP 2 – TRAIN RANDOM FOREST MODEL
def train_model(df: pd.DataFrame):
    """Train Random Forest model pipeline."""
    
    X: np.ndarray = df[FEATURE_NAMES].values
    y: np.ndarray = df["target"].values
    
    print(f"\n    Samples    : {len(df)}")
    print(f"    Features   : {len(FEATURE_NAMES)}")
    print(f"    No Disease : {int(np.sum(y == 0))}  |  "
          f"Disease : {int(np.sum(y == 1))}")
    
    # Age distribution warning
    young_healthy = df[(df['age'] < 35) & (df['target'] == 0)]
    young_diseased = df[(df['age'] < 35) & (df['target'] == 1)]
    print(f"\n    ⚠  Dataset Note: Only {len(young_healthy)} healthy patients under 35")
    print(f"       (vs {len(young_diseased)} diseased patients under 35)")
    
    # ── Scaling ──────────────────────────────────────────────
    scaler: StandardScaler = StandardScaler()
    X_scaled: np.ndarray   = scaler.fit_transform(X)
    
    print(f"\n[2] StandardScaler applied")
    
    # ── Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.20,
        random_state=42,
        shuffle=True,
        stratify=y
    )
    print(f"\n[3] Train-Test Split → Train: {len(X_train)} | Test: {len(X_test)}")
    
    # ── Random Forest Model ───────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print(f"\n[4] Random Forest model trained successfully")
    
    # ── Evaluation ───────────────────────────────────────────
    y_pred: np.ndarray = model.predict(X_test)
    y_prob: np.ndarray = model.predict_proba(X_test)[:, 1]
    
    acc     = float(accuracy_score(y_test, y_pred))
    f1      = float(f1_score(y_test, y_pred))
    prec    = float(precision_score(y_test, y_pred))
    rec     = float(recall_score(y_test, y_pred))
    cm      = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _    = roc_curve(y_test, y_prob)
    roc_auc        = float(auc(fpr, tpr))
    
    print(f"\n[5] Evaluation Results")
    print(f"    Accuracy  : {acc*100:.2f} %")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"    CM        → TP:{tp}  TN:{tn}  FP:{fp}  FN:{fn}")
    
    return {
        "model": model,
        "scaler": scaler,
        "acc": acc, "f1": f1, "prec": prec, "rec": rec,
        "roc_auc": roc_auc
    }

# STEP 3 – CLINICAL RULES ENGINE (Fixes the young healthy issue)
def apply_clinical_rules(values: list[float], ml_prediction: int, ml_confidence: float) -> tuple[int, float, str]:
    """
    Apply clinical override rules to fix dataset bias.
    Returns: (final_prediction, final_confidence, reason)
    """
    
    age = values[0]
    cp = values[2]
    oldpeak = values[9]
    thalach = values[7]
    ca = values[11]
    thal = values[12]
    exang = values[8]
    trestbps = values[3]
    chol = values[4]
    
    # RULE 1: Young, healthy athlete override
    if age < 35 and oldpeak == 0 and thalach > 180 and ca == 0 and thal == 3:
        return (0, 95.0, "Young athlete with excellent exercise capacity — clinical override")
    
    # RULE 2: Very low probability profile (age < 40, no risk factors)
    if age < 40 and oldpeak < 0.5 and ca == 0 and thal == 3 and exang == 0:
        if trestbps < 140 and chol < 240:
            return (0, 92.0, "Very low risk profile (age < 40, no major risk factors)")
    
    # RULE 3: Severe disease indicators (override ML false negatives)
    if age > 50 and ca >= 2 and thal in [6, 7] and oldpeak > 1.5:
        return (1, 94.0, "Multiple severe risk factors detected")
    
    # RULE 4: Atypical chest pain in young patients is often NOT cardiac
    if age < 40 and cp == 1 and oldpeak == 0 and exang == 0:
        return (0, 88.0, "Atypical chest pain in young patient — low cardiac probability")
    
    # RULE 5: Perfect exercise test (No heart disease can be ruled out with high confidence)
    if oldpeak == 0 and thalach > 170 and exang == 0 and age < 50:
        return (0, 90.0, "Normal exercise stress test — excellent functional capacity")
    
    # RULE 6: Severe hypertension in young patients (refer but not necessarily heart disease)
    if age < 40 and trestbps > 160 and ml_prediction == 1 and ml_confidence < 75:
        return (0, 65.0, "Elevated BP in young patient — monitor but low CAD probability")
    
    # Default: return ML prediction
    return (ml_prediction, ml_confidence, "Based on ML model prediction")

# STEP 4 – CLINICAL VALIDATION
def validate_clinical_plausibility(values: list[float]) -> tuple[bool, list[str]]:
    """Check if input combination is clinically plausible."""
    
    warnings_list = []
    
    age = values[0]
    ca = values[11]
    thal = values[12]
    oldpeak = values[9]
    thalach = values[7]
    chol = values[4]
    trestbps = values[3]
    
    # Check for valid thalassemia values
    if thal not in [3, 6, 7]:
        warnings_list.append(f"⚠ Thalassemia value {thal} is invalid. Use 3=Normal, 6=Fixed, 7=Reversible")
        return False, warnings_list
    
    # Clinical warnings (not blocking)
    if age < 35 and ca > 2:
        warnings_list.append(f"⚠ Age {age} with {ca} blocked vessels is extremely rare - verify")
    
    if age < 35 and oldpeak > 2.0:
        warnings_list.append(f"⚠ High ST depression ({oldpeak}) at age {age} is concerning")
    
    if trestbps > 180:
        warnings_list.append(f"⚠ Severe hypertension ({trestbps} mm/Hg) - verify reading")
    
    if chol > 400:
        warnings_list.append(f"⚠ Extremely high cholesterol ({chol}) - verify reading")
    
    if age < 40 and thalach > 190 and oldpeak == 0:
        warnings_list.append(f"ℹ High performance athlete profile detected")
    
    return True, warnings_list

def get_validated_input(prompt: str, cast_type: type,
                        low: float, high: float) -> float | None:
    """Prompt user, validate type and range. Returns None on 'exit'."""
    while True:
        raw = input(f"   {prompt}: ").strip()
        if raw.lower() in ("exit", "quit", "q"):
            return None
        try:
            val = cast_type(raw)
        except ValueError:
            print(f"   ⚠  Invalid input. Enter a numeric value.")
            continue
        if not (low <= val <= high):
            print(f"   ⚠  Out of valid range [{low} – {high}]. Try again.")
            continue
        return float(val)

# STEP 5 – CHATBOT PREDICTOR WITH CLINICAL RULES
def run_chatbot(model: RandomForestClassifier,
                scaler: StandardScaler) -> None:
    """Interactive clinical chatbot with clinical rule overrides."""
    
    print("\n" + "=" * 70)
    print("      AI CLINICAL ASSISTANT — RANDOM FOREST + RULES")
    print("=" * 70)
    print("""
  DISCLAIMER:
  This tool is for educational demonstration only.
  It is NOT a substitute for professional medical diagnosis.
  Always consult a qualified cardiologist for medical decisions.

  CLINICAL RULES ACTIVE:
  • Young athlete override (age < 35, normal stress test)
  • Exercise stress test interpretation
  • Atypical chest pain in young patients
""")
    print("  Enter patient data below. Type 'exit' at any prompt to quit.\n")
    print("  📌 IMPORTANT: Thalassemia uses Cleveland encoding:")
    print("     3 = Normal    6 = Fixed Defect    7 = Reversible Defect\n")
    
    session = 0
    while True:
        session += 1
        print(f"{'─'*70}")
        print(f"  Patient Assessment  #{session}")
        print(f"{'─'*70}")
        
        values: list[float] = []
        cancelled = False
        
        for idx, (prompt, cast_type, low, high) in enumerate(FEATURE_PROMPTS):
            label = f"{idx + 1:>2}. {prompt}"
            val = get_validated_input(label, cast_type, low, high)
            if val is None:
                cancelled = True
                break
            values.append(val)
        
        if cancelled:
            print("\n  Session ended. Goodbye!\n")
            break
        
        # Clinical validation
        is_plausible, warnings = validate_clinical_plausibility(values)
        
        if warnings:
            print(f"\n  {'!'*66}")
            for warning in warnings:
                print(f"  {warning}")
            if not is_plausible:
                print(f"\n  ✗ Invalid inputs detected. Please re-enter with correct values.")
                continue
            print(f"  {'!'*66}\n")
        
        # ML Prediction
        raw_arr: np.ndarray    = np.array([values])
        scaled_arr: np.ndarray = scaler.transform(raw_arr)
        ml_prediction: int     = int(model.predict(scaled_arr)[0])
        confidence: np.ndarray = model.predict_proba(scaled_arr)[0]
        ml_confidence = float(confidence[1]) * 100 if ml_prediction == 1 else float(confidence[0]) * 100
        
        # Apply clinical rules
        final_prediction, final_confidence, rule_reason = apply_clinical_rules(
            values, ml_prediction, ml_confidence
        )
        
        # Risk level mapping
        if final_prediction == 1:
            if final_confidence >= 70:
                risk_level = "HIGH RISK"
                risk_color = "🔴"
            elif final_confidence >= 40:
                risk_level = "MODERATE RISK"
                risk_color = "🟡"
            else:
                risk_level = "LOW RISK"
                risk_color = "🟢"
        else:
            risk_level = "LOW RISK"
            risk_color = "🟢"
        
        print(f"\n{'='*60}")
        print(f"       DIAGNOSTIC REPORT  —  Patient #{session}")
        print(f"{'='*60}")
        
        if final_prediction == 1:
            print(f"  {risk_color}  FINDING  : HEART DISEASE DETECTED")
            print(f"       Risk Level   : {risk_level}")
            print(f"       Confidence   : {final_confidence:.1f}%")
            print(f"\n  CLINICAL REASONING:")
            print(f"  • {rule_reason}")
            print(f"\n  RECOMMENDATION:")
            print(f"  • Please refer patient to a cardiologist immediately.")
            print(f"  • Additional diagnostic tests strongly advised.")
        else:
            print(f"  {risk_color}  FINDING  : NO HEART DISEASE DETECTED")
            print(f"       Risk Level   : {risk_level}")
            print(f"       Confidence   : {final_confidence:.1f}%")
            print(f"\n  CLINICAL REASONING:")
            print(f"  • {rule_reason}")
            if rule_reason != "Based on ML model prediction":
                print(f"  • ML model predicted: {'DISEASE' if ml_prediction else 'NO DISEASE'} ({ml_confidence:.1f}%)")
                print(f"  • Clinical rules overrode ML prediction based on patient demographics")
            print(f"\n  RECOMMENDATION:")
            print(f"  • Patient shows no immediate signs of heart disease.")
            print(f"  • Routine follow-up advised.")
        
        # Show what triggered the rule (FIXED TYPO HERE)
        if final_prediction != ml_prediction:
            print(f"\n  🔄 CLINICAL OVERRIDE APPLIED:")
            print(f"     ML said: {'DISEASE' if ml_prediction else 'NO DISEASE'}")
            print(f"     Clinical rules determined: {'NO DISEASE' if final_prediction == 0 else 'DISEASE'}")
        
        print(f"{'='*60}")
        print(f"\n  ⚠  DISCLAIMER: AI prediction — not a medical diagnosis.\n")
        
        again = input("  Run another assessment? (yes / no): ").strip().lower()
        if again not in ("yes", "y"):
            print("\n  Clinical session closed. Goodbye!\n")
            break
# MAIN
def main() -> None:
    print_banner()
    
    LOCAL_PATH = r"C:\Users\ML\Downloads\heart.csv"
    df = load_dataset(LOCAL_PATH)
    
    results = train_model(df)
    
    run_chatbot(results["model"], results["scaler"])

if __name__ == "__main__":
    main()