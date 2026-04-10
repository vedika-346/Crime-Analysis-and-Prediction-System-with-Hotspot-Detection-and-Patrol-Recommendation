import os, pickle, warnings, time
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.ensemble           import RandomForestClassifier, IsolationForest
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.naive_bayes        import GaussianNB
from sklearn.svm                import LinearSVC
from sklearn.calibration        import CalibratedClassifierCV
from sklearn.cluster            import DBSCAN, KMeans
from sklearn.metrics            import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
import xgboost as xgb

# ── Optional: SMOTE for class balancing ──────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
    print("SMOTE available — will balance training set")
except ImportError:
    HAS_SMOTE = False
    print("SMOTE not installed (pip install imbalanced-learn) — skipping balancing")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
D = lambda f: os.path.join(BASE_DIR, "data", f)
M = lambda f: os.path.join(BASE_DIR, f)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading cleaned data...")
data = pd.read_csv(D("cleaned_crime.csv"))
print(f"  {len(data):,} rows  |  {data['Crime_Type'].nunique()} crime types")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SMARTER CLASS CONSOLIDATION
# Merge classes with < 500 samples into semantically close buckets.
# Fewer, cleaner classes = dramatically higher accuracy.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Class consolidation ──")

# Map rare / overlapping types into broader buckets
REMAP = {
    # Violence — each type kept separate
    'HOMICIDE':                  'HOMICIDE',
    'ASSAULT':                   'ASSAULT',
    'BATTERY':                   'BATTERY',
    'ROBBERY':                   'ROBBERY',
    'KIDNAPPING':                'KIDNAPPING',
    'HUMAN TRAFFICKING':         'HUMAN TRAFFICKING',
    # Property
    'BURGLARY':                  'PROPERTY',
    'MOTOR VEHICLE THEFT':       'PROPERTY',
    'ARSON':                     'PROPERTY',
    'CRIMINAL DAMAGE':           'PROPERTY',
    # Theft (keep separate — very common, distinct pattern)
    'THEFT':                     'THEFT',
    # Narcotics
    'NARCOTICS':                 'NARCOTICS',
    'OTHER NARCOTIC VIOLATION':  'NARCOTICS',
    # Sexual offences
    'CRIM SEXUAL ASSAULT':       'SEX OFFENSE',
    'SEX OFFENSE':               'SEX OFFENSE',
    'PROSTITUTION':              'SEX OFFENSE',
    'STALKING':                  'SEX OFFENSE',
    # Weapons
    'WEAPONS VIOLATION':         'WEAPONS',
    # Fraud / financial
    'DECEPTIVE PRACTICE':        'FRAUD',
    'FORGERY':                   'FRAUD',
    'GAMBLING':                  'FRAUD',
    'EXTORTION':                 'FRAUD',
    # Disorder
    'CRIMINAL TRESPASS':         'DISORDER',
    'LIQUOR LAW VIOLATION':      'DISORDER',
    'DISORDERLY CONDUCT':        'DISORDER',
    'PUBLIC PEACE VIOLATION':    'DISORDER',
    'INTIMIDATION':              'DISORDER',
    'OBSCENITY':                 'DISORDER',
    'PUBLIC INDECENCY':          'DISORDER',
    # Other / rare → catch-all
    'OTHER OFFENSE':             'OTHER',
    'NON-CRIMINAL':              'OTHER',
    'NON - CRIMINAL':            'OTHER',
    'CONCEALED CARRY LICENSE VIOLATION': 'OTHER',
    'RITUALISM':                 'OTHER',
}

data['Crime_Type'] = data['Crime_Type'].replace(REMAP)

# Drop anything still rare after remapping (< 200 samples)
counts = data['Crime_Type'].value_counts()
keep   = counts[counts >= 200].index
data   = data[data['Crime_Type'].isin(keep)]
print(f"  Classes after consolidation: {data['Crime_Type'].nunique()}")
print(f"  Class distribution:\n{data['Crime_Type'].value_counts().to_string()}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FEATURE ENGINEERING (richer than before)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Feature engineering ──")

data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# ── Time features ────────────────────────────────────────────────────────────
data['IsNight']     = data['Hour'].between(20, 23) | data['Hour'].between(0, 5)
data['IsNight']     = data['IsNight'].astype(int)
data['IsRushHour']  = data['Hour'].isin([7,8,9,16,17,18]).astype(int)
data['Season']      = data['Month'].map({
    12:0,1:0,2:0,   # Winter
    3:1,4:1,5:1,    # Spring
    6:2,7:2,8:2,    # Summer
    9:3,10:3,11:3   # Autumn
})

# ── Crime_Description encoding (biggest accuracy boost) ──────────────────────
if 'Crime_Description' in data.columns:
    # Keep top-80 descriptions, collapse rest to 'OTHER_DESC'
    top_desc = data['Crime_Description'].value_counts().head(80).index
    data['Crime_Description'] = data['Crime_Description'].where(
        data['Crime_Description'].isin(top_desc), 'OTHER_DESC'
    )
    desc_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    data['Desc_Encoded'] = desc_enc.fit_transform(
        data[['Crime_Description']]
    ).astype(int)
    with open(M("desc_encoder.pkl"), "wb") as f:
        pickle.dump(desc_enc, f)
    HAS_DESC = True
    print("  Crime_Description encoded — top-80 descriptions kept")
else:
    HAS_DESC = False
    print("  Crime_Description column not found — skipping")

# ── Ward / Block encoding ─────────────────────────────────────────────────────
if 'Ward' in data.columns:
    data['Ward'] = pd.to_numeric(data['Ward'], errors='coerce').fillna(0).astype(int)
    HAS_WARD = True
else:
    HAS_WARD = False

# ── Arrest / Domestic → already int from eda.py ──────────────────────────────
for col in ['Arrest', 'Domestic']:
    if col in data.columns:
        data[col] = data[col].astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ENCODE TARGET & BUILD FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Data preprocessing ──")

le = LabelEncoder()
data['Crime_Type_Encoded'] = le.fit_transform(data['Crime_Type'])
n_classes = len(le.classes_)
print(f"  Final class count: {n_classes} → {list(le.classes_)}")

FEATURES = [
    'Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'IsWeekend',
    'IsNight', 'IsRushHour', 'Season',
    'Hour_Sin', 'Hour_Cos', 'Month_Sin', 'Month_Cos',
    'Latitude', 'Longitude', 'Community_Area',
]
if HAS_DESC:  FEATURES.append('Desc_Encoded')
if HAS_WARD:  FEATURES.append('Ward')
for col in ['Arrest', 'Domestic']:
    if col in data.columns:
        FEATURES.append(col)

# Keep only columns that exist
FEATURES = [f for f in FEATURES if f in data.columns]
print(f"  Features used ({len(FEATURES)}): {FEATURES}")

X = data[FEATURES].copy()
y = data['Crime_Type_Encoded'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── SMOTE — balance minority classes in training set ──────────────────────────
if HAS_SMOTE:
    print("  Applying SMOTE...")
    min_samples = y_train.value_counts().min()
    k = min(5, min_samples - 1) if min_samples > 1 else 1
    sm = SMOTE(random_state=42, k_neighbors=k)
    try:
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"  After SMOTE — Train: {len(X_train):,}")
    except Exception as e:
        print(f"  SMOTE failed ({e}) — using original training set")

# ── Scaler for distance-based models ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
with open(M("scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("  StandardScaler fitted and saved")

# ══════════════════════════════════════════════════════════════════════════════
# HELPER — evaluate
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(name, y_true, y_pred, model=None, X_ev=None, needs_proba=True):
    acc  = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_m = f1_score(y_true, y_pred, average='macro',    zero_division=0)
    auc  = None
    if needs_proba and model is not None and X_ev is not None:
        try:
            proba = model.predict_proba(X_ev)
            auc   = roc_auc_score(
                y_true, proba, multi_class='ovr',
                average='weighted', labels=list(range(n_classes))
            )
        except Exception:
            auc = None
    report = classification_report(y_true, y_pred,
                                   target_names=le.classes_, zero_division=0)
    cm_arr = confusion_matrix(y_true, y_pred)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"  Accuracy    : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Weighted F1 : {f1_w:.4f}")
    print(f"  Macro F1    : {f1_m:.4f}")
    if auc: print(f"  ROC-AUC     : {auc:.4f}")
    print(f"{'='*55}")
    return dict(model=name, accuracy=acc, f1_weighted=f1_w,
                f1_macro=f1_m, roc_auc=auc, report=report, cm=cm_arr.tolist())

# ══════════════════════════════════════════════════════════════════════════════
# 1. XGBoost — primary, heavily tuned
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 1. XGBoost (tuned) ──")
t0 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.5,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    use_label_encoder=False,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_preds = xgb_model.predict(X_test)
xgb_eval  = evaluate("XGBoost (tuned)", y_test, xgb_preds, xgb_model, X_test)
print(f"  Time: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Random Forest — deeper, more trees
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 2. Random Forest (tuned) ──")
t0 = time.time()
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_eval  = evaluate("Random Forest (tuned)", y_test, rf_preds, rf_model, X_test)
print(f"  Time: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# 3. KNN
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 3. KNN (k=15) ──")
t0 = time.time()
knn_model = KNeighborsClassifier(n_neighbors=15, n_jobs=-1, metric='euclidean')
knn_model.fit(X_train_scaled, y_train)
knn_preds = knn_model.predict(X_test_scaled)
knn_eval  = evaluate("KNN (k=15)", y_test, knn_preds, knn_model, X_test_scaled)
print(f"  Time: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Naive Bayes
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 4. Naive Bayes ──")
t0 = time.time()
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
nb_preds = nb_model.predict(X_test_scaled)
nb_eval  = evaluate("Naive Bayes", y_test, nb_preds, nb_model, X_test_scaled)
print(f"  Time: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SVM
# ══════════════════════════════════════════════════════════════════════════════
print("\n── 5. SVM (LinearSVC) ──")
t0 = time.time()
svm_base  = LinearSVC(max_iter=3000, random_state=42, C=1.0)
svm_model = CalibratedClassifierCV(svm_base, cv=3)
svm_model.fit(X_train_scaled, y_train)
svm_preds = svm_model.predict(X_test_scaled)
svm_eval  = evaluate("SVM (LinearSVC)", y_test, svm_preds, svm_model, X_test_scaled)
print(f"  Time: {time.time()-t0:.1f}s")

# ══════════════════════════════════════════════════════════════════════════════
# Save all models + artefacts
# ══════════════════════════════════════════════════════════════════════════════
models_to_save = {
    "model.pkl":     xgb_model,
    "rf_model.pkl":  rf_model,
    "knn_model.pkl": knn_model,
    "nb_model.pkl":  nb_model,
    "svm_model.pkl": svm_model,
    "encoder.pkl":   le,
    "features.pkl":  FEATURES,
}
for fname, obj in models_to_save.items():
    with open(M(fname), "wb") as f:
        pickle.dump(obj, f)
print("\nAll models saved.")

# ══════════════════════════════════════════════════════════════════════════════
# Metrics CSVs
# ══════════════════════════════════════════════════════════════════════════════
all_evals = [xgb_eval, rf_eval, knn_eval, nb_eval, svm_eval]

metrics_df = pd.DataFrame([{
    'Model':       e['model'],
    'Accuracy':    e['accuracy'],
    'F1_Weighted': e['f1_weighted'],
    'F1_Macro':    e['f1_macro'],
    'ROC_AUC':     e['roc_auc']
} for e in all_evals])
metrics_df.to_csv(D("model_metrics.csv"), index=False)

# Per-class report from best model (XGBoost)
report_rows = []
for line in xgb_eval['report'].strip().split('\n')[2:-4]:
    parts = line.split()
    if len(parts) >= 5:
        report_rows.append({
            'Crime_Type': ' '.join(parts[:-4]),
            'Precision':  float(parts[-4]),
            'Recall':     float(parts[-3]),
            'F1':         float(parts[-2]),
            'Support':    int(parts[-1])
        })
pd.DataFrame(report_rows).to_csv(D("per_class_metrics.csv"), index=False)

# Confusion matrix (all consolidated classes)
cm_full  = np.array(xgb_eval['cm'])
cm_df    = pd.DataFrame(cm_full, index=le.classes_, columns=le.classes_)
cm_df.to_csv(D("confusion_matrix.csv"))

# ── SHAP ──────────────────────────────────────────────────────────────────────
print("\nComputing SHAP values...")
try:
    import shap
    sample_X  = X_test.sample(min(2000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(sample_X)
    if isinstance(shap_vals, list):
        mean_shap = np.mean([np.abs(s).mean(axis=0) for s in shap_vals], axis=0)
    else:
        mean_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({'Feature': FEATURES, 'Mean_SHAP': mean_shap})
    shap_df.sort_values('Mean_SHAP', ascending=False).to_csv(
        D("shap_importance.csv"), index=False)
    print("  SHAP saved")
except Exception as e:
    print(f"  SHAP skipped ({e}) — using XGB built-in importance")
    fi = xgb_model.feature_importances_
    pd.DataFrame({'Feature': FEATURES, 'Mean_SHAP': fi}).sort_values(
        'Mean_SHAP', ascending=False).to_csv(D("shap_importance.csv"), index=False)

# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

data_full = pd.read_csv(D("clustered_crime.csv")) if os.path.exists(D("clustered_crime.csv")) else data.copy()

print("\n── DBSCAN hotspot detection ──")
coords = data[['Latitude', 'Longitude']]
db = DBSCAN(eps=0.008, min_samples=20, n_jobs=-1)
data['Cluster'] = db.fit_predict(coords)
n_clusters = data['Cluster'].nunique() - (1 if -1 in data['Cluster'].values else 0)
print(f"  Clusters found: {n_clusters}")

print("── KMeans patrol zones (k=10) ──")
km = KMeans(n_clusters=10, random_state=42, n_init='auto')
data['Patrol_Zone'] = km.fit_predict(data[['Latitude', 'Longitude']])
km_centers = pd.DataFrame(km.cluster_centers_, columns=['Lat', 'Lon'])
km_centers.index.name = 'Zone'
km_centers.reset_index().to_csv(D("patrol_zone_centers.csv"), index=False)

print("── IsolationForest anomaly detection ──")
iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
data['Anomaly_Score'] = iso.fit_predict(
    data[['Hour', 'Latitude', 'Longitude', 'Community_Area', 'Month']]
)
data['Is_Anomaly'] = (data['Anomaly_Score'] == -1).astype(int)
print(f"  Anomalies: {data['Is_Anomaly'].sum():,}")

data.to_csv(D("clustered_crime.csv"), index=False)

# ── Area summary ──────────────────────────────────────────────────────────────
area_summary = data.groupby('Community_Area').agg(
    Total_Crimes      = ('Crime_Type',   'count'),
    Most_Common_Crime = ('Crime_Type',   lambda x: x.mode().iloc[0]),
    Peak_Hour         = ('Hour',         lambda x: x.mode().iloc[0]),
    Avg_Latitude      = ('Latitude',     'mean'),
    Avg_Longitude     = ('Longitude',    'mean'),
    Anomaly_Count     = ('Is_Anomaly',   'sum'),
    Dominant_Zone     = ('Patrol_Zone',  lambda x: x.mode().iloc[0])
).reset_index()

area_summary['Crime_Score']    = (area_summary['Total_Crimes'] /
                                   area_summary['Total_Crimes'].max())
area_summary['Hour_Weight']    = area_summary['Peak_Hour'].apply(
    lambda x: 1.5 if (x >= 20 or x <= 5) else 1.0)
area_summary['Anomaly_Weight'] = 1 + (
    area_summary['Anomaly_Count'] / area_summary['Anomaly_Count'].max()) * 0.5
area_summary['Risk_Score']     = (area_summary['Crime_Score'] *
                                   area_summary['Hour_Weight'] *
                                   area_summary['Anomaly_Weight'])
area_summary.sort_values('Risk_Score', ascending=False, inplace=True)
area_summary.to_csv(D("area_summary.csv"), index=False)

# ── Patrol recommendations ────────────────────────────────────────────────────
def patrol_strategy(row):
    h = int(row['Peak_Hour'])
    crime = row['Most_Common_Crime']
    shift = ("Night Patrol"   if (h >= 20 or h <= 5) else
             "Morning Patrol" if h <= 12 else "Evening Patrol")
    flag  = " [⚠ Anomalous activity]" if row['Anomaly_Count'] > 5 else ""
    return (f"{shift} — Community Area {int(row['Community_Area'])} "
            f"at {h:02d}:00 hrs, high {crime.lower()} incidents.{flag}")

patrol = area_summary.head(15).copy()
patrol['Recommendation'] = patrol.apply(patrol_strategy, axis=1)
patrol.to_csv(D("patrol_recommendations.csv"), index=False)

# ── Prophet forecast ──────────────────────────────────────────────────────────
print("\n── Prophet forecast ──")
try:
    from prophet import Prophet
    daily = (data.groupby(data['Date'].dt.date if 'Date' in data.columns
                          else pd.to_datetime(data.get('Date', pd.Series())).dt.date)
             .size().reset_index(name='y')
             .rename(columns={0: 'ds', 'Date': 'ds', 'index': 'ds'}))
    daily.columns = ['ds', 'y']
    daily['ds'] = pd.to_datetime(daily['ds'])
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False)
    m.fit(daily)
    future   = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
        D("forecast.csv"), index=False)
    print("  forecast.csv saved")
except Exception as e:
    print(f"  Prophet skipped ({e})")

print("\n✓ model.py complete")
print(f"  Best model accuracy: {max(e['accuracy'] for e in all_evals)*100:.1f}%")
print(f"  Saved: clustered_crime.csv, area_summary.csv, patrol_recommendations.csv")
print(f"  Saved: model_metrics.csv ({len(all_evals)} models compared)")
print(f"  Saved: per_class_metrics.csv, confusion_matrix.csv, shap_importance.csv")