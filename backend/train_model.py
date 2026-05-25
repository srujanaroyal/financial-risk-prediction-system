import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv('C:\\Users\\Srujana Royal\\OneDrive\\Desktop\\financial-risk-project\\dataset\\Financial Risk Classification Dataset.csv')

print("Dataset Loaded Successfully")
print(df.head())

# =========================
# CHECK DATASET
# =========================

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Distribution:")
print(df['loan_default'].value_counts())

# =========================
# MANUAL BALANCING
# =========================

majority = df[df['loan_default'] == 0]
minority = df[df['loan_default'] == 1]

print("\nBefore Balancing:")
print(df['loan_default'].value_counts())

# Upsample minority class

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=500,
    random_state=42
)

# Combine majority + minority

df_balanced = pd.concat([majority, minority_upsampled])

print("\nAfter Balancing:")
print(df_balanced['loan_default'].value_counts())

# =========================
# FEATURES AND TARGET
# =========================

X = df_balanced.drop('loan_default', axis=1)
y = df_balanced['loan_default']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# MODEL
# =========================

model = SVC(
    kernel='rbf',
    C=1,
    gamma='scale',
    probability=True,
    random_state=42
)

# =========================
# TRAIN MODEL
# =========================

print("\nTraining Model...")

model.fit(X_train_scaled, y_train)

print("Model Training Completed")

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test_scaled)

# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, 'svm_financial_risk_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'label_columns.pkl')

print("\nFiles Saved Successfully")

print("\nGenerated Files:")
print("1. svm_financial_risk_model.pkl")
print("2. scaler.pkl")
print("3. label_columns.pkl")