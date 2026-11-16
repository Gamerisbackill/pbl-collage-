
# SmartMach - Machine Learning based Predictive Maintenance
# Course: Machine Learning Lab (24M15CA114)
# Dataset: AI4I 2020 Predictive Maintenance (ai4i2020.csv)


# 1. IMPORT LIBRARIES

import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Set this to False if you do not want to type machine values at the end.
RUN_INTERACTIVE = True


# 2. LOAD DATASET (CSV FILE)
# Option A (VS Code / Jupyter): keep ai4i2020.csv in the same folder as this file.
# Option B (Google Colab): run this cell, a "Choose Files" button appears,
#                          select ai4i2020.csv from your computer.
DATA_FILE = "ai4i2020.csv"

if os.path.exists(DATA_FILE):
    # File found in the current folder (or already uploaded to Colab)
    df = pd.read_csv(DATA_FILE)
else:
    try:
        # Google Colab: ask the user to upload the CSV
        from google.colab import files
        print("Please upload ai4i2020.csv")
        uploaded = files.upload()
        df = pd.read_csv(io.BytesIO(next(iter(uploaded.values()))))
    except ImportError:
        # Not Colab and file missing: stop with a clear message
        raise FileNotFoundError(
            "ai4i2020.csv not found. Download it from the UCI repository "
            "(AI4I 2020 Predictive Maintenance Dataset) and place it next to this file.")

print("Dataset shape:", df.shape)
print(df.head())


# 3. DATA PREPROCESSING

# Rename columns to make them easier to use.
df.rename(columns={
    "Air temperature [K]": "Air_temp",
    "Process temperature [K]": "Process_temp",
    "Rotational speed [rpm]": "Rot_speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool_wear",
    "Machine failure": "Machine_failure"
}, inplace=True)

# Remove missing rows and duplicate sensor records.
df.dropna(inplace=True)
df.drop_duplicates(
    subset=["Type", "Air_temp", "Process_temp", "Rot_speed",
            "Torque", "Tool_wear", "Machine_failure"],
    inplace=True
)

# Remove ID columns and failure-reason columns.
# Failure-reason columns are removed to avoid giving the answer to the model.
columns_to_remove = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
df.drop(columns=columns_to_remove, inplace=True, errors="ignore")

# Convert machine type into numbers.
df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})


# 4. FEATURE ENGINEERING

# Create two additional useful features.
df["Temp_diff"] = df["Process_temp"] - df["Air_temp"]
df["Power"] = df["Torque"] * df["Rot_speed"] * 2 * np.pi / 60


# 5. EXPLORATORY DATA ANALYSIS (EDA)

print("\nFailure count:")
print(df["Machine_failure"].value_counts())

# Graph 1: Number of machines with and without failure.
df["Machine_failure"].value_counts().plot(kind="bar")
plt.title("Machine Failure Count")
plt.xlabel("0 = No Failure, 1 = Failure")
plt.ylabel("Number of Machines")
plt.tight_layout()
plt.show()

# Graph 2: Correlation between numerical columns.
corr = df.corr(numeric_only=True)
plt.figure(figsize=(9, 6))
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.title("Correlation Heatmap")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.show()


# 6. SELECT INPUTS AND OUTPUT

features = [
    "Type", "Air_temp", "Process_temp", "Rot_speed",
    "Torque", "Tool_wear", "Temp_diff", "Power"
]

X = df[features]                 # Input sensor values
y = df["Machine_failure"]        # Output: 0 or 1

# Split data: 80% for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 7. CREATE AND TRAIN MACHINE LEARNING MODELS

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=1000
    ),
    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced", max_depth=6, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        class_weight="balanced", n_estimators=100, random_state=42
    ),
    "SVM": SVC(
        class_weight="balanced", probability=True, random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

trained_models = {}

for model_name, model in models.items():
    # Scaling makes numerical features comparable for the algorithms.
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    trained_models[model_name] = pipeline
    print(model_name, "training completed.")


# 8. TEST AND COMPARE MODELS
results = []

for model_name, model in trained_models.items():
    predictions = model.predict(X_test)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0)
    })

    print("\n", model_name)
    print(classification_report(
        y_test, predictions,
        target_names=["No Failure", "Failure"],
        zero_division=0
    ))

    # Confusion matrix of this model
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        display_labels=["No Failure", "Failure"]
    )
    plt.title(model_name)
    plt.tight_layout()
    plt.show()

results_df = pd.DataFrame(results).set_index("Model")
print("\nMODEL COMPARISON")
print(results_df.round(3))

results_df.plot(kind="bar", figsize=(10, 5))
plt.title("Comparison of Machine Learning Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Select the model with the highest F1 score.
# F1 is useful because machine failures are less common than normal cases.
best_model_name = results_df["F1 Score"].idxmax()
best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)
print(results_df.loc[best_model_name])


# 9. IMPORTANT PARAMETERS INFLUENCING FAILURE

# Random Forest can tell which inputs matter most for predicting failure.
rf_model = trained_models["Random Forest"].steps[-1][1]
importance = pd.Series(rf_model.feature_importances_, index=features).sort_values()

print("\nFeature importance (Random Forest):")
print(importance.sort_values(ascending=False).round(3))

importance.plot(kind="barh", figsize=(8, 5))
plt.title("Important Parameters Affecting Machine Failure")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# 10. PREDICT FAILURE AND RISK LEVEL FOR A NEW MACHINE

def predict_machine(machine_type, air_temp, process_temp,
                    rot_speed, torque, tool_wear):

    # Convert machine type into the same numbers used during training.
    type_values = {"L": 0, "M": 1, "H": 2}
    machine_type = str(machine_type).strip().upper()

    if machine_type not in type_values:
        print("Machine type must be L, M, or H.")
        return

    # Check that every sensor value is a valid positive number.
    try:
        air_temp, process_temp, rot_speed, torque, tool_wear = [
            float(v) for v in (air_temp, process_temp, rot_speed, torque, tool_wear)]
    except ValueError:
        print("All sensor values must be numbers.")
        return

    if min(air_temp, process_temp, torque, tool_wear) < 0 or rot_speed <= 0:
        print("Values cannot be negative and rotational speed must be above 0.")
        return

    # Calculate the two additional features.
    temp_diff = process_temp - air_temp
    power = torque * rot_speed * 2 * np.pi / 60

    # Prepare input in the same feature order as the training data.
    new_machine = pd.DataFrame([[
        type_values[machine_type], air_temp, process_temp,
        rot_speed, torque, tool_wear, temp_diff, power
    ]], columns=features)

    # Predict failure probability.
    probability = best_model.predict_proba(new_machine)[0][1]
    prediction = best_model.predict(new_machine)[0]

    # Convert probability into a risk category.
    # These thresholds are project assumptions.
    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    print("\nMACHINE PREDICTION")
    if prediction == 1:
        print("Result: Machine failure predicted")
    else:
        print("Result: No machine failure predicted")

    print("Failure probability:", round(probability * 100, 2), "%")
    print("Risk level:", risk)



# 11. TRY A SAMPLE MACHINE
predict_machine(
    machine_type="M",
    air_temp=298.1,
    process_temp=308.6,
    rot_speed=1551,
    torque=42.8,
    tool_wear=100
)


# 12. INTERACTIVE PREDICTION (USER TYPES MACHINE VALUES)

if RUN_INTERACTIVE:
    print("\n===== INTERACTIVE PREDICTION =====")
    while True:
        choice = input("\nPredict for a machine? (y/n): ").strip().lower()
        if choice != "y":
            print("Program finished.")
            break

        # Read values typed by the user; predict_machine() checks them.
        predict_machine(
            input("Machine type (L/M/H): "),
            input("Air temperature [K] (e.g. 300): "),
            input("Process temperature [K] (e.g. 310): "),
            input("Rotational speed [rpm] (e.g. 1500): "),
            input("Torque [Nm] (e.g. 40): "),
            input("Tool wear [min] (e.g. 100): ")
        )
