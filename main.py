
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load the official training and testing datasets
train_data = pd.read_csv("data/UNSW_NB15_training-set.csv")
test_data = pd.read_csv("data/UNSW_NB15_testing-set.csv")

print("Datasets loaded successfully!")
print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))

# 2. Separate features and target labels
# Drop ID and attack_cat to avoid identifier and target leakage
X_train = train_data.drop(
    columns=["label", "attack_cat", "id"]
)
y_train = train_data["label"]

X_test = test_data.drop(
    columns=["label", "attack_cat", "id"]
)
y_test = test_data["label"]

# 3. Identify categorical and numerical columns
categorical_columns = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCategorical features:", categorical_columns)
print("Number of numerical features:", len(numerical_columns))

# 4. Replace infinite values with missing values
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# 5. Prepare numerical and categorical data
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# 6. Define the models
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=20,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
}

results = []

# 7. Train and evaluate both models
for name, classifier in models.items():

    print("\n" + "=" * 50)
    print("Training:", name)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Attack"],
        zero_division=0
    ))

    # 8. Save the confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Normal", "Attack"],
        cmap="Blues"
    )

    plt.title("Confusion Matrix - " + name)
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(filename + "_confusion_matrix.png")
    plt.close()

# 9. Compare model performance
results_df = pd.DataFrame(results)

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

# 10. Plot performance comparison
results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1-score"]
].plot(kind="bar", figsize=(10, 6), ylim=(0, 1))

plt.title("UNSW-NB15 Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

print("\nProject completed successfully!")
print("Results saved in model_comparison.csv")
print("Comparison chart saved in model_comparison.png")
