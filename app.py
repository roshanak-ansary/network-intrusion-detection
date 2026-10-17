
import streamlit as st
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
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="Sentinel | Network Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- DESIGN ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #08111f 0%, #0d192b 100%);
    color: #e7efff;
}
[data-testid="stSidebar"] {
    background: #0a1423;
    border-right: 1px solid #20334c;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}
h1, h2, h3 {
    color: #f1f6ff !important;
}
p, label, .stMarkdown {
    color: #c3d1e6;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #13243a, #102033);
    border: 1px solid #29415c;
    padding: 20px 22px;
    border-radius: 15px;
    box-shadow: 0 5px 20px #00000018;
}
[data-testid="stMetricLabel"] {
    color: #9eb5d1 !important;
}
[data-testid="stMetricValue"] {
    color: #45e0cb !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #29415c;
    border-radius: 12px;
    overflow: hidden;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid #29415c;
}
.stTabs [data-baseweb="tab"] {
    color: #a9bed8;
    padding: 12px 18px;
}
.stTabs [aria-selected="true"] {
    color: #45e0cb !important;
}
div.stButton > button {
    background: #16b8a6;
    color: #071522;
    border: none;
    border-radius: 9px;
    font-weight: 700;
}
.hero {
    background: linear-gradient(110deg, #122943, #102035);
    border: 1px solid #294761;
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 25px;
}
.eyebrow {
    color: #45e0cb;
    letter-spacing: 2px;
    font-size: 12px;
    font-weight: bold;
}
.hero-title {
    color: #f4f8ff;
    font-size: 31px;
    font-weight: 750;
    margin: 8px 0;
}
.hero-subtitle {
    color: #a9bfd9;
    font-size: 15px;
}
.section-label {
    color: #45e0cb;
    font-size: 12px;
    letter-spacing: 1.5px;
    font-weight: bold;
    margin-bottom: 8px;
}
.info-card {
    background: #102035;
    border: 1px solid #29415c;
    border-radius: 14px;
    padding: 18px;
    height: 100%;
}
.small-note {
    color: #8ea7c5;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------- MODEL TRAINING ----------
@st.cache_resource(show_spinner="Training models on UNSW-NB15...")
def train_models():
    train = pd.read_csv("data/UNSW_NB15_training-set.csv")
    test = pd.read_csv("data/UNSW_NB15_testing-set.csv")

    X_train = train.drop(columns=["label", "attack_cat", "id"])
    y_train = train["label"]

    X_test = test.drop(columns=["label", "attack_cat", "id"])
    y_test = test["label"]

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    cat_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    num_cols = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    predictions = {}

    for name, classifier in models.items():
        preprocessor = ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ])

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "F1-score": f1_score(
                y_test, pred, zero_division=0
            )
        })

    return (
        pd.DataFrame(results),
        y_test,
        predictions,
        len(train),
        len(test)
    )


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 🛡️ SENTINEL")
    st.caption("NETWORK INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Overview", "Model Analysis", "Confusion Matrices", "About Project"]
    )

    st.divider()
    st.markdown("**SYSTEM STATUS**")
    st.markdown("🟢 Dataset: UNSW-NB15")
    st.markdown("🟢 ML pipeline: Ready")
    st.markdown("🟢 Classification: Binary")
    st.caption("Academic Machine Learning Project")


# ---------- RUN ----------
st.markdown("""
<div class="hero">
    <div class="eyebrow">CYBERSECURITY / MACHINE LEARNING</div>
    <div class="hero-title">Network Intrusion Detection</div>
    <div class="hero-subtitle">
        Intelligent traffic classification · UNSW-NB15 benchmark dataset
    </div>
</div>
""", unsafe_allow_html=True)

try:
    results, y_test, predictions, train_size, test_size = train_models()

    if page == "Overview":
        st.markdown('<div class="section-label">01 / OVERVIEW</div>',
                    unsafe_allow_html=True)
        st.header("Security Analytics Dashboard")
        st.write("A comparative evaluation of two machine learning classifiers.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Training Records", f"{train_size:,}")
        c2.metric("Testing Records", f"{test_size:,}")
        c3.metric("Models Evaluated", "02")

        st.write("")
        st.subheader("Performance at a Glance")

        chart = results.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1-score"]
        ]
        st.bar_chart(chart, y_label="Score", height=350)

        st.subheader("Evaluation Summary")
        display = results.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
            display[col] = display[col].map(lambda x: f"{x:.2%}")

        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="info-card">
            <b>How to interpret the results</b><br><br>
            Recall measures how many actual attacks are detected.
            Precision measures how reliable the attack alerts are.
            F1-score balances Precision and Recall.
        </div>
        """, unsafe_allow_html=True)

    elif page == "Model Analysis":
        st.markdown('<div class="section-label">02 / MODEL ANALYSIS</div>',
                    unsafe_allow_html=True)
        st.header("Classifier Performance")

        model_name = st.selectbox(
            "Select classifier",
            results["Model"].tolist()
        )
        row = results[results["Model"] == model_name].iloc[0]

        cols = st.columns(4)
        metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

        for col, metric in zip(cols, metrics):
            col.metric(metric, f"{row[metric]:.2%}")

        st.subheader("Metric Comparison")
        st.bar_chart(
            results.set_index("Model")[metrics],
            y_label="Score",
            height=350
        )

        st.subheader("Classification Report")
        report = classification_report(
            y_test,
            predictions[model_name],
            labels=[0, 1],
            target_names=["Normal", "Attack"],
            zero_division=0,
            output_dict=True
        )
        st.dataframe(
            pd.DataFrame(report).transpose().style.format("{:.3f}"),
            use_container_width=True
        )

        st.download_button(
            "Download evaluation results (CSV)",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="model_comparison.csv",
            mime="text/csv"
        )

    elif page == "Confusion Matrices":
        st.markdown('<div class="section-label">03 / ERROR ANALYSIS</div>',
                    unsafe_allow_html=True)
        st.header("Confusion Matrix")
        st.write("Actual labels versus predictions on the held-out test set.")

        model_name = st.selectbox(
            "Select classifier",
            results["Model"].tolist()
        )

        cm = confusion_matrix(
            y_test, predictions[model_name], labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(7, 4.5))
        fig.patch.set_facecolor("#0d192b")
        ax.set_facecolor("#13243a")

        image = ax.imshow(cm, cmap="GnBu")
        ax.set_title(model_name, color="white", pad=16)
        ax.set_xlabel("Predicted label", color="white")
        ax.set_ylabel("Actual label", color="white")
        ax.set_xticks([0, 1], ["Normal", "Attack"])
        ax.set_yticks([0, 1], ["Normal", "Attack"])
        ax.tick_params(colors="white")

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white", fontsize=15, fontweight="bold"
                )

        for spine in ax.spines.values():
            spine.set_color("#29415c")

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        tn, fp, fn, tp = cm.ravel()
        a, b, c, d = st.columns(4)
        a.metric("True Normal", f"{tn:,}")
        b.metric("False Alarm", f"{fp:,}")
        c.metric("Missed Attacks", f"{fn:,}")
        d.metric("Detected Attacks", f"{tp:,}")

    else:
        st.markdown('<div class="section-label">04 / PROJECT DETAILS</div>',
                    unsafe_allow_html=True)
        st.header("About This Project")

        st.markdown("""
        <div class="info-card">
            <h3>Objective</h3>
            Evaluate machine learning models for identifying malicious
            network traffic.<br><br>
            <h3>Dataset</h3>
            UNSW-NB15, using the supplied training and testing CSV files.<br><br>
            <h3>Algorithms</h3>
            Decision Tree and Random Forest.<br><br>
            <h3>Evaluation</h3>
            Accuracy, Precision, Recall, F1-score and Confusion Matrix.<br><br>
            <h3>Target</h3>
            Binary classification: Normal traffic or Attack.
        </div>
        """, unsafe_allow_html=True)

        st.caption(
            "This is an academic evaluation on a benchmark dataset, "
            "not a live network monitoring system."
        )

    st.divider()
    st.caption(
        "SENTINEL · Network Intrusion Detection · UNSW-NB15"
    )

except FileNotFoundError:
    st.error(
        "Dataset files were not found. Check that both CSV files "
        "are inside the data folder."
    )
except Exception as error:
    st.error(f"Error: {error}")
