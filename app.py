
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(
    page_title="Fake Job Posting Classifier",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 Fake Job Posting Classifier")
st.write(
    "Analyze a job posting and identify whether it appears "
    "Normal or Potentially Suspicious using machine learning."
)

DATA_PATH = Path(__file__).parent / "dataset" / "DataSet.csv"

@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)

    text_columns = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits"
    ]

    for column in text_columns:
        if column not in df.columns:
            df[column] = ""

    df["job_text"] = (
        df[text_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    labels = df["fraudulent"].map({"f": 0, "t": 1})

    if labels.isna().any():
        labels = df["fraudulent"].map({0: 0, 1: 1, "0": 0, "1": 1})

    valid = labels.notna() & df["job_text"].str.strip().ne("")
    X = df.loc[valid, "job_text"]
    y = labels.loc[valid].astype(int)

    if y.nunique() != 2:
        raise ValueError("Dataset must contain both normal and suspicious examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)

    return vectorizer, model, X_test, y_test

try:
    vectorizer, model, X_test, y_test = train_model()
except Exception as error:
    st.error(f"Could not load the dataset or train the model: {error}")
    st.stop()

st.success("Machine learning model is ready!")
# Model Evaluation Section
st.divider()
st.header("📊 Model Evaluation")

X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

col1, col2 = st.columns(2)

col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Suspicious Job Precision", f"{precision:.2%}")

col3, col4 = st.columns(2)

col3.metric("Suspicious Job Recall", f"{recall:.2%}")
col4.metric("Suspicious Job F1-Score", f"{f1:.2%}")

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cm_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual Suspicious"],
    columns=["Predicted Normal", "Predicted Suspicious"]
)

st.dataframe(cm_df, use_container_width=True)
st.subheader("⚠️ Failure Case Analysis")
st.subheader("🔎 Examples of Misclassified Jobs")

# Find test examples where prediction was wrong
wrong_mask = (y_test.to_numpy() != y_pred)

wrong_jobs = X_test[wrong_mask].to_frame(name="Job Text")
wrong_jobs["Actual Label"] = y_test[wrong_mask].map({
    0: "Normal",
    1: "Potentially Suspicious"
}).values
wrong_jobs["Predicted Label"] = [
    "Potentially Suspicious" if p == 1 else "Normal"
    for p in y_pred[wrong_mask]
]

if not wrong_jobs.empty:
    st.dataframe(wrong_jobs.head(5), use_container_width=True)
else:
    st.success("No misclassified examples in this test split.")
false_positives = int(cm[0, 1])
false_negatives = int(cm[1, 0])

col1, col2 = st.columns(2)

col1.metric("False Positives", false_positives)
col2.metric("Missed Suspicious Jobs", false_negatives)

st.write(
    "**False Positive:** A normal job incorrectly classified as suspicious."
)

st.write(
    "**False Negative:** A suspicious job incorrectly classified as normal."
)

if false_positives > 0 or false_negatives > 0:
    st.warning(
        "The model makes some mistakes on the test dataset. "
        "Review these cases before relying on predictions."
    )
else:
    st.success(
        "No errors were found in this test split. "
        "This does not guarantee perfect real-world performance."
    )

title = st.text_input("Job Title", placeholder="e.g. Python Developer")
company = st.text_area("Company Profile (optional)", height=80)
description = st.text_area(
    "Job Description",
    height=180,
    placeholder="Paste the job responsibilities and offer details here..."
)
requirements = st.text_area("Requirements (optional)", height=100)
benefits = st.text_area("Benefits (optional)", height=80)

if st.button("Analyze Job Posting", type="primary", use_container_width=True):
    job_text = " ".join(
        [title, company, description, requirements, benefits]
    ).strip()

    if not job_text:
        st.warning("Please enter a job title or job description first.")
    else:
        job_tfidf = vectorizer.transform([job_text])
        prediction = int(model.predict(job_tfidf)[0])

        if prediction == 1:
            st.error("⚠️ Prediction: Potentially Suspicious")
            st.write(
                "The model detected patterns associated with "
                "fraudulent job postings. Verify the employer "
                "and never pay registration or application fees."
            )
        else:
            st.success("✅ Prediction: Normal")
            st.write(
                "The model classified this posting as normal. "
                "This does not guarantee that the job is genuine."
            )

        st.caption(
            "Prototype result only. Always verify job offers independently."
        )

st.divider()
st.subheader("Try an Example")

if st.button("Load Suspicious Job Example"):
    st.info(
        "Example: Work from home! Earn huge money daily. "
        "Pay a registration fee today to secure your job."
    )

st.caption("Built with Python, Pandas, Scikit-learn and Streamlit.")
