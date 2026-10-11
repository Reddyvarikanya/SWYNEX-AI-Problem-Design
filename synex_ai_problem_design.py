import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/DataSet.csv")


# ==========================================
# 2. CONVERT TARGET LABELS
# ==========================================

df["fraudulent"] = df["fraudulent"].map({
    "f": 0,
    "t": 1
})


# ==========================================
# 3. SELECT TEXT COLUMNS
# ==========================================

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]


# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

for column in text_columns:
    df[column] = df[column].fillna("")


# ==========================================
# 5. COMBINE TEXT
# ==========================================

df["job_text"] = (
    df["title"] + " " +
    df["company_profile"] + " " +
    df["description"] + " " +
    df["requirements"] + " " +
    df["benefits"]
)


# ==========================================
# 6. REMOVE EMPTY TEXT
# ==========================================

df = df[df["job_text"].str.strip() != ""]


# ==========================================
# 7. INPUT AND TARGET
# ==========================================

X = df["job_text"]
y = df["fraudulent"]


# ==========================================
# 8. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 9. TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print("TF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# ==========================================
# 10. TRAIN LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# ==========================================
# 11. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test_tfidf)


# ==========================================
# 12. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Potentially Suspicious"
        ]
    )
)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 14. CONFUSION MATRIX GRAPH
# ==========================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Normal",
        "Suspicious"
    ],
    yticklabels=[
        "Normal",
        "Suspicious"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Fake Job Posting Classifier - Confusion Matrix"
)

plt.tight_layout()


# ==========================================
# 15. SAVE GRAPH
# ==========================================

plt.savefig(
    "result/confusion_matrix.png"
)

plt.show()







# ==========================================
# 16. INTERACTIVE JOB POSTING PREDICTION
# ==========================================

print("\n" + "=" * 45)
print("FAKE JOB POSTING CLASSIFIER")
print("=" * 45)

print("\nEnter a job description below.")
print("Type 'exit' to stop the program.")

while True:
    job_description = input("\nEnter job description: ").strip()

    if job_description.lower() == "exit":
        print("Thank you for using the classifier!")
        break

    if not job_description:
        print("Please enter a job description.")
        continue

    # Convert the entered text into TF-IDF features
    job_tfidf = vectorizer.transform([job_description])

    # Predict using the already-trained model
    prediction = model.predict(job_tfidf)[0]

    print("\nPrediction:")
    if prediction == 1:
        print("Potentially Suspicious")
    else:
        print("Normal")

    print("\nNote: This is an automated prediction,")
    print("not proof that a job is genuine or fraudulent.")



