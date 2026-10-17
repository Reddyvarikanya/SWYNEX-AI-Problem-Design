# SWYNEX AI Problem Design: Fake Job Posting Classifier

## 1. Project Overview

The Fake Job Posting Classifier is a machine learning project designed to identify potentially suspicious job postings and distinguish them from normal job postings.

Fake job postings may mislead job seekers, collect personal information, or involve fraudulent payment requests. This project uses natural language processing (NLP) and machine learning to analyze job descriptions and predict whether a posting is normal or potentially suspicious.

## 2. Problem Statement

Job seekers may find it difficult to distinguish genuine job opportunities from fraudulent postings. Manually checking every job advertisement takes time and may not always identify suspicious content.

**Objective:** Develop a machine learning prototype that analyzes job posting text and classifies it as either `Normal` or `Potentially Suspicious`.

## 3. Methodology

The project follows these steps:

1. **Data collection:** Load the job posting dataset from `dataset/DataSet.csv`.
2. **Data preprocessing:** Select relevant text fields, handle missing values, combine text, and prepare classification labels.
3. **Feature extraction:** Use TF-IDF (Term Frequency–Inverse Document Frequency) to convert text into numerical features.
4. **Model training:** Train a Logistic Regression classifier using the training dataset.
5. **Model evaluation:** Evaluate performance on a held-out test dataset using accuracy, precision, recall, F1-score, and a confusion matrix.
6. **Prediction:** Allow users to enter job posting details through a Streamlit interface and receive a prediction.

### Machine Learning Algorithms and Tools

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- Git and GitHub

## 4. Project Structure

```text
SWYNEX-AI-Problem-Design/
├── dataset/
│   └── DataSet.csv
├── result/
│   └── confusion_matrix.png
├── app.py
├── synex_ai_problem_design.py
├── requirements.txt
└── README.md
```

## 5. Demo and Usage

### Installation

Make sure Python is installed. Open a terminal in the project directory and run:

```bash
python -m pip install -r requirements.txt
```

### Run the Streamlit Application

```bash
python -m streamlit run app.py
```

The application opens in a browser. Enter the job title, company profile, description, requirements, and benefits, then click **Analyze Job Posting**.

### Example 1: Potentially Suspicious Job

**Input:**

- Job Title: Work From Home — Easy Income
- Description: Earn a very high income with no experience required. Pay a registration fee to get started.

**Expected output:** Potentially Suspicious

### Example 2: Normal Job Posting

**Input:**

- Job Title: Junior Python Developer
- Description: The candidate will develop and maintain Python applications, work with a development team, and participate in code reviews.

**Expected output:** Normal

*These examples illustrate expected behavior; actual predictions depend on the trained model.*

## 6. Model Evaluation

The prototype evaluates the classifier using a held-out test split.

The recorded evaluation results from the Streamlit application are:

| Metric | Result |
|---|---:|
| Accuracy | 96.34% |
| Precision — Suspicious Class | 57.66% |
| Recall — Suspicious Class | 91.33% |
| F1-score — Suspicious Class | 70.69% |
| False Positives | 116 |
| Missed Suspicious Jobs | 15 |

**Understanding the results:**

- **Accuracy:** The overall proportion of correct predictions.
- **Precision:** How many postings predicted as suspicious were actually suspicious.
- **Recall:** How many suspicious postings were correctly identified.
- **F1-score:** A combined measure of precision and recall.
- **False positive:** A normal job incorrectly predicted as suspicious.
- **False negative:** A suspicious job incorrectly predicted as normal.

The evaluation values describe the recorded test split and are not a guarantee of real-world performance.

## 7. Limitations

- The model depends on the quality, size, and representativeness of the training dataset.
- New fraud patterns may not be recognized by the trained model.
- Legitimate postings may be incorrectly flagged as suspicious.
- Some fraudulent postings may be classified as normal