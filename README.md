# SWYNEX AI Problem Design: Fake Job Posting Classifier

## 1. Project Overview

The Fake Job Posting Classifier is a machine learning project that identifies potentially suspicious job postings and distinguishes them from normal job postings.

Fake job postings can mislead job seekers, collect personal information, or request fraudulent payments. This project uses Natural Language Processing (NLP) and machine learning to analyze job posting text.

## 2. Problem Statement

Job seekers may find it difficult to identify fake job advertisements. Manually checking every posting takes time and may not always reveal suspicious activity.

**Objective:** Build a machine learning prototype that classifies job postings as **Normal** or **Potentially Suspicious**.

## 3. Methodology

The project follows these steps:

1. **Data Collection:** Load the job posting dataset from `dataset/DataSet.csv`.
2. **Data Preprocessing:** Handle missing values, select relevant text fields, and prepare classification labels.
3. **Feature Extraction:** Use TF-IDF (Term Frequency–Inverse Document Frequency) to convert text into numerical features.
4. **Model Training:** Train a Logistic Regression classifier using the training dataset.
5. **Model Evaluation:** Measure performance using accuracy, precision, recall, F1-score, and a confusion matrix.
6. **Prediction:** Use a Streamlit web interface to analyze job posting details.

### Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
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

Ensure Python is installed. Open a terminal in the project directory and run:

```bash
python -m pip install -r requirements.txt
```

### Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal. Enter the job title, company profile, description, requirements, and benefits, then click **Analyze Job Posting**.

### Example 1: Potentially Suspicious Job

**Input:**
- Job Title: Work From Home — Easy Income
- Description: Earn a very high income with no experience required. Pay a registration fee to get started.

**Expected output:** Potentially Suspicious

### Example 2: Normal Job Posting

**Input:**
- Job Title: Junior Python Developer
- Description: Develop and maintain Python applications, collaborate with a development team, and participate in code reviews.

**Expected output:** Normal

*These examples illustrate expected behavior. Actual predictions depend on the trained model and are not proof that a job is fraudulent or legitimate.*

## 6. Model Evaluation

The prototype evaluates the model on a held-out test dataset. The recorded Streamlit evaluation results are:

| Metric | Result |
|---|---:|
| Accuracy | 96.34% |
| Precision — Suspicious Class | 57.66% |
| Recall — Suspicious Class | 91.33% |
| F1-score — Suspicious Class | 70.69% |
| False Positives | 116 |
| Missed Suspicious Jobs | 15 |

### Understanding the Results

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted suspicious postings that were actually suspicious.
- **Recall:** Proportion of suspicious postings correctly identified.
- **F1-score:** Combined measure of precision and recall.
- **False Positive:** A normal job incorrectly classified as suspicious.
- **False Negative:** A suspicious job incorrectly classified as normal.

These results describe the recorded test split and do not guarantee real-world performance.

## 7. Limitations

- Performance depends on the quality and representativeness of the training dataset.
- The model may not recognize new fraud patterns.
- Genuine job postings may be incorrectly flagged as suspicious.
- Some fraudulent postings may be classified as normal.
- Text alone cannot verify an employer's identity or guarantee a job's authenticity.
- Additional testing is needed before real-world deployment.

## 8. Ethical Considerations

- Treat predictions as preliminary warnings, not proof of fraud.
- Verify employers and job offers through official websites and trusted recruitment channels.
- Do not use the model as the sole basis for rejecting applicants, employers, or job advertisements.
- Handle personal information responsibly and avoid collecting unnecessary sensitive data.
- Review model errors regularly and communicate its limitations clearly.
- Use human judgment alongside model predictions.

## 9. Future Improvements

- Train and evaluate the model using larger and more diverse datasets.
- Improve text preprocessing and feature engineering.
- Compare additional machine learning algorithms.
- Provide clearer explanations for predictions.
- Test the application with users and monitor performance over time.

## 10. Conclusion

This project demonstrates how Natural Language Processing and machine learning can help screen job postings for potentially suspicious content. The Streamlit prototype allows users to enter job details, view predictions, and inspect model evaluation results.

The system is a decision-support tool and should be combined with independent verification and human judgment.

**Disclaimer:** This prototype is intended for educational purposes. Its predictions do not guarantee that a job posting is genuine or fraudulent.