# SWYNEX-AI-Problem-Design
Fake Job Posting Classifier using TF-IDF and Logistic Regression for the SWYNEX AI Problem Design task.



## Task 2: Working Prototype

### Project Overview
A Streamlit web application that analyzes job postings using a machine learning model and classifies them as **Normal** or **Potentially Suspicious**.

### Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

### How to Run
1. Clone or download this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in the terminal.

### Example Inputs and Outputs

**Example 1: Suspicious Job**
- Input: A work-from-home job promising unusually high income and requesting a registration fee.
- Expected output: Potentially Suspicious.

**Example 2: Normal Job**
- Input: A Junior Python Developer role describing programming responsibilities and relevant technical skills.
- Expected output: Normal.

*Actual predictions depend on the trained model. These results are screening signals, not proof of fraud or legitimacy.*

### Limitations
The prototype uses a dataset-trained machine learning model. Its predictions may be incorrect, and users should independently verify employers and job offers.