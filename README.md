# 🤖 AI Job Description Analyzer

An AI and Machine Learning based web application that analyzes job descriptions and evaluates their overall quality.

The system uses Natural Language Processing (NLP), TF-IDF vectorization, and Logistic Regression to classify job descriptions into different quality categories and provide useful recommendations.

---

## 📌 Project Overview

The AI Job Description Analyzer helps recruiters, HR professionals, and organizations identify the quality and completeness of a job description.

The system analyzes the provided job description and identifies:

- Job Description Quality
- Quality Score
- Model Confidence
- Required Skills
- Missing / Recommended Skills
- Required Experience
- Educational Qualification
- Salary Information
- Issues in the Job Description
- Suggestions and Recommendations

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze job descriptions using Machine Learning.
2. Classify job descriptions based on their quality.
3. Detect important information from job descriptions.
4. Identify missing or recommended skills.
5. Detect incomplete job description information.
6. Provide suggestions to improve job descriptions.
7. Provide an easy-to-use web-based interface.

---

## 🧠 Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Regular Expressions (Regex)
- HTML
- CSS
- Jinja2
- Excel Dataset

---

## 📊 Machine Learning Approach

The project follows these major steps:

### 1. Dataset Collection

A dataset containing job descriptions and their quality labels is used for training the machine learning model.

The dataset contains information such as:

- Job Title
- Job Description
- Required Skills
- Experience
- Education
- Responsibilities
- Salary
- Benefits
- Quality Score
- Quality Label
- Issues

---

### 2. Data Preprocessing

The job descriptions are cleaned before training.

The preprocessing includes:

- Converting text to lowercase
- Removing unnecessary special characters
- Removing extra spaces
- Preparing text for feature extraction

---

### 3. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text into numerical features that can be processed by the machine learning model.

---

### 4. Model Training

A Logistic Regression classifier is trained using the processed job description data.

The model predicts one of the following quality labels:

- Good
- Average
- Poor

The dataset is divided into training and testing sets using an 80/20 split.

---

## 📈 Model Evaluation

The trained model was evaluated using a held-out test set.

The model achieved:

**99% accuracy on the prepared test dataset.**

> Note: The dataset used for this project is a prepared/synthetic dataset, so this accuracy should not be interpreted as real-world performance without testing on an independent real-world dataset.

---

## 🌐 Web Application

The Flask-based web application provides an interactive interface where users can enter a job description and receive an automated analysis.

### The system displays:

### Quality Analysis
- Quality Label
- Quality Score
- Model Confidence

### Skill Analysis
- Detected Skills
- Missing / Recommended Skills

### Job Information
- Experience
- Education
- Salary

### Quality Issues
- Missing information
- Incomplete responsibilities
- Other detected issues

### Recommendations
- Suggestions for improving the job description

---

## 📁 Project Structure

```text
AI_Job_Description_Analyzer/
│
├── app.py
├── preprocessing.py
├── train_model.py
├── requirements.txt
│
├── job_description_model.pkl
├── tfidf_vectorizer.pkl
├── processed_job_description_dataset.csv
│
├── AI_Job_Description_Dataset_1000 (1).xlsx
│
├── templates/
│   └── index.html
│
└── README.md
```

---

## ⚙️ Installation

### Step 1: Install Python

Make sure Python is installed on your system.

### Step 2: Install Required Libraries

Open the terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Run the Flask application using:

```bash
python app.py
```

After running the application, open the following address in your browser:

```text
http://127.0.0.1:5000
```

---

## 🧪 Example Input

```text
We are looking for a Python Developer with 2 years of experience.

The candidate should have a Bachelor's degree in Computer Science.

Required skills include Python, Flask, SQL and JavaScript.

Responsibilities include developing web applications,
maintaining software, fixing bugs and working with the
development team.

Salary is 80000-120000 PKR with health insurance and annual bonus.
```

---

## 📋 Example Output

The system analyzes the job description and provides:

* Quality Label: Good
* Quality Score
* Model Confidence
* Detected Skills
* Missing / Recommended Skills
* Experience
* Education
* Salary
* Issues Detected
* Suggestions / Recommendations

---

## 🔄 System Workflow

```text
Job Description
       ↓
Text Preprocessing
       ↓
TF-IDF Feature Extraction
       ↓
Logistic Regression Model
       ↓
Quality Prediction
       ↓
Information Extraction
       ↓
Issue Detection
       ↓
Recommendations
       ↓
Final Analysis
```

---

## ⭐ Key Features

* 🤖 Machine Learning based classification
* 📝 Job description analysis
* 🛠️ Skill detection
* ❌ Missing skill identification
* 🎓 Education detection
* 💼 Experience detection
* 💰 Salary detection
* ⚠️ Issue detection
* 💡 Recommendations
* 📊 Quality scoring
* 🌐 User-friendly Flask interface

---

## 🔮 Future Enhancements

Future versions of the system can include:

* Support for more job domains
* Larger real-world datasets
* Advanced NLP models
* Transformer-based models such as BERT
* Automatic job description improvement
* Resume-to-job matching
* Multi-language support
* Advanced analytics dashboard

