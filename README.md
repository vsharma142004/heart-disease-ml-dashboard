# ❤️ Heart Disease Risk Prediction using Machine Learning

## 📌 Project Overview

Cardiovascular diseases are one of the leading causes of death worldwide. Early detection of heart disease risk can help healthcare professionals take preventive measures and improve patient outcomes.

This project develops a **supervised machine learning model** to predict the likelihood of heart disease using structured clinical data from the **UCI Cleveland Heart Disease dataset**. The project includes data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and deployment of an **interactive web application using Streamlit**.

The deployed application allows users to input patient health parameters and receive **real-time predictions of heart disease risk** along with a visual risk indicator.

---

## 🚀 Live Demo

You can interact with the deployed application here:

**Streamlit App:**
https://huggingface.co/spaces/vsharma142004/Heart_Disease_Classifier

---

## 🧠 Problem Statement

Heart disease diagnosis often requires multiple clinical tests and expert interpretation. Machine learning can assist medical professionals by identifying patterns in patient data and predicting potential risk.

This project aims to:

* Build a machine learning model to **classify heart disease risk**
* Analyze important clinical features affecting heart health
* Provide an **interactive prediction interface for users**

---

## 📊 Dataset

The model is trained using the **UCI Cleveland Heart Disease Dataset**.

Dataset Source:
https://archive.ics.uci.edu/ml/datasets/heart+Disease

The dataset contains **clinical attributes of patients**, including:

| Feature  | Description                       |
| -------- | --------------------------------- |
| age      | Age of patient                    |
| sex      | Gender (1 = male, 0 = female)     |
| cp       | Chest pain type                   |
| trestbps | Resting blood pressure            |
| chol     | Serum cholesterol                 |
| fbs      | Fasting blood sugar               |
| restecg  | Resting ECG results               |
| thalach  | Maximum heart rate achieved       |
| exang    | Exercise induced angina           |
| oldpeak  | ST depression induced by exercise |
| slope    | Slope of peak exercise ST segment |
| ca       | Number of major vessels           |
| thal     | Thalassemia status                |
| target   | Presence of heart disease         |

---

## ⚙️ Project Pipeline

### 1️⃣ Data Preprocessing

* Handled missing values in the dataset
* Converted categorical variables where necessary
* Applied **StandardScaler** for feature normalization
* Split data into **training and testing sets**

### 2️⃣ Exploratory Data Analysis (EDA)

EDA was performed to understand patterns in the dataset, including:

* Feature distributions
* Class imbalance
* Correlation between medical attributes
* Impact of different clinical variables on heart disease

### 3️⃣ Model Training

The model was built using **TensorFlow/Keras** with supervised learning techniques.

Training steps included:

* Train-test split
* Model architecture design
* Cross-validation
* Regularization using dropout
* Hyperparameter tuning

### 4️⃣ Model Evaluation

The final model performance was evaluated using multiple classification metrics.

**Results:**

| Metric   | Score |
| -------- | ----- |
| Accuracy | 83%   |
| F1 Score | 0.83  |
| Recall   | 86%   |

These metrics indicate the model performs well in identifying patients at risk of heart disease.

---

## 🌐 Web Application

An interactive **Streamlit dashboard** was developed to make the model accessible to users.

### Features of the Web App

* Patient health parameter input
* Real-time heart disease risk prediction
* Probability-based risk indicator
* Interactive risk gauge visualization
* Dataset exploration tools

The application is deployed on **Hugging Face Spaces** for public access.

---

## 🛠️ Tech Stack

| Category             | Tools Used          |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Machine Learning     | TensorFlow / Keras  |
| Data Processing      | Pandas, NumPy       |
| Visualization        | Plotly              |
| Model Deployment     | Streamlit           |
| Hosting              | Hugging Face Spaces |

---

## 📁 Project Structure

```
heart-disease-ml-dashboard
│
├── streamlit_app.py
├── heart_model.h5
├── scaler.pkl
├── processed.cleveland.data
├── requirements.txt
└── README.md
```

---

## 📈 Example Prediction Workflow

1. User enters patient medical attributes
2. Input features are scaled using the trained scaler
3. Model processes the data
4. Prediction probability is generated
5. Risk level is displayed in the dashboard

---

## ⚠️ Disclaimer

This application is built for **educational and research purposes only**.

It should **not be used as a substitute for professional medical diagnosis or treatment**.

---

## 🔮 Future Improvements

Possible improvements for this project include:

* Implement **SHAP explainability** for model predictions
* Train on larger cardiovascular datasets
* Deploy a **REST API for model access**
* Add authentication for medical professional access
* Improve model performance with ensemble methods

---

## 👨‍💻 Author

**Vansh Sharma**

GitHub:
https://github.com/vsharma142004

LinkedIn:
(www.linkedin.com/in/vanshsharma142004)

---

## ⭐ Acknowledgements

* UCI Machine Learning Repository for the dataset
* Streamlit for rapid ML application development
* Hugging Face Spaces for hosting the application
