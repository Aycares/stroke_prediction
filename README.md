# Stroke Prediction with ZenML Pipeline

## 1. Introduction  
Stroke is a serious medical condition that occurs when the blood supply to part of the brain is interrupted or reduced, preventing brain tissue from getting enough oxygen and nutrients. Early detection of individuals at high risk of stroke can help in taking preventive measures and saving lives.  

This project applies **machine learning** to predict the likelihood of a stroke using patient medical and demographic data. The pipeline is built with **ZenML**, which ensures the workflow is **modular, reproducible, and easy to deploy**.  

By the end of this project:
- A trained ML model will be available to predict stroke risk.
- The model will be deployed as a REST API using FastAPI.
- The API will have Swagger documentation for easy testing.

---

## 2. Project Objective  
The goal of this project is to:
1. **Build** an end-to-end machine learning pipeline for stroke prediction.
2. **Address class imbalance** (since strokes are rare in the dataset).
3. **Deploy** the trained model as an API for real-time predictions.
4. **Ensure reproducibility** using ZenML pipelines.

---

## 3. Dataset Information  
The dataset used is the **Healthcare Stroke Dataset** from Kaggle.  
It contains patient attributes such as:
- **gender** – Male, Female, or Other.
- **age** – Patient’s age in years.
- **hypertension** – 0 if the patient does not have hypertension, 1 if they do.
- **heart_disease** – 0 if the patient does not have a heart disease, 1 if they do.
- **ever_married** – Yes or No.
- **work_type** – Private, Self-employed, Govt_job, Children, Never_worked.
- **Residence_type** – Urban or Rural.
- **avg_glucose_level** – Average glucose level in blood (mg/dL).
- **bmi** – Body Mass Index.
- **smoking_status** – formerly smoked, never smoked, smokes, or Unknown.
- **stroke** – Target variable: 0 = No stroke, 1 = Stroke.

---

## 4. Challenges in the Dataset  
- **Class imbalance:** Most patients do not have a stroke (around 95% "No Stroke", 5% "Stroke").
- **Missing values:** BMI contains some missing values.
- **Categorical features:** Need encoding for machine learning models.

---

## 5. Pipeline Workflow  

### Step 1: **Data Loading**  
The dataset is loaded from a CSV file into a Pandas DataFrame.

### Step 2: **Data Preprocessing**
- Handle missing values in BMI.
- Encode categorical features (e.g., gender, work_type).
- Scale numerical features (e.g., age, avg_glucose_level, bmi) to standardize the ranges.

### Step 3: **Model Training**  
We train an **XGBoost Classifier** with tuned hyperparameters to improve prediction performance. XGBoost is chosen because it:
- Handles missing values well.
- Works well with mixed numerical and categorical features.
- Is less prone to overfitting with proper regularization.

### Step 4: **Model Evaluation**  
We evaluate the model using:
- **Accuracy** – Overall correctness.
- **Precision** – Proportion of predicted strokes that are actual strokes.
- **Recall** – Proportion of actual strokes correctly predicted.
- **F1-score** – Harmonic mean of precision and recall.

Given the healthcare context, **recall** is particularly important (catching as many actual stroke cases as possible).

### Step 5: **Model Deployment**  
The trained model is deployed using **FastAPI**:
- An endpoint `/predict` accepts patient data in JSON format.
- Returns the predicted stroke risk (0 or 1) along with confidence.

---

### 6.  Key Learnings from the Project
ZenML helps create reproducible ML pipelines that are easy to maintain.

XGBoost provides strong performance when tuned properly.

In healthcare ML, recall is often more important than accuracy because missing a positive case can have serious consequences.

### 7. Conclusion

This project demonstrates how machine learning, combined with ZenML pipelines and FastAPI deployment, can create an efficient, reproducible, and production-ready stroke prediction system.

By integrating data preprocessing, class balancing, and XGBoost model training into an automated workflow, the project addresses real-world challenges like class imbalance and the need for high recall in healthcare.

The result is a solution that not only predicts stroke risk accurately but is also ready for real-time deployment and scaling. This system can be extended into a fully functional medical decision-support tool.


Author
Adeleye Ayokunle






