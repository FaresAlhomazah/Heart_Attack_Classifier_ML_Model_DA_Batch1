# 🩺 Heart Attack Risk Prediction Project in South Africa 🇿🇦 with Machine Learning 🤖

## ✨ Overview

This project leverages the power of machine learning to predict heart attack risk in individuals from South Africa. 🇿🇦 By analyzing demographic details, medical history, lifestyle habits, and clinical measurements, we aim to provide insights into heart attack outcomes. We explored several algorithms and performed extensive data preprocessing, training, optimization, and evaluation to identify the best performing model. 🚀

## 📝 Dataset Description

This dataset contains information related to heart attack risk factors in individuals from South Africa. It includes demographic details, medical history, lifestyle habits, and clinical measurements to assess heart attack outcomes. The dataset is designed for predictive modeling, statistical analysis, and machine learning applications in healthcare research.

Key Features:

*   Demographics: Age, Gender
*   Clinical Data: Cholesterol, Blood Pressure, Triglycerides, LDL, HDL
*   Lifestyle Factors: Smoking, Alcohol Intake, Physical Activity, Diet, Stress
*   Medical History: Diabetes, Obesity, Family History of Heart Disease, Medication Usage
*   Target Variable: Heart Attack Outcome (0 = No, 1 = Yes)

Column Descriptions:

*   Patient_ID (Unique Identifier)
*   Age (Years)
*   Gender (Male/Female)
*   Cholesterol_Level (mg/dL)
*   Blood_Pressure_Systolic (mmHg)
*   Blood_Pressure_Diastolic (mmHg)
*   Smoking_Status (Yes/No)
*   Alcohol_Intake (Low/Moderate/High)
*   Physical_Activity (Sedentary/Active/Highly Active)
*   Obesity_Index (BMI)
*   Diabetes_Status (Yes/No)
*   Family_History_Heart_Disease (Yes/No)
*   Diet_Quality (Poor/Average/Good)
*   Stress_Level (Low/Medium/High)
*   Heart_Attack_History (Yes/No)
*   Medication_Usage (Yes/No)
*   Triglycerides_Level (mg/dL)
*   LDL_Level (mg/dL)
*   HDL_Level (mg/dL)
*   Heart_Attack_Outcome (0 = No, 1 = Yes)

## 🗂️ Files Included

*   heart_attack_south_africa.csv: 📊 The dataset for training and evaluation.
*   Heart_Attack_Risk_Prediction.ipynb: ⚙️ The main Jupyter Notebook containing the data analysis, preprocessing, model training, and evaluation logic.
*   heart_break_classifier_model.pkl: 🧠 The trained and serialized machine learning model, ready for deployment.
*   README.md: 📖 This file that you're currently reading.
*   requirements.txt: 📚 A list of all required Python libraries for this project.

## 🛠️ Libraries Used

*   pandas: 🐼 For data manipulation and analysis using DataFrames.
*   numpy: 🧮 For efficient numerical computation.
*   matplotlib.pyplot: 📈 For creating static, interactive, and animated visualizations.
*   seaborn: 📊 For visually appealing statistical plots.
*   scikit-learn (sklearn): ⚙️ A comprehensive machine learning library that includes:
    *   Pipeline: 🔗 For building sequential data processing steps.
    *   SimpleImputer: 🧹 For handling missing values.
    *   StandardScaler: ⚖️ For scaling numerical features.
    *   OneHotEncoder, OrdinalEncoder, LabelEncoder: 🏷️ For encoding categorical features.
    *   train_test_split, GridSearchCV, StratifiedKFold, cross_val_score: 🧪 For model selection and evaluation.
    *   accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve: 📉 Performance evaluation metrics.
*   imblearn: ⚖️ For handling imbalanced datasets with techniques like SMOTE.
*   xgboost (xgb): 🌲 For the XGBoost gradient boosting algorithm.
*   lightgbm (lgb): 💡 For the LightGBM gradient boosting algorithm.
*   warnings: 🚫 For suppressing warnings.
*   joblib: 💾 For saving and loading machine learning models.

## ⚙️ Project Workflow
1.  📥 Data Loading & Exploration:
    *   Suppresses warnings using warnings.filterwarnings('ignore').
    *   Loads the dataset from heart_attack_south_africa.csv using pandas.
    *   Displays the dataset shape, information, first few rows, and descriptive statistics.

2.  🔎 Missing Value Check:
    *   Identifies and prints missing values in each column.

3.  📊 Unique Object Identification:
    *   Identifies columns with object data types and prints the unique objects for each column.

4.  📉 Data Visualization:
    *   Plots the numerical data as histograms.
    *   Creates subplots for each categorical feature to display their distribution.

5.  📈 Correlation Analysis:
    *   Extracts numerical columns and calculates the correlation matrix.
    *   Sorts and displays correlation values with respect to Heart_Attack_Outcome.

6.  ✂️ Feature Selection:
    *   Drops less relevant columns based on correlation analysis and domain knowledge.

7.  🧹 Data Preprocessing:
    *   Identifies categorical and numerical columns.
    *   Creates a numerical pipeline with SimpleImputer (strategy='median') and StandardScaler.
    *   Creates a categorical pipeline with SimpleImputer (strategy='most_frequent') and OneHotEncoder.
    *   Combines the two pipelines using ColumnTransformer.

8.  🔀 Data Splitting and Balancing:
    *   Separates features (X) and target label (y).
    *   Applies preprocessing to features using the defined preprocessor.
    *   Splits the data into training and testing sets using train_test_split with stratification.
    *   Applies SMOTE (Synthetic Minority Over-sampling Technique) only on the training data to address class imbalance.

9.  🏋️ Model Training & Tuning:
    *   Utilizes GridSearchCV and StratifiedKFold for model selection and hyperparameter tuning.
    *   Tunes parameters for the following models:
        *   XGBoost Classifier: XGBClassifier
        *   Stochastic Gradient Descent Classifier: SGDClassifier
        *   Support Vector Classifier: SVC
        *   LightGBM Classifier: LGBMClassifier

10. 🧪 Model Evaluation:
    *   Evaluates the performance of the best models using:
        *   Accuracy Score
        *   Classification Report
        *   Confusion Matrix
        *   ROC AUC Score and Curve

11. 📦 Model Deployment:
    *   Creates a full pipeline that includes preprocessing steps and the best performing model.
    *   Trains the full pipeline on the entire dataset.
    *   Saves the trained model using joblib for later use.

## 🤖 Algorithms Used

*   XGBoost (XGBClassifier): Gradient boosting algorithm known for its high performance. 🌲
*   Stochastic Gradient Descent (SGDClassifier): Linear model optimized with gradient descent. 📉
*   Support Vector Classifier (SVC): Powerful model that finds the best hyperplane to separate classes. 💪
*   LightGBM (LGBMClassifier): Gradient boosting framework that uses tree-based learning algorithms. 💡
*   SMOTE (Synthetic Minority Over-sampling Technique): Over-sampling technique to handle imbalanced datasets. ⚖️

🏆 Best Model: After extensive evaluation, LightGBM (LGBMClassifier) provided the best balance of performance metrics.

## 🚀 How to Use the Model

1.  Install Requirements:
   
    pip install -r requirements.txt
    
2.  Run the Project:
    *   Open Heart_Attack_Risk_Prediction.ipynb in Jupyter Notebook or Jupyter Lab.
    *   Execute the cells sequentially to reproduce the analysis and results.

3.  Load and Use the Trained Model:
    *   The code snippet below shows how to load the trained model from heart_break_classifier_model.pkl and use it for predictions. Make sure that you have installed all necessary libraries.

    `python
    import joblib
    from sklearn.metrics import accuracy_score

    # Load the trained model
    heart_attack_model = joblib.load("heart_break_classifier_model.pkl")
# Assuming you have X_test and y_test from your data splitting step
    # Make predictions using the loaded model
    y_pred = heart_attack_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
   

## 📝 Additional Notes

*   Extensive data exploration and preprocessing were performed to prepare the data for modeling.
*   SMOTE was applied to address class imbalance, improving model performance.
*   `GridSearchCV` was used for hyperparameter tuning to optimize model performance.
*   The project includes detailed visualizations and performance metrics to evaluate model effectiveness.

## 💻 Model Deployment Code:

The following code snippet demonstrates how the full pipeline is created and the trained model is saved:

python
full_pipeline_with_clf = Pipeline([
        ("preparation", preprocessor),
        ("model", best_model)
    ])

full_pipeline_with_clf.fit(X, y)
heart_break_classifier_model = full_pipeline_with_clf

import joblib
joblib.dump(heart_break_classifier_model, "heart_break_classifier_model.pkl")
load_the_model = joblib.load("heart_break_classifier_model.pkl")
accuracy = accuracy_score(y_test, best_lgb_model.predict(X_test))
print(f'Accuracy: {accuracy:.2f}')



## 🎉 Conclusion

This project provides a comprehensive approach to predicting heart attack risk using machine learning in South Africa. By combining various techniques, we have developed an effective model that can assist in identifying individuals at high risk. Feel free to explore, contribute, and improve upon this project!

## 🧑‍💻 Team Members 
* Fares Alhomazah: [LinkedIn Profile](https://www.linkedin.com/in/fares-abdulghani-alhomazah-6b1802288?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
* Ahmed Aljaifi: [LinkedIn Profile](https://www.linkedin.com/in/ahmed-al-jaifi-ab213617a)
