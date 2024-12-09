## **Model Card**

### **Model Overview**
- **Model Name**: Accident Severity Prediction
- **Objective**: To predict accident severity (Fatal, Serious, Slight) based on environmental, road, and location features.
- **Type of Task**: Multi-class classification
- **Models Used**:
  - **Extra Trees Classifier**: An ensemble method for robust predictions.
  - **Naive Bayes Classifier**: Probabilistic classifier based on Bayes' theorem.
  - **Decision Tree Classifier**: A tree-based method for interpretable predictions.

---

### **Intended Use**
- **Target Users**:
  - Road safety analysts
  - Transportation authorities
  - Emergency response teams
- **Applications**:
  - Identifying accident hotspots
  - Allocating resources for traffic management
  - Developing safety measures based on key predictors

---

### **Datasets**
- **Source**: UK Accident Dataset
- **Size**:
  - Rows: Up to 50,000 sampled
  - Columns: Latitude, Longitude, Date, Day_of_Week, etc.
- **Target Variable**: `Accident_Severity` (mapped to classes: 0 = Fatal, 1 = Serious, 2 = Slight)

---

### **Metrics**
- **Evaluation Metrics**:
  - **Accuracy**: Overall correctness of predictions.
  - **Mean Squared Error (MSE)**: Quantifies prediction errors.
  - **Precision, Recall, F1-Score**: For detailed performance across classes.
  - **Confusion Matrix**: Visualization of true vs. predicted labels.

---

### **Performance**
Evaluating Extra Trees...
Accuracy: 0.96
Mean Squared Error: 0.04

Extra Trees Classification Report:
              precision    recall  f1-score   support

           0       0.99      1.00      1.00     10949
           1       0.91      1.00      0.95     10840
           2       1.00      0.89      0.94     10796

    accuracy                           0.96     32585
   macro avg       0.97      0.96      0.96     32585
weighted avg       0.97      0.96      0.96     32585


Evaluating Naive Bayes...
Accuracy: 0.38
Mean Squared Error: 1.44

Naive Bayes Classification Report:
              precision    recall  f1-score   support

           0       0.39      0.53      0.45     10949
           1       0.34      0.06      0.11     10840
           2       0.36      0.53      0.43     10796

    accuracy                           0.38     32585
   macro avg       0.36      0.37      0.33     32585
weighted avg       0.36      0.38      0.33     32585


Evaluating Decision Tree...
Accuracy: 0.94
Mean Squared Error: 0.07

Decision Tree Classification Report:
              precision    recall  f1-score   support

           0       0.99      1.00      0.99     10949
           1       0.87      1.00      0.93     10840
           2       1.00      0.83      0.91     10796

    accuracy                           0.94     32585
   macro avg       0.95      0.94      0.94     32585
weighted avg       0.95      0.94      0.94     32585


---

### **Limitations**
1. **Data Bias**: Dataset may overrepresent specific accident types or regions.
2. **Feature Availability**: Model assumes all relevant features are present; missing features may degrade performance.
3. **Imbalanced Data**: Requires oversampling techniques like `RandomOverSampler` to balance classes.

---

### **Ethical Considerations**
- **Fairness**: Ensure that predictions are not biased toward specific geographic or demographic groups.
- **Privacy**: Protect sensitive location data and personal information if present.
- **Safety**: Misclassification of severe accidents may lead to underprepared responses.

---

## **Data Sheet**

### **Motivation**
1. **Why is this dataset created?**
   - To analyze and predict the severity of road accidents for improved traffic management and safety measures.
2. **Who funded the dataset?**
   - Sourced from Kaggle. https://www.kaggle.com/datasets/devansodariya/road-accident-united-kingdom-uk-dataset

---

### **Composition**
1. **Features**:
   - **Numerical**: Latitude, Longitude, day, month, `Speed_limit`.
   - **Categorical**: Day_of_Week, `Urban_or_Rural_Area`.
2. **Size**: 50,000 records (sampled).
3. **Target Variable**: `Accident_Severity` (Multi-class: Fatal, Serious, Slight).

---

### **Collection Process**
1. **How is the data collected?**
   - Through official accident reports, sensor data, and location-based inputs.
2. **What is the timeframe?**
   - Likely historical data spanning multiple years.

---

### **Preprocessing Steps**
1. Handle missing values:
   - Drop rows with null values in key columns (`Longitude`, `Time`, etc.).
2. Categorical encoding:
   - Apply `LabelEncoder` for categorical variables.
3. Scaling:
   - Standardize numerical features using `StandardScaler`.
4. Class balancing:
   - Use `RandomOverSampler` to handle imbalanced target classes.

---

### **Usage**
1. **Primary Use**:
   - Predicting accident severity for actionable insights.
2. **Not Suitable For**:
   - Predicting individual driver behavior or direct causation analysis.

---

### **Ethics and Privacy**
1. **Sensitive Information**:
   - Ensure location data and personal identifiers are anonymized.
2. **Bias Awareness**:
   - Regular audits to verify model fairness across demographic and geographic boundaries.
