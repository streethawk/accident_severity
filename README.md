
# **Model Card**

## **Model Overview**
- **Model Name**: Accident Severity Prediction
- **Objective**: Predict the severity of road accidents (Fatal, Serious, Slight) based on environmental, road, and location data.
- **Type of Task**: Multi-class classification
- **Models Used**:
  - **Extra Trees Classifier**: Ensemble method offering robust predictions.
  - **Naive Bayes Classifier**: Probabilistic model leveraging Bayes' theorem.
  - **Decision Tree Classifier**: Tree-based approach for interpretable results.

---

## **Intended Use**
### **Target Users**
- Road safety analysts
- Transportation authorities
- Emergency response teams

### **Applications**
- Identifying accident-prone areas
- Resource allocation for traffic management
- Developing targeted safety measures

---

## **Datasets**
- **Source**: UK Accident Dataset (2000–2018)
- **Sample Size**: 
  - Rows: 50,000 (sampled)
  - Columns: Latitude, Longitude, Date, Day_of_Week, etc.
- **Target Variable**: `Accident_Severity` (classes: 0 = Fatal, 1 = Serious, 2 = Slight)

### **Dataset Description**
- **Context**: Comprehensive dataset covering over 1.8 million accidents, offering insights into changes in traffic patterns and safety over time.
- **Content**: Features include:
  - `Latitude`, `Longitude`: Accident location
  - `Accident_Severity`: Severity scale (1-5)
  - `Number_of_Vehicles`: Vehicles involved
  - `Number_of_Casualties`: Casualties
  - `Light_Conditions`, `Weather_Conditions`, `Road_Surface_Conditions`: Environment factors
  - `Year`: Year of the event

### **Inspiration**
Key questions include:
- Which days of the week see the most casualties?
- How do speed limits vary with accidents across different days?
- What role do light and weather conditions play in predicting severity?
- Can the data improve accident severity prediction?

---

## **Metrics**
- **Evaluation Metrics**:
  - **Accuracy**: Overall model performance.
  - **Mean Squared Error (MSE)**: Measures prediction error.
  - **Precision, Recall, F1-Score**: Provides class-specific insights.
  - **Confusion Matrix**: Visualizes true vs. predicted labels.

---

## **Performance**
### **Extra Trees Classifier**
- **Accuracy**: 96%
- **Mean Squared Error**: 0.04
- **Classification Report**:
  - Precision: 97%
  - Recall: 96%
  - F1-Score: 96%

### **Naive Bayes Classifier**
- **Accuracy**: 38%
- **Mean Squared Error**: 1.44
- **Classification Report**:
  - Precision: 36%
  - Recall: 37%
  - F1-Score: 33%

### **Decision Tree Classifier**
- **Accuracy**: 94%
- **Mean Squared Error**: 0.07
- **Classification Report**:
  - Precision: 95%
  - Recall: 94%
  - F1-Score: 94%

---

## **Limitations**
1. **Data Bias**: Dataset may favor certain accident types or regions.
2. **Feature Availability**: Missing features can impact performance.
3. **Imbalanced Data**: Requires techniques like `RandomOverSampler` for balancing.

---

## **Ethical Considerations**
- **Fairness**: Avoid geographic or demographic bias.
- **Privacy**: Ensure sensitive location and personal data are anonymized.
- **Safety**: Minimize severe accident misclassification to prevent underprepared responses.

---

## **Data Sheet**

### **Motivation**
1. **Purpose**: To analyze and predict accident severity for improved traffic management and safety.
2. **Source**: Kaggle dataset [link](https://www.kaggle.com/datasets/devansodariya/road-accident-united-kingdom-uk-dataset).

### **Composition**
1. **Features**:
   - **Numerical**: Latitude, Longitude, Day, Month, Speed_limit.
   - **Categorical**: Day_of_Week, Urban_or_Rural_Area.
2. **Sample Size**: 50,000 records.
3. **Target Variable**: `Accident_Severity`.

### **Collection Process**
1. **Data Collection**: Accident reports, sensor data, location-based data.
2. **Timeframe**: Historical data spanning multiple years.

---

## **Preprocessing Steps**
1. **Handling Missing Values**:
   - Drop rows with null values in key columns (`Longitude`, `Time`, etc.).
2. **Categorical Encoding**:
   - Use `LabelEncoder` for categorical features.
3. **Feature Scaling**:
   - Standardize numerical variables using `StandardScaler`.
4. **Class Balancing**:
   - Apply `RandomOverSampler` to address imbalanced classes.

---

## **Usage**
1. **Primary Use**:
   - Accident severity prediction for actionable insights.
2. **Not Suitable For**:
   - Individual driver behavior prediction or causation analysis.

---

## **Ethics and Privacy**
1. **Sensitive Information**:
   - Ensure anonymization of location and personal data.
2. **Bias Awareness**:
   - Conduct audits to maintain fairness across demographics and geographies.
