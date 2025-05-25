
# Industrial IoT Predictive Maintenance Project

## Project Overview
This project aims to predict machine failures and remaining useful life (RUL) using supervised learning techniques on an Industrial IoT synthetic dataset. The dataset includes sensor data and machine status information for predictive maintenance and failure forecasting tasks.

## Dataset Description
The dataset contains the following features:
- Machine_ID: Unique identifier for each machine
- Machine_Type: Type of machine (e.g., Mixer, Industrial Chiller)
- Installation_Year: Year the machine was installed
- Operational_Hours: Total operational hours of the machine
- Temperature_C: Temperature in Celsius
- Vibration_mms: Vibration in mm/s
- Sound_dB: Sound level in dB
- Oil_Level_pct: Oil level percentage
- Coolant_Level_pct: Coolant level percentage
- Power_Consumption_kW: Power consumption in kW
- Last_Maintenance_Days_Ago: Days since last maintenance
- Maintenance_History_Count: Number of maintenance events
- Failure_History_Count: Number of failure events
- AI_Supervision: Whether AI supervision is enabled
- Error_Codes_Last_30_Days: Number of error codes in the last 30 days
- Remaining_Useful_Life_days: Remaining useful life in days (target for regression)
- Failure_Within_7_Days: Whether the machine will fail within 7 days (target for classification)
- Additional sensor features: Laser_Intensity, Hydraulic_Pressure_bar, Coolant_Flow_L_min, Heat_Index, AI_Override_Events

## Steps Taken
1. **Data Loading and Initial Exploration**
   - Loaded the dataset using pandas.
   - Performed initial exploration using `df.head()`, `df.info()`, `df.describe()`, and `df.isnull().sum()`.

2. **Exploratory Data Analysis (EDA)**
   - Visualized the distribution of numerical features using histograms and KDE plots.
   - Analyzed the correlation matrix to understand feature relationships.
   - Examined the distribution of the target variable `Failure_Within_7_Days`.

3. **Data Preprocessing**
   - Handled missing values by filling with mean (for numerical) and mode (for categorical).
   - Applied Label Encoding to categorical features.
   - Scaled numerical features using Min-Max scaling.
   - Split the data into training and testing sets (80% train, 20% test).

4. **Modeling and Evaluation**
   - **Regression (Predicting Remaining Useful Life)**
     - Linear Regression: Baseline model to predict RUL.
     - XGBoost Regressor: Used 10% of training data for faster training.
     - Evaluated models using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score.
   - **Classification (Predicting Failure Within 7 Days)**
     - Logistic Regression: Simple and fast model for binary classification.
     - Random Forest Classifier: More powerful model using multiple decision trees.
     - Evaluated models using Accuracy, Precision, Recall, and F1 Score.

## Algorithms Used
- **Linear Regression**: Chosen as a baseline model for its simplicity and interpretability.
- **XGBoost Regressor**: Selected for its high performance and ability to handle large datasets efficiently.
- **Logistic Regression**: Used for binary classification due to its simplicity and effectiveness.
- **Random Forest Classifier**: Chosen for its robustness and ability to handle complex datasets.

## Model Evaluation
- **Regression Models**
  - Linear Regression: MSE = 0.0471, MAE = 0.1276, R² = 0.1586
  - XGBoost Regressor: MSE = 0.0257, MAE = 0.0498, R² = 0.5406
- **Classification Models**
  - Logistic Regression: Accuracy = 0.9646, Precision = 0.7195, Recall = 0.6659, F1 Score = 0.6917
  - Random Forest Classifier: Accuracy = 0.9633, Precision = 0.7093, Recall = 0.6508, F1 Score = 0.6788

## Conclusion
- The XGBoost Regressor outperformed Linear Regression in predicting the remaining useful life of machines.
- Logistic Regression and Random Forest Classifier both performed well in predicting machine failures within 7 days, with Logistic Regression slightly outperforming Random Forest.
- Future work could include hyperparameter optimization, unsupervised learning, and deploying the model using a web interface.

## Kaggle Notebook
[Link to Kaggle Notebook](#)

