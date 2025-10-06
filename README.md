Customer Churn Prediction using Machine Learning
This project uses machine learning to predict customer churn for a telecommunications company. By analyzing various customer attributes and service usage, the model can identify customers who are likely to churn (leave the company). This allows the company to proactively intervene and improve customer retention.

Project Overview
The goal of this project is to build and evaluate a machine learning model that can predict customer churn. The key steps involved are:

Data Loading and Exploration: Loading the dataset and performing an initial analysis to understand its structure, identify missing values, and check for class imbalance.

Data Preprocessing: Cleaning the data by handling missing values, encoding categorical features, and addressing class imbalance using the Synthetic Minority Oversampling Technique (SMOTE).

Model Training: Training and evaluating several classification models, including Decision Tree, Random Forest, and XGBoost, using cross-validation.

Model Evaluation: Assessing the performance of the best-performing model on a test set using key metrics like accuracy, a confusion matrix, and a classification report.

Predictive System: Building a simple system to demonstrate how the trained model can be used to make predictions on new customer data.

Dataset
The dataset used is named WA_Fn-UseC_-Telco-Customer-Churn.csv. It contains 7,043 customer records and 21 columns, which include demographic information, service subscriptions, and monthly and total charges.

The target variable is Churn, a categorical column indicating whether a customer has churned (Yes) or not (No).

Initial Data Insights:
The customerID column was dropped as it is not needed for modeling.

The TotalCharges column, which was of an object data type, contained blank spaces that were replaced with 0.0 and then converted to a numerical (float) type.

The target variable Churn showed a class imbalance, with more customers not churning (No) than churning (Yes). The raw count was 5,174 for 'No' and 1,869 for 'Yes'.

Methodology
Exploratory Data Analysis (EDA): Histograms and box plots were used to understand the distribution of numerical features such as tenure, MonthlyCharges, and TotalCharges. A correlation heatmap was also generated to visualize the relationships between these features.

Preprocessing:

All categorical features were converted into numerical representations using LabelEncoder.

The dataset was split into training (80%) and testing (20%) sets.

SMOTE was applied to the training data to handle the class imbalance, creating a balanced dataset for training the models.

Model Training and Selection:

Three classification models were trained and validated using a 5-fold cross-validation approach on the SMOTE-resampled training data:

DecisionTreeClassifier

RandomForestClassifier

XGBClassifier

The Random Forest Classifier showed the highest average cross-validation accuracy (approximately 84%) and was selected as the final model.

Evaluation: The selected Random Forest model was evaluated on the unseen test data. The final accuracy was approximately 78%. A detailed classification report provided insights into the model's precision, recall, and F1-score for each class.

Serialization: The final trained model and the LabelEncoder objects were saved as pickle files (customer_churn_model.pkl and encoders.pkl) to allow for easy deployment and future use.

Files in this Repository
Customer_Churn_Prediction_using_ML.ipynb: The main Jupyter notebook containing all the code for data analysis, preprocessing, model training, and evaluation.

customer_churn_model.pkl: A serialized file of the trained Random Forest model.

encoders.pkl: A serialized file of the LabelEncoder objects used for preprocessing categorical features.

Prerequisites
To run this notebook, you will need a Python environment with the following libraries installed:

numpy

pandas

matplotlib

seaborn

scikit-learn

imblearn

xgboost
