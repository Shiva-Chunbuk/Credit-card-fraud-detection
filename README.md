**# Credit-card-fraud-detection**
http://localhost:8501/

**#Description:**
The "Credit Card Fraud Detection Streamlit App" is a web-based tool designed to help users explore and analyze a dataset related to credit card transactions. This interactive application provides various functionalities to investigate and understand the data, visualize class imbalances, and train/evaluate a machine learning model for predicting credit card fraud.

**#Key Features**
**Dataset Summary:** Users can view a summary of the credit card transaction dataset, including the first few rows, data types, and basic statistics. This information provides an initial understanding of the data.

**Univariate Analysis:** The app offers the capability to perform univariate analysis on selected categorical columns, such as 'SEX,' 'EDUCATION,' 'MARRIAGE,' and 'default.payment.next.month.' Users can explore the distribution of values within these columns using count plots.

**Class Imbalance Visualization:** The app provides an option to visualize the class imbalance in the target variable ('default.payment.next.month'). Users can see the percentage distribution of "Yes" and "No" values using a pie chart.

**Machine Learning Model:** Users can train and evaluate a RandomForestClassifier machine learning model using the dataset. The app splits the data into training and testing sets, fits the model, makes predictions, and displays essential evaluation metrics, including the classification report, confusion matrix, and accuracy score.


**How to Use:**

**1. Open the Streamlit app in your web browser.
2.Use the sidebar to navigate through different options:**
Explore dataset summary.
Perform univariate analysis on categorical columns.
Visualize class imbalance.
Train and evaluate a machine learning model.
**Interact with the app to gain insights into the dataset and assess the performance of the machine learning model.**
