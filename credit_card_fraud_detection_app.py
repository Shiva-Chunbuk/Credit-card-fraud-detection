import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Title and description
st.title("Credit Card Fraud Detection")
st.write("This is a Streamlit app for credit card fraud detection.")

# Load the dataset
df = pd.read_csv('credit.csv')

# Sidebar
st.sidebar.title("Options")

# Data Exploration
st.sidebar.subheader("Data Exploration")
if st.sidebar.checkbox("Show dataset summary"):
    st.subheader("Dataset Summary")
    st.write(df.head())
    st.write(df.info())
    st.write(df.describe())

# Univariate Analysis
st.sidebar.subheader("Univariate Analysis")
cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'default.payment.next.month']
if st.sidebar.checkbox("Show univariate analysis plots"):
    fig, ax = plt.subplots(1, 4, figsize=(25, 5))
    for cols, subplots in zip(cat_cols, ax.flatten()):
        sns.countplot(x=df[cols], ax=subplots)
    st.pyplot(fig)

# Visualizing the imbalance
# st.sidebar.subheader("Class Imbalance")
# if st.sidebar.checkbox("Show class imbalance"):
#     yes_percentage = (((df['default.payment.next.month'] == 1).sum()) / len(df['default.payment.next.month'])) * 100
#     no_percentage = (((df['default.payment.next.month'] == 0).sum()) / len(df['default.payment.next.month'])) * 100
#     x = [yes_percentage, no_percentage]
#     plt.pie(x, labels=['Yes', 'No'], colors=['red', 'white'], radius=2, autopct='%1.0f%%')
#     plt.title('default.payment.next.month')
#     st.pyplot()
# Visualizing the imbalance
st.sidebar.subheader("Class Imbalance")
if st.sidebar.checkbox("Show class imbalance"):
    yes_percentage = (((df['default.payment.next.month'] == 1).sum()) / len(df['default.payment.next.month'])) * 100
    no_percentage = (((df['default.payment.next.month'] == 0).sum()) / len(df['default.payment.next.month'])) * 100
    x = [yes_percentage, no_percentage]
    fig, ax = plt.subplots()  # Create a Matplotlib figure
    ax.pie(x, labels=['Yes', 'No'], colors=['red', 'white'], radius=2, autopct='%1.0f%%')
    ax.set_title('default.payment.next.month')
    st.pyplot(fig)  # Pass the figure to st.pyplot

# Machine Learning Model
st.sidebar.subheader("Machine Learning Model")
if st.sidebar.checkbox("Train and evaluate a RandomForestClassifier"):
    st.subheader("Machine Learning Model - RandomForestClassifier")

    # Split the data into features and target
    X = df.drop('default.payment.next.month', axis=1)
    y = df['default.payment.next.month']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # Create and train the model
    rfc = RandomForestClassifier()
    model = rfc.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Display evaluation metrics
    st.write("Classification Report:")
    st.text(classification_report(y_test, predictions))
    st.write("Confusion Matrix:")
    st.text(confusion_matrix(y_test, predictions))
    st.write("Accuracy Score:")
    st.text(accuracy_score(y_test, predictions))
