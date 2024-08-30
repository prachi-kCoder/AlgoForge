import streamlit as st
from LOR import logistic_regression_app
from random_forest import random_forest_app
from DecisionTree import decision_tree_app
from KNN import knn_app

st.set_page_config(page_title="Classifier Selection", layout='centered')  # Ensure this is at the top

st.sidebar.title("Classifier Selection")
algorithm = st.sidebar.selectbox("Choose a Classifier", ["Logistic Regression", "Random Forest", "Decision Tree","KNN"])

if algorithm == "Logistic Regression":
    logistic_regression_app()
elif algorithm == "Random Forest":
    random_forest_app()
elif algorithm == "Decision Tree":
    decision_tree_app()
elif algorithm == "KNN":
    knn_app()
