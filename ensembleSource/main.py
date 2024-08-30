import streamlit as st
from voting_app import voting_app
from bagging_app import bagging_app
from boosting_regressor_app import boosting_regressor_app
from boosting_classifier_app import boosting_classifier_app

# from stacking_app import stacking_app

st.set_page_config(page_title="Ensemble Learning Algorithms", layout='centered')

st.sidebar.title("Ensemble Learning Algorithms")
algorithm = st.sidebar.selectbox(
    "Choose an Ensemble Method", 
    ["Voting Classifier", "Bagging", "Gradient Boosting Regressor", "Gradient Boosting Classifier"]
)

if algorithm == "Voting Classifier":
    voting_app()
elif algorithm == "Bagging":
    bagging_app()  # Call the Bagging app function
elif algorithm == "Gradient Boosting Regressor":
    boosting_regressor_app()  
elif algorithm == "Gradient Boosting Classifier":
    boosting_classifier_app()
    st.write("Select a valid ensemble method from the sidebar.")


