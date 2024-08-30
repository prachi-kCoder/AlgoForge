import streamlit as st
from voting_app import voting_app
# from boosting_app import boosting_app
# from stacking_app import stacking_app

st.set_page_config(page_title="Ensemble Learning Algorithms", layout='centered')
st.sidebar.title("Ensemble Learning Algorithms")
algorithm = st.sidebar.selectbox(
    "Choose an Ensemble Method", 
    ["Voting Classifier", "Boosting", "Stacking"]
)

if algorithm == "Voting Classifier":
    voting_app()
elif algorithm == "Boosting":
    st.write("Boosting algorithm implementation coming soon!")  # Placeholder text
    # boosting_app()  # Uncomment this when the boosting_app function is implemented
elif algorithm == "Stacking":
    st.write("Stacking algorithm implementation coming soon!")  # Placeholder text
    # stacking_app()  # Uncomment this when the stacking_app function is implemented
else:
    st.write("Select a valid ensemble method from the sidebar.")



