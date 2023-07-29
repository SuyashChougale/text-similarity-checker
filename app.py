# Import the check_similarity function from the model.py file
from model import check_similarity

# Import the Streamlit library
import streamlit as st

# Set the title of the app
st.title("Similarity check Between 2 texts")

# Create text input boxes for user input
input1 = st.text_input("text1: ")
input2 = st.text_input("text2: ")

# Create a button to trigger the similarity check
if st.button('check'):
    # Check if both text inputs are not empty
    if input1 and input2:
        # Call the check_similarity function from the model.py file
        score = check_similarity(input1, input2)

        # Display the similarity score
        st.success(f"Similarity score: {score}")
