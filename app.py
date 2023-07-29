from model import check_similarity

import streamlit as st
st.title("Similarity check Between 2 texts")
input1 = st.text_input("text1: ")
input2 = st.text_input("text2: ")

if st.button('check'):
    if input1 and input2:
        score = check_similarity(input1,input2)
        # st.write(score)
        st.success(f"Similarity score: {score}")
        # st.info(score)