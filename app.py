import streamlit as st
from predict import predict_spam

st.title("Smart Spam Message Detector")

st.write("Enter a message to check if it is spam or not")

message = st.text_area("Enter Message")

if st.button("Detect Spam"):

    if message.strip() == "":
        st.warning("Please enter a message")

    else:
        label, prob = predict_spam(message)

        if label == "Spam":
            st.error("⚠ This message is SPAM")
        else:
            st.success("✅ This message is NOT spam")

        st.write("Confidence Score:", round(prob,2))