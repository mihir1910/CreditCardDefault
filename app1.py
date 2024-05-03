import numpy as np
import joblib
import pandas as pd
import streamlit as st

# Load the trained model
classifier = joblib.load("credit.pkl")


# Define the prediction function
def predict_credit_default(limit_bal, sex, education, marriage, age, pay_0, pay_2, pay_4, pay_5, pay_6, prev_payment):
    # Map categorical values to numeric representation
    sex_mapping = {"Male": 1, "Female": 2}
    education_mapping = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
    marriage_mapping = {"Married": 1, "Single": 2, "Others": 3}

    sex = sex_mapping.get(sex)
    education = education_mapping.get(education)
    marriage = marriage_mapping.get(marriage)

    prediction = classifier.predict(
        [[limit_bal, sex, education, marriage, age, pay_0, pay_2, pay_4, pay_5, pay_6, prev_payment]])
    return prediction[0]  # Extract the prediction value from the array


def main():
    st.title("Credit Card Defaulter Prediction")

    # Custom CSS for light blue background
    st.markdown(
        """
        <style>
        body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
        background:rgb(130, 106, 251)
        """,
        unsafe_allow_html=True
    )

    # HTML template for the header
    st.markdown(
        """
        <style>
        .header {
            background-color: #f63366;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .header h1 {
            color: #ffffff;
        }
        </style>
        """
        , unsafe_allow_html=True)

    # Header section
    st.markdown('<div class="header"><h1>Streamlit Credit Card Defaulter ML App</h1></div>', unsafe_allow_html=True)

    # Input fields
    limit_bal = st.text_input("Credit Limit", " ")
    sex = st.selectbox("Sex", ["Male", "Female"])
    education = st.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
    marriage = st.selectbox("Marital Status", ["Married", "Single", "Others"])
    age = st.text_input("Age", " ")

    # Payment status options dictionary
    pay_options = {
        "-2": "Paid on time",
        "1": "Delay by 1 month",
        "2": "Delay by 2 months",
        "3": "Delay by 3 months",
        "4": "Delay by 4 months",
        "5": "Delay by 5 months",
        "6": "Delay by 6 months",
        "7": "Delay by 7 months",
        "8": "Delay by 8 or more months"
    }

    # Payment status select boxes
    pay_6 = st.selectbox("Payment Status (April)", options=list(pay_options.keys()),
                         format_func=lambda x: pay_options[x])
    pay_5 = st.selectbox("Payment Status (May)", options=list(pay_options.keys()), format_func=lambda x: pay_options[x])
    pay_4 = st.selectbox("Payment Status (June)", options=list(pay_options.keys()),
                         format_func=lambda x: pay_options[x])
    pay_2 = st.selectbox("Payment Status (August)", options=list(pay_options.keys()),
                         format_func=lambda x: pay_options[x])
    pay_0 = st.selectbox("Payment Status (September)", options=list(pay_options.keys()),
                         format_func=lambda x: pay_options[x])

    prev_payment = st.text_input("Previous Payment Amount", " ")

    # Prediction
    if st.button("Predict"):
        try:
            result = predict_credit_default(float(limit_bal), sex, education, marriage, int(age), int(pay_0),
                                            int(pay_2), int(pay_4), int(pay_5), int(pay_6), float(prev_payment))
            if result == 0:
                st.success("The prediction is: Not a defaulter")
            elif result == 1:
                st.error("The prediction is: Defaulter")
            else:
                st.warning("Invalid prediction result")
        except ValueError:
            st.error("Please ensure all input fields are correctly filled.")

    # About section
    if st.button("About"):
        st.info("This app predicts credit card defaulters using machine learning.")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()
