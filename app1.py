import numpy as np
import joblib
import pandas as pd
# from flasgger import Swagger
import streamlit as st

from PIL import Image

# app=Flask(__name__)
# Swagger(app)
classifier=joblib.load("credit.pkl")


# @app.route('/')
def welcome():
    return "Welcome All"


# @app.route('/predict',methods=["Get"])
def predict_note_authentication(limit_bal,sex,education,marriage,age,pay_0,pay_2,pay_4,pay_5,pay_6,prev_payment):
    """Credit card Defaulter
    This is using docstrings for specifications.
    ---
    parameters:
      - name: limit_bal
        in: query
        type: number
        required: true
      - name: sex
        in: query
        type: number
        required: true
      - name: education
        in: query
        type: number
        required: true
      - name: marriage
        in: query
        type: number
        required: true
      - name: age
        in: query
        type: number
        required: true
      - name: pay_0
        in: query
        type: number
        required: true
      - name: pay_2
        in: query
        type: number
        required: true
      - name: pay_4
        in: query
        type: number
        required: true
      - name: pay_5
        in: query
        type: number
        required: true
      - name: pay_6
        in: query
        type: number
        required: true
      - name: prev_payment
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """

    prediction = classifier.predict([[limit_bal,sex,education,marriage,age,pay_0,pay_2,pay_4,pay_5,pay_6,prev_payment]])
    print(prediction)
    return prediction


def main():
    st.title("Credit Card Defaulter")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Credit Card Defaulter ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    limit_bal = st.text_input("limit_bal", "Type Here")
    sex = st.text_input("sex", "1-male,2-female")
    education = st.text_input("education", "Type Here")
    marriage = st.text_input("marriage", "Type Here")
    age = st.text_input("age", "Type Here")
    pay_0 = st.text_input("pay_0", "Type Here")
    pay_2 = st.text_input("pay_2", "Type Here")
    pay_4 = st.text_input("pay_4", "Type Here")
    pay_5 = st.text_input("pay_5", "Type Here")
    pay_6 = st.text_input("pay_6", "Type Here")
    prev_payment = st.text_input("prev_payment", "Type Here")


    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(limit_bal,sex,education,marriage,age,pay_0,pay_2,pay_4,pay_5,pay_6,prev_payment)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()