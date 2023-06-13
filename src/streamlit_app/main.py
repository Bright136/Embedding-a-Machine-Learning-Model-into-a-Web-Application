import streamlit as st
import requests

# Set API Endpoint
URL = 'https://radiant-lowlands-86946.herokuapp.com//predict'


# Create a function to make prediction
def make_prediction(pg: float, bwr1: float, bp : float, bwr2: float, bwr3: float, bmi: float, bwr4: float, age: int, insurance: bool):

    parameters={
        'plasma_glucose':pg,
        'blood_work_result_1':bwr1,
        'blood_pressure':bp,
        'blood_work_result_2':bwr2,
        'blood_work_result_3':bwr3,
        'body_mass_index':bmi,
        'blood_work_result_4':bwr4,
        'age':int(age),
        'insurance':bool(insurance)}
    response = requests.post(URL, params=parameters)
    print(type(age))
    response_text =  response.json()
    sepsis_status = response_text['results'][0]['0']['output']['Predicted Label']
    return sepsis_status














