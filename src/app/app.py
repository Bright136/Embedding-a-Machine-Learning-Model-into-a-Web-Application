import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.utils import load_pickle, make_prediction, process_label, process_json_csv, output_batch, return_columns
from src.module import Inputs
import pandas as pd
import numpy as np
from typing import List


# Create an instance of FastAPI
app = FastAPI(debug=True)

DIRPATH = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(DIRPATH, '..', 'assets', 'ml_components', 'model-1.pkl')
transformer_path = os.path.join(DIRPATH, '..', 'assets', 'ml_components', 'preprocessor.pkl')
properties_path = os.path.join(DIRPATH, '..', 'assets', 'ml_components', 'properties.pkl')


# Load the trained model, pipeline, and other properties
model = load_pickle(model_path)
transformer = load_pickle(transformer_path)
properties = load_pickle(properties_path)

# Configure static and template files
app.mount("/static", StaticFiles(directory="src/app/static"), name="static") # Mount static files
templates = Jinja2Templates(directory="src/app/templates") # Mount templates for HTML

# Root endpoint to serve index.html template
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {'request': request})

# Health check endpoint
@app.get("/health")
def check_health():
    return {"status": "ok"}

# Model information endpoint
@app.post('/model-info')
async def model_info():
    model_name = model.__class__.__name__
    model_params = model.get_params()
    features = properties['train features']
    print(features)
    model_information =  {'model info': {
            'model name ': model_name,
            'model parameters': model_params,
            'train feature': features}
            }
    return model_information
 

# Prediction endpoint
@app.post('/predict')
async def predict(plasma_glucose: float, blood_work_result_1: float, 
                  blood_pressure: float, blood_work_result_2: float, 
                  blood_work_result_3: float, body_mass_index: float, 
                  blood_work_result_4: float, age: int, insurance: bool):
    
    # Create a dataframe from inputs 
    data = pd.DataFrame([[plasma_glucose,blood_work_result_1,blood_pressure,
                           blood_work_result_2,blood_work_result_3,body_mass_index, 
                           blood_work_result_4, age,insurance]], columns=return_columns())

    data_copy = data.copy() # Create a copy of the dataframe
    label, prob = make_prediction(data, transformer, model) # Get the labels
    data_copy['Predicted Label'] = label[0] # Get the labels from making a prediction
    data_copy['Predicted Label'] = data_copy.apply(process_label, axis=1)
    inputs = data.to_dict('index') # Convert dataframe to dictionary
    outputs = data_copy[['Predicted Label']].to_dict('index')    
    response = {'inputs': inputs,
                'outputs': outputs}
    return response


# Batch prediction endpoint
@app.post('/predict-batch')
async def predict_batch(inputs: Inputs):
    # Create a dataframe from inputs
    data = pd.DataFrame(inputs.return_dict_inputs())
    data_copy = data.copy() # Create a copy of the data
    labels, probs = make_prediction(data, transformer, model) # Get the labels
    data_labels = pd.DataFrame(labels, columns=['Predicted Label'])
    data_labels['Predicted Label'] = data_labels.apply(process_label, axis=1)

    response = output_batch(data, data_labels)

    return response



# Upload data endpoint
@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    file_type = file.content_type
    print(f'INFO    {file_type}')

    valid_formats = ['text/csv', 'application/json']
    
    if file_type not in valid_formats:
        return JSONResponse(content={"error": f"Invalid file format. Must be one of: {', '.join(valid_formats)}"})
    
    else:
        contents = await file.read()
        data= process_json_csv(contents=contents,file_type=file_type, valid_formats=valid_formats)  
        data_copy = data.copy() # Create a copy of the data
        labels, probs = make_prediction(data, transformer, model) # Get the labels
        data_copy['Predicted Label'] = labels# Create the predicted label column
        data_copy['Predicted Label'] = data_copy.apply(process_label, axis=1)
        data_dict = data_copy.to_dict('index') # Convert data to a dictionary
        # print(data_dict.index)

    return {'outputs': data_dict}

# Run the FastAPI application
if __name__ == '__main__':
    uvicorn.run('app:app', reload=True)
