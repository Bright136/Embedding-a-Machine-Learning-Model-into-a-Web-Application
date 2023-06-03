import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.utils import load_pickle, get_label, return_columns, process_csv, process_json, process_label
from src.module import Inputs
import pandas as pd
from typing import List


# Create an instance of FastAPI
app = FastAPI(debug=True)

DIRPATH = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(DIRPATH, '..', 'assets', 'ml_components', 'model.pkl')
transformer_path = os.path.join(DIRPATH, '..', 'assets', 'ml_components', 'full_pipeline.pkl')
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
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {'request': request})

# Health check endpoint
@app.get("/health")
def check_health():
    return {"status": "ok"}

# Model information endpoint
@app.get('/model-info')
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
@app.get('/predict')
async def predict(plasma_glucose: float, blood_work_result_1: float, 
                  blood_pressure: float, blood_work_result_2: float, 
                  blood_work_result_3: float, body_mass_index: float, 
                  blood_work_result_4: float, age: int, insurance: bool):
    
    # Create a dataframe from inputs 
    data = pd.DataFrame({'Plasma Glucose': [plasma_glucose], 'Blood Work Result-1': [blood_work_result_1],
                         'Blood Pressure': [blood_pressure], 'Blood Work Result-2': [blood_work_result_2],
                         'Blood Work Result-3': [blood_work_result_3], 'Body Mass Index': [body_mass_index],
                         'Blood Work Result-4': [blood_work_result_4], 'Age': [age], 'Insurance': [insurance]})
    
    data_copy = data.copy() # Create a copy of the dataframe
    label, prob = get_label(data, transformer, model) # Get the labels
    data_copy['Predicted Label'] = label[0] # Get the labels from making a prediction
    data_copy['Predicted Label'] = data_copy.apply(process_label, axis=1)
    inputs = data.to_dict('index') # Convert dataframe to dictionary
    outputs = data_copy[['Predicted Label']].to_dict('index')    
    response = {'inputs': inputs,
                'outputs': outputs}
    return response


# Batch prediction endpoint
@app.post('/predict_batch')
async def predict_batch(inputs: Inputs):
    # Create a dataframe from inputs
    data = pd.DataFrame(inputs.return_dict_inputs())
    data_copy = data.copy() # Create a copy of the data
    labels, probs = get_label(data, transformer, model) # Get the labels
    data_copy['Predicted Label'] = labels
    data_dict = data_copy.to_dict('index') # Convert the data to a dictionary
    return {'outputs': data_dict}



# Upload data endpoint
@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    file_type = file.content_type
    print(f'INFO    {file_type}')
    valid_formats = ['text/csv', 'application/json']
    
    if file_type not in valid_formats:
        return JSONResponse(content={"error": f"Invalid file format. Must be one of: {', '.join(valid_formats)}"})
    
    elif file_type == valid_formats[0]:
        contents = await file.read()  # Read the file contents as a byte string
        data = process_csv(contents=contents)
        
    elif file_type == valid_formats[1]:
        contents = await file.read()  # Read the file contents as a byte string
        data = process_json(contents=contents)
        
    data_copy = data.copy() # Create a copy of the data
    labels, probs = get_label(data, transformer, model) # Get the labels
    data_copy['Predicted Label'] = labels# Create the predicted label column
    data_dict = data_copy.to_dict('index') # Convert data to a dictionary
    
    return {'outputs': data_dict}

# Run the FastAPI application
if __name__ == '__main__':
    uvicorn.run('app:app', reload=True)
