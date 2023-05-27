import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from assets.utils import load_pickle, get_label, return_columns
from assets.module import Inputs
from io import StringIO
import pandas as pd
from typing import List

# create an instance of FastApi
app = FastAPI(debug=True)
model = load_pickle('src/app/assets/model.pkl') # load the model
transformer = load_pickle('src/app/assets/full_pipeline.pkl') # load the pipeline
properties = load_pickle('src/app/assets/properties.pkl') # load the other properties saved from the modeling 

# Configure static and template file
app.mount("/static", StaticFiles(directory="src/app/static"), name="static") # mount statis files
templates = Jinja2Templates(directory="src/app/templates") # mount templates for html

# set display for root
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {'request': request})


# check the health status of the api
@app.get("/health")
def check_health():
    return {"status": "ok"}

# get the model infomation
@app.get('/model-info')
async def model_info():
    model_params = model.get_params()
    features = properties['train features']
    print(features)
    return {'model info': {
            'model parameters': model_params,
            'train feature': features}
            }

# make a prediction with the api
@app.get('/predict')
async def predict(plasma_glucose: float, blood_work_result_1: float, 
                  blood_pressure: float, blood_work_result_2: float, 
                  blood_work_result_3: float, body_mass_index: float, 
                  blood_work_result_4: float, age: int, insurance: bool):
    
    # create dataframe from inputs 
    data = pd.DataFrame({'Plasma Glucose': [plasma_glucose], 'Blood Work Result-1':	[blood_work_result_1],
                         'Blood Pressure': [blood_pressure], 'Blood Work Result-2': [blood_work_result_2],
                        'Blood Work Result-3': [blood_work_result_3], 'Body Mass Index': [body_mass_index],
                        'Blood Work Result-4':	[blood_work_result_4], 'Age': [age], 'Insurance': [insurance]})

    # set a copy on the dataframe
    data_copy = data.copy()
    # get the labels
    label = get_label(data, transformer,  model)
    # get the labels from making a prediction
    data_copy['Predicted Label'] = label[0]
    # convert dataframe to dicionary
    data_dict =  data_copy.to_dict('index')

    return {'outputs': data_dict}


# make a batch prediction
@app.post('/predict_batch')
async def predict_batch(inputs: Inputs):
    # create dataframe from inputs
    data = pd.DataFrame(inputs.return_dict_inputs())
    # rename the columsn in teh  dataframe to columns the transformer understands
    data = data.rename(columns={'plasma_glucose': 'Plasma Glucose','blood_work_result_1':'Blood Work Result-1', 
                            'blood_pressure':'Blood Pressure', 'blood_work_result_2':'Blood Work Result-2', 
                            'blood_work_result_3':'Blood Work Result-3', 'body_mass_index':'Body Mass Index',
                            'blood_work_result_4':'Blood Work Result-4', 'age': 'Age', 'insurance':'Insurance'})
    data_copy = data.copy() # set a copy on the data 
    label = get_label(data, transformer,  model) # get the labels
    data_copy['Predicted Label'] = label
    data_dict =  data_copy.to_dict('index') # convert dataframe to dicionary
    return {'outputs': data_dict}


if __name__=='__main__':
    uvicorn.run('app:app', reload=True)

