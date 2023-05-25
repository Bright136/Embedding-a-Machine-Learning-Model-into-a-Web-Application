import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from assets.utils import feature_engineering, load_pickle
import pandas as pd
# import tabulate

app = FastAPI(debug=True)




@app.get('/predict')
async def predict(plasma_glucose: float, blood_work_result_1: float, 
                  blood_pressure: float, blood_work_result_2: float, 
                  blood_work_result_3: float, body_mass_index: float, 
                  blood_work_result_4: float, age: int, insurance: bool):
    
    # create dataframe from inputs 
    data = pd.DataFrame({'Plasma Glucose': [plasma_glucose], 'Blood Work Result-1':	blood_work_result_1,
                         'Blood Pressure': blood_pressure, 'Blood Work Result-2': blood_work_result_2,
                        'Blood Work Result-3': blood_work_result_3, 'Body Mass Index':	body_mass_index,
                        'Blood Work Result-4':	blood_work_result_4, 'Age': age, 'Insurance': insurance})

    # run function to create new features
    prepared_data = feature_engineering(data)
    print(f'INFO:   {prepared_data.to_markdown()}')

    # convert dataframe to dicionary
    data_dict =  data.to_dict('index')

    return {'outputs': data_dict}









if __name__=='__main__':
    uvicorn.run('app:app', reload=True)

