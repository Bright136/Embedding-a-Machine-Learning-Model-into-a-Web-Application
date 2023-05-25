import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from assets.utils import feature_engineering, load_pickle, combine_cats_nums
import pandas as pd

# create an instance of FastApi
app = FastAPI(debug=True)

# load the model
model = load_pickle('src/app/assets/model.pkl')
# load the pipeline
transformer = load_pickle('src/app/assets/full_pipeline.pkl')

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

    data_copy = data.copy()
    # run function to create new features
    data['Insurance'] = data['Insurance'].astype(int).astype(str)
    # create the new feature just like in training sessionn
    feature_engineering(data)
    # transform the data using the transformer
    transformed_data = transformer.transform(data)

    # get and concatenate the numerical and categorical features
    # create a dataframe from the transformed data 
    combine_cats_nums(transformed_data, transformer)

    # make prediction
    label = model.predict(transformed_data)
    print(label)
    data_copy['Label'] = label[0]
    print(f'INFO:   {data.to_markdown()}')
    
    # # convert dataframe to dicionary
    data_dict =  data_copy.to_dict('index')

    return {'outputs': data_dict}


if __name__=='__main__':
    uvicorn.run('app:app', reload=True)

