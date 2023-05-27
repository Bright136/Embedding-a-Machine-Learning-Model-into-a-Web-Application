from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle

def load_pickle(filename):
    with open(filename, 'rb') as file:
        contents = pickle.load(file)
    return contents




# function to create a new column 'Bmi'
def create_bmi_range(row):
    if (row['Body Mass Index'] <= 18.5):
        return 'Under Weight'
    elif (row['Body Mass Index'] > 18.5) and (row['Body Mass Index'] <= 24.9):
        return 'Healthy Weight'
    elif (row['Body Mass Index'] > 24.9) and (row['Body Mass Index'] <= 29.9):
        return 'Over Weight'
    elif (row['Body Mass Index'] > 29.9) and (row['Body Mass Index'] < 40):
        return 'Obesity'
    elif row['Body Mass Index'] >= 40:
        return 'Severe Obesity'


# create a function to create a new column called blood pressure ranges
def blood_pressure_ranges(row):
    if row['Blood Pressure'] < 80:
        return 'normal'
    elif row['Blood Pressure'] >= 80 and row['Blood Pressure'] <= 89:
        return 'elevated'
    elif row['Blood Pressure'] >= 90:
        return 'high'


def feature_engineering(data):
    # create features the BMI_Range and BP_Range for x_train
    data['BMI_Range'] = data.apply(create_bmi_range, axis=1)
    data['BP_range'] = data.apply(blood_pressure_ranges, axis=1)
    # create age group
    age_labels =['{0}-{1}'.format(i, i+20) for i in range(0, 81,20)]
    data['Age Group'] = pd.cut(data['Age'], bins=(range(0, 120, 20)), right=False, labels=age_labels)

    data.drop(columns=['Blood Pressure', 'Age', 'Body Mass Index'], inplace=True)
    


def combine_cats_nums(transformed_data, full_pipeline):
    cat_features = full_pipeline.named_transformers_['categorical']['cat_encoder'].get_feature_names()
    num_features = ['Plasma Glucose', 'Blood Work Result-1', 'Blood Work Result-2', 'Blood Work Result-3', 'Blood Work Result-4']
    columns_ = np.concatenate([num_features, cat_features])
    prepared_data = pd.DataFrame(transformed_data, columns=columns_)
    prepared_data = prepared_data.rename(columns={'x0_0':'Insurance_0', 'x0_1': 'Insurance_1'})
    

def get_label(data, transformer, model):

    # run function to create new features
    data['Insurance'] = data['Insurance'].astype(int).astype(str)
    # create new features
    feature_engineering(data)
    # transform the data using the transformer
    transformed_data = transformer.transform(data)

    # get and concatenate the numerical and categorical features
    # create a dataframe from the transformed data 
    combine_cats_nums(transformed_data, transformer)

    # make prediction
    label = model.predict(transformed_data)
    return label
    
