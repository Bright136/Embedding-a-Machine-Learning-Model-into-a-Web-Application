import pandas as pd
import numpy as np
import pickle
from io import StringIO
from fastapi.responses import JSONResponse
# from cachetools import cached, TTLCache

# # Define the cache
# cache = TTLCache(maxsize=5, ttl=3600,)  # Cache with a maximum size of 1 and a TTL of 1 hour

# # # Load the model
# @cached(cache)
def load_pickle(filename):
    with open(filename, 'rb') as file:
        contents = pickle.load(file)
    return contents


def filetype_error(valid_formats):
    return JSONResponse(content={"error": f"Invalid file format. Must be one of: {', '.join(valid_formats)}"})


def feature_engineering(data):
    data['Insurance'] = data['Insurance'].astype(int).astype(str) # run function to create new features
    # create features 
    data['All-Product']  = data['Blood Work Result-4'] * data['Blood Work Result-1']* data['Blood Work Result-2']* data['Blood Work Result-3'] * data['Plasma Glucose']* data['Blood Pressure'] * data['Age']* data['Body Mass Index'] # Multiply all numerical features

    all_labels =['{0}-{1}'.format(i, i+500000000000) for i in range(0, round(2714705253292.0312),500000000000)]
    data['All-Product_range'] = pd.cut(data['All-Product'], bins=(range(0, 3500000000000, 500000000000)), right=False, labels=all_labels)
    
    age_labels =['{0}-{1}'.format(i, i+20) for i in range(0, 83,20)]
    data['Age Group'] = pd.cut(data['Age'], bins=(range(0, 120, 20)), right=False, labels=age_labels) # create categorical features for age

    labels =['{0}-{1}'.format(i, i+30) for i in range(0, round(67.1),30)]
    data['BMI_range'] = pd.cut(data['Body Mass Index'], bins=(range(0, 120, 30)), right=False, labels=labels) # create categorical features for bodey mass index

    bp_labels =['{0}-{1}'.format(i, i+50) for i in range(0, round(122),50)] 
    data['BP_range'] = pd.cut(data['Blood Pressure'], bins=(range(0, 200, 50)), right=False, labels=bp_labels) # create categorical features for blood pressure

    labels =['{0}-{1}'.format(i, i+7) for i in range(0, round(17),7)]
    data['PG_range'] = pd.cut(data['Plasma Glucose'], bins=(range(0, 28, 7)), right=False, labels=labels) # create categorical features for plasma glucose

    data.drop(columns=['Blood Pressure', 'Age', 'Body Mass Index','Plasma Glucose', 'All-Product', 'Blood Work Result-3', 'Blood Work Result-2'], inplace=True) # drop unused columns

    


def combine_cats_nums(transformed_data, full_pipeline):
    cat_features = full_pipeline.named_transformers_['categorical']['cat_encoder'].get_feature_names() # get the feature from the categorical transformer
    num_features = ['Blood Work Result-1', 'Blood Work Result-4']
    columns_ = np.concatenate([num_features, cat_features]) # concatenate numerical and categorical features
    prepared_data = pd.DataFrame(transformed_data, columns=columns_) # create a dataframe from the transformed data
    prepared_data = prepared_data.rename(columns={'x0_0':'Insurance_0', 'x0_1': 'Insurance_1'}) # rename columns
    

def make_prediction(data, transformer, model):
    new_columns = return_columns() 
    dict_new_old_cols = dict(zip(data.columns, new_columns))
    data = data.rename(columns=dict_new_old_cols)
    feature_engineering(data) # create new features
    transformed_data = transformer.transform(data) # transform the data using the transformer    
    combine_cats_nums(transformed_data, transformer)# create a dataframe from the transformed data 
    # make prediction
    label = model.predict(transformed_data) # make a prediction
    probs = model.predict_proba(transformed_data)
    return label, probs.max()



# function to create a new column 'Bmi'
def process_label(row):
    if row['Predicted Label'] == 1:
        return 'Sepsis status is Positive'
    elif row['Predicted Label'] == 0:
        return 'Sepsis status is Negative'

def return_columns():
    # create new columns
    new_columns =  ['Plasma Glucose','Blood Work Result-1', 'Blood Pressure', 
                    'Blood Work Result-2', 'Blood Work Result-3', 'Body Mass Index',
                    'Blood Work Result-4', 'Age', 'Insurance']
    return new_columns


def process_json_csv(contents, file_type, valid_formats):

    # Read the file contents as a byte string
    contents = contents.decode()  # Decode the byte string to a regular string
    new_columns = return_columns() # return new_columns
    if file_type == valid_formats[0]:
        data = pd.read_csv(StringIO(contents))
    # Process the uploaded file
    elif file_type == valid_formats[1]:
        data = pd.read_json(contents)
    data = data.drop(columns=['ID'])
    dict_new_old_cols = dict(zip(data.columns, new_columns)) # get dict of new and old cols
    print(f'INFO    {dict_new_old_cols}')
    data = data.rename(columns=dict_new_old_cols)
    return data

        
    