# 🚀Embedding-a-Machine-Learning-Model-into-a-Web-Application 🚀


[![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![fastapi](https://img.shields.io/badge/FastAPI-009485?style=for-the-badge&logo=fastapi&logoColor=white)](https://img.shields.io/badge/FastAPI-3776AB?style=for-the-badge&logo=fastapi&logoColor=white)
![Issues](https://img.shields.io/github/issues/eaedk/streamlit-iris-app?style=for-the-badge&logo=appveyor)
![PR](https://img.shields.io/github/issues-pr/eaedk/streamlit-iris-app?style=for-the-badge&logo=appveyor)
[![Open Source Love png1](https://badges.frapsoft.com/os/v1/open-source.png?v=103)](https://github.com/ellerbrock/open-source-badges/)


## Screenshots of the App
<div align='center'> 
    <img src=""/>

</div>



## Project Description 



## Table of Contents
1. [Overview Of the Project](#overview)

  - [Description of dataset](#dataset)

2. [Application / Deployed Links](#application)

3. [Technology Stack](#technology)

4. [Deliverables](#deliverables)

5. [Installation](#installation)

6. [Execution](#execution)

7. [App Usage](#usage)

8. [API Endpoints](#api-endpoints)

9. [Collaborators](#collaborators)

10. [Contributing Instructions](#instructions)

11. [Contact Information](#ontact)


## 1. Overview Of the Project <a name="overview"></a>

- The sepsis prediction project revolves around a machine learning model designed to accurately predict sepsis in intensive care unit (ICU) patients. The model has undergone rigorous training and evaluation to ensure its effectiveness in identifying patients at risk of sepsis.

- The project provides a comprehensive solution, including a well-documented FastAPI hosted on a platform like the Hugging Face Model Hub and Heroku. This API allows seamless integration of the sepsis prediction model into existing healthcare systems, providing healthcare professionals with valuable insights to improve patient care.

- To simplify deployment and usage, the project includes a Dockerfile that streamlines the setup process and ensures the necessary dependencies are installed. This enables easy deployment of the sepsis prediction model in various environments, both local and cloud-based.

- Detailed documentation and practical examples are provided to guide users in effectively utilizing the sepsis prediction model. The documentation covers installation instructions, API usage guidelines, and highlights the potential applications of the model in real-world healthcare scenarios, empowering healthcare providers to make informed decisions and enhance patient outcomes.

### i. Description of dataset <a name="dataset"></a>

Here is the provided information converted into an HTML table:

<table>
  <tr>
    <th>Column Name</th>
    <th>Attribute/Target</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>ID</td>
    <td>N/A</td>
    <td>Unique number to represent patient ID</td>
  </tr>
  <tr>
    <td>PRG</td>
    <td>Attribute1</td>
    <td>Plasma glucose</td>
  </tr>
  <tr>
    <td>PL</td>
    <td>Attribute 2</td>
    <td>Blood Work Result-1 (mu U/ml)</td>
  </tr>
  <tr>
    <td>PR</td>
    <td>Attribute 3</td>
    <td>Blood Pressure (mm Hg)</td>
  </tr>
  <tr>
    <td>SK</td>
    <td>Attribute 4</td>
    <td>Blood Work Result-2 (mm)</td>
  </tr>
  <tr>
    <td>TS</td>
    <td>Attribute 5</td>
    <td>Blood Work Result-3 (mu U/ml)</td>
  </tr>
  <tr>
    <td>M11</td>
    <td>Attribute 6</td>
    <td>Body mass index (weight in kg/(height in m)^2)</td>
  </tr>
  <tr>
    <td>BD2</td>
    <td>Attribute 7</td>
    <td>Blood Work Result-4 (mu U/ml)</td>
  </tr>
  <tr>
    <td>Age</td>
    <td>Attribute 8</td>
    <td>Patients age (years)</td>
  </tr>
  <tr>
    <td>Insurance</td>
    <td>N/A</td>
    <td>If a patient holds a valid insurance card</td>
  </tr>
  <tr>
    <td>Sepssis</td>
    <td>Target</td>
    <td>Positive: if a patient in ICU will develop sepsis, and Negative: otherwise</td>
  </tr>
</table>

## 2. Application / Deployed Links <a name="application"></a>
<table>
  <tr>
    <th>API</th>
    <th>Deployed links</th>
  </tr>
  <tr>
    <td>FastApi</td>
    <td><a href="https://bright1-sepsis-prediction-api.hf.space/docs">Sepsis Prediction API-huggingface</a></td>
  </tr>
  <tr>
    <td>FastApi</td>
    <td><a href="https://radiant-lowlands-86946.herokuapp.com/docs">Sepsis Prediction API-heroku</a></td>
  </tr>

</table>

<table>
  <tr>
    <th>App</th>
    <th>Deployed links</th>
  </tr>
  <tr>
    <td>App</td>
    <td><a href="">App</a></td>
  </tr>

</table>

## 3. Technology Stack <a name="technology"></a>
 
<table>
  <tr>
    <th>Technology</th>
    <th>Version</th>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.9</td>
  </tr>
  <tr>
    <td>FastAPI</td>
    <td>0.95.2</td>
  </tr>
  <tr>
    <td>Uvicorn</td>
    <td>0.22.0</td>
  </tr>
    <tr>
    <td>Scikit-learn</td>
    <td>0.24.1</td>
  </tr>
  </tr>
    <tr>
    <td>Pandas</td>
    <td>1.2.4</td>
  </tr>
  </tr>
    <tr>
    <td>Jinja2</td>
    <td>3.1.2</td>
  </tr>
  
</table>

## 4. Deliverables <a name="deliverables"></a>
1. A jupyter notebook for training a classification model
2. A classification Model
3. An API App built with FastApi
4. A Dockerfile for easy deployment 



## 5. Installation <a name="installation"></a>
Clone the repository to your local machine:


        git clone https://github.com/Bright136/Embedding-a-Machine-Learning-Model-into-a-Web-Application.git

Navigate to the project directory:

        cd Embedding-a-Machine-Learning-Model-into-a-Web-Application
Create a new virtual environment and activate the virtual:

- Windows:

        python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

- Linux & MacOs:

        python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt



## 6. Execution <a name='execution'></a>
1. Notebooks

To run any the notebooks:
- Navigate the project folder on anaconda terminal
- Run the command 'jupyter notebook'
- Navigate to the notebook 'Sepssis_prediction_with_ml.ipynb'
- Run cells in the notebook


2. API

To execute the API, follow these steps:
After all requirement have been install

At the root of your repository in your terminal
`root :: Embedding-a-Machine-Learning-Model-into-a-Web-Application> ...`
run the command:


            uvicorn src.app.app:app --reload 

OR

            python src/app/app.py

Open your browser and go to http://127.0.0.1:8000/docs to access the API documentation

## 7. Endpoints <a name="api-endpoints"></a>

1. **/**: This Endpoint display a welcome message-” Welcome to the Sepsis API...”.
2. **/health**: Checks status of the API
3. **model-info**: Returns model information
4. **/predict**: Recieve inouts and retuens a single prediction.
5. **/predict-batch**: Receives multiples inputs and returns multiple predictions
5. **/upload-data**: Receives JSON or CSV file, process it and returns predictions

## 8. App Usage <a name="usage"></a>
To test the various endpoints of the API using the provided documentation, follow these steps:

1. Start by accessing the API documentation, which provides detailed information about the available endpoints and their functionalities.

2. Locate the section that describes the input fields and parameters required for each endpoint. It will specify the expected data format, such as JSON or form data, and the necessary input fields.


4. Enter the required input data into the corresponding input fields or parameters as specified in the documentation.

5. Send the request by clicking the "Execute" button or using the appropriate method in your chosen tool. The API will process the request and generate the output based on the provided inputs.

6. Retrieve the response from the API, which will contain the generated output. This output may include predictions, probability scores, or any other relevant information related to sepsis prediction.

7. Repeat the process to test different endpoints or vary the input data to explore the capabilities of the API. Make sure to follow the documentation's guidelines for each endpoint to ensure accurate results.


## 9. Collaborators <a name="collaborators"></a>


## 10. Contributing Instructions <a name="instructions"></a>
To contribute to this project, follow these guidelines:

- Fork the repository.
- Create a new branch: git checkout -b my-new-feature
- Make your changes and commit them: git commit -am 'Add some feature'
- Push to the branch: git push origin my-new-feature
- Create a new pull request

## 11. Contact Information <a name="contact"></a>

<table>
  <tr>
    <th>Name</th>
    <th>Twitter</th>
    <th>LinkedIn</th>
    <th>GitHub</th>
    <th>Hugging Face</th>
  </tr>
  <tr>
    <td>Bright Eshun</td>
    <td><a href="https://twitter.com/bright_eshun_">@bright_eshun_</a></td>
    <td><a href="https://www.linkedin.com/in/bright-eshun-9a8a51100/">@brighteshun</a></td>
    <td><a href="https://github.com/Bright136">@bright136</a></td>
    <td><a href="https://huggingface.co/bright1">@bright1</a></td>
  </tr>
</table>
