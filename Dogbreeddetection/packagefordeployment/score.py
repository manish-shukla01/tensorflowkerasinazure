#Example: scikit-learn and Swagger
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array,load_img,array_to_img
from tensorflow.keras.applications.inception_v3 import InceptionV3
import PIL
import matplotlib.pyplot as plt
from azureml.core import Workspace, Model
import json
from azureml.core.webservice import AciWebservice, Webservice, LocalWebservice
from azureml.core.authentication import ServicePrincipalAuthentication
import requests
import os
from azureml.contrib.services.aml_response import AMLResponse
sp = ServicePrincipalAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47", # tenantID
                                    service_principal_id="2faad9e6-8bf9-4e7d-a732-70e3daa5ffd5", # clientId
                                    service_principal_password="#############") # clientSecret

def init():
    global model
    global deduped
    # AZUREML_MODEL_DIR is an environment variable created during deployment. Join this path with the filename of the model file.
    # It holds the path to the directory that contains the deployed model (./azureml-models/$MODEL_NAME/$VERSION).
    # If there are multiple models, this value is the path to the directory containing all deployed models (./azureml-models).
    # Alternatively: model_path = Model.get_model_path('sklearn_mnist')
    #model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'sklearn_mnist_model.pkl')
    # Deserialize the model file back into a sklearn model
    workspace = Workspace('bd04922c-a444-43dc-892f-74d5090f8a9a', 'mlplayarearg', 'testdeployment', auth=sp)
    model_path = Model.get_model_path(model_name='dogbreedclassifiernew',_workspace=workspace)
    model = tf.keras.models.load_model(model_path)
    mappingdata = pd.read_csv('./packagefordeployment/mydata.csv')
    mappingbreedtonum = mappingdata[['breed','breednum']]
    deduped = mappingbreedtonum.drop_duplicates()
    deduped.set_index('breednum', inplace=True)


def run(data):
    try:
        data1 = json.loads(data)
        allfiles = data1['filelocations']
        inputdf = pd.DataFrame(allfiles, columns = ['FileNames'])
        #print(inputdf)
        #print(allfiles)
        imgarray = np.zeros((inputdf.shape[0],299,299,3))
        counter = 0
        for x in allfiles:
            response = requests.get(x)
            file = open("sample_image.jpg", "wb")
            file.write(response.content)
            file.close()
            oneimage = img_to_array(load_img('sample_image.jpg', target_size=(299,299,3)))
            imgarray[counter] = oneimage
            if os.path.exists("'sample_image.jpg'"): 
                os.remove("'sample_image.jpg'")
            counter = counter + 1
        imgarray = imgarray/255
        #print(imgarray.shape)
        prediction =  model.predict(imgarray)
        breednum = np.argmax(prediction,axis=1)
        outputdf = pd.DataFrame(breednum, columns=['Result'])
        print(outputdf)
        merged = outputdf.merge(deduped,left_on=['Result'], right_on=['breednum'])
        
        inputwithoutput = inputdf.join(merged)
        finaldf = inputwithoutput[['FileNames', 'breed']]
        finaldf.set_index('FileNames', inplace=True)

        #print(inputwithoutput)
        #print(finaldf)
        #breedname =  deduped.iloc[breednum[0]].values[0] 
        #resp = AMLResponse(finaldf.to_dict(), 200)
        #resp.headers['Access-Control-Allow-Origin'] = "*"
        #return resp
        return  finaldf.to_dict()
        #return data 
        # You can return any data type, as long as it is JSON serializable.

    except Exception as e:
        error = str(e)
        return error