#Example: scikit-learn and Swagger
import tensorflow as tf
import pandas as pd
import numpy as np
from azureml.core import Workspace, Model
from azureml.core.authentication import ServicePrincipalAuthentication
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.preprocessing.text import tokenizer_from_json
import tensorflow.keras.preprocessing.text
import os
import json

def init():
    global model
    global ntokenizer
    # AZUREML_MODEL_DIR is an environment variable created during deployment. Join this path with the filename of the model file.
    # It holds the path to the directory that contains the deployed model (./azureml-models/$MODEL_NAME/$VERSION).
    # If there are multiple models, this value is the path to the directory containing all deployed models (./azureml-models).
    # Alternatively: model_path = Model.get_model_path('sklearn_mnist')
    #model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'sklearn_mnist_model.pkl')
    # Deserialize the model file back into a sklearn model
    

    sp = ServicePrincipalAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47", # tenantID
                                    service_principal_id="2faad9e6-8bf9-4e7d-a732-70e3daa5ffd5", # clientId
                                    service_principal_password="#######") # clientSecret


    workspace = Workspace('bd04922c-a444-43dc-892f-74d5090f8a9a', 'mlplayarearg', 'testdeployment',auth=sp)
    model_path = Model.get_model_path(model_name='sentimentanalysisv2',_workspace=workspace)
    model = tf.keras.models.load_model(model_path)
    print(model.summary())
   

    print (tf.__version__)
    with open('./packagefordeployment/tokenizer0421.json') as f:
        data = json.load(f)
        ntokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)


import numpy as np
def run(data):
    try:
        data = json.loads(data)
        sentences = data['sentences']

        testsequences = ntokenizer.texts_to_sequences(sentences)
        testpadded = pad_sequences(testsequences, maxlen=40)
        answer = model.predict(testpadded)
        answer = answer.reshape(len(sentences))
        answer = np.array(answer, dtype = np.str)
        #print(answer)
        #res = dict(zip(sentences, answer)) 
        #print(res)
        answer = list(answer)
        
        #result = {"sentiment":str(answer[0])}
        return json.dumps(answer)
        
    except Exception as e:
        error = str(e)
        return error