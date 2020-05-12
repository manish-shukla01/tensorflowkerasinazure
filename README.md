## Train and deploy model in azure using keras/tensorflow and then deploy using Azure Machine learning

### Example 1: Train vision model and deploy to Azure ACI

1. Train model using transfer learning from InceptionV3 using GPU DSVM using kaggle dataset. Code with inline documentation is [here.](https://github.com/manish-shukla01/tensorflowkerasinazure/blob/master/Dogbreeddetection/dogbreeddetector.ipynb)
2. Save the model to Azure machine learning.
3. Package the model and its depencies using Azure machine learning to deploy a webservice. Whatever is there in the folder package for deployment gets packaged and can be referenced from score.py.
4. Call the service using postman. its very similar to the sentiment analysis in example 2.
5. Simple react UI to show the code e2e. (to be added later)

### Example 2: Sentiment analysis model deployment to azure

1. Train model using word embedding from Glove Model. Code with inline documentation is [here.](https://github.com/manish-shukla01/tensorflowkerasinazure/blob/master/sentimentanalysis/nlpstuff.ipynb)
2. Save the model to Azure machine learning.
3. Package the model and its depencies using Azure machine learning to deploy a webservice
4. Call the service using postman. 
![postmancall]('https://github.com/manish-shukla01/tensorflowkerasinazure/blob/master/images/sentimentpostmancall.png)

