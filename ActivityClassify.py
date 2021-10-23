import os

import pandas
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import numpy as np
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from datetime import datetime
import pandas as pd
import logging
import sys

MODEL_PATH = 'model'
LOG_PATH='logs'
LOG_FILE='ActivityClassify.log'
OUTPUT_PATH = 'outputs'
STANDARD_PREDICT_INPUT_FILE = 'activity_predict_inputs.json'

logging.basicConfig(filename=os.path.join(LOG_PATH,LOG_FILE),level=logging.DEBUG,format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

#def activity_predict(input_data: dict, ):
def activity_predict(*args, **kwargs):
    """ This function is responsible for loading and model and classification classes and based on the input data it returns one or more predictions"""
    
    # Load Prediction Input file
    if 'Headline' not in kwargs:
        with open(os.path.join(OUTPUT_PATH, STANDARD_PREDICT_INPUT_FILE), 'r') as fp:
            input_data = json.load(fp)
        predict_dataset_file = True
    else: 
        # Single classification 
        headline = [kwargs['Headline']]
        # Category is optional
        category = [kwargs.get('Category')]
        #category = [input_data['Category']]

    #load classes
    class_file = kwargs.pop('class_file', 'classes.txt')
    logging.info("Loading classes file: "+class_file)
    labels = []
    text_file = open(os.path.join(MODEL_PATH, class_file), "r")
    for line in text_file:
        line = line.replace("\n", "")
        labels.append(line)
    
    # load json model
    model_name = kwargs.pop('model_name', 'ActivityMap_Cat_Headline')
    model_file_name = "model_"+model_name+".json"
    logging.info("Loading model called: "+model_file_name)
    json_file = open(os.path.join(MODEL_PATH, model_file_name))
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(os.path.join(MODEL_PATH, "model_"+model_name+".h5"))
    
    # load tockenizers
    token_model_file = "tokenizer"+model_name+'.pickle'
    logging.info("Loading model tokenizer: "+token_model_file)
    import pickle
    with open(os.path.join(MODEL_PATH, token_model_file), 'rb') as handle:
        tokenizer = pickle.load(handle)
    predict_dataset_file = False 
    

    # Headlines 
    x_pred = tokenizer.texts_to_matrix(headline, mode='tfidf')
    # Categorias
    y_pred = tokenizer.texts_to_matrix(category, mode='tfidf')
    
    logging.info("Headline "+str(headline))
    logging.info("Category "+str(category))
    predictions = []
    scores = []
    model_predictions = model.predict({'main_input': x_pred, 'cat_input': y_pred})
    top5_pred_probs = []
    # dataframe prediction
    if predict_dataset_file is True:
        logging.info("Running Entire Dataset Prediction")
        df = pandas.DataFrame()
        predictions = []
        scores = []
        for i in range(0, len(x_pred)):
            prediction = model_predictions[i]
            predicted_label = labels[np.argmax(prediction)]
            ihighest_score = np.argmax(prediction)
            predicted_score = prediction[ihighest_score] 
            predictions.append(predicted_label)
            scores.append(predicted_score)
        df['Prediction'] = predictions
        df['Score'] = scores
        now = datetime.now()
        timestamp = datetime.timestamp(now)        
        #newfilename = os.path.splitext(filename)[0]+timestamp+'.csv'
        
        if 'filename' in kwargs:
            # Using absolute paths
            filename = kwargs.pop('filename')
            newfilename = os.path.split(filename)[1]
            newfilename = os.path.splitext(newfilename)[0]+str(timestamp)+'.csv'
        else:
            newfilename = "Predictions_"+str(timestamp)+'.csv'
        newfilename = os.path.join(OUTPUT_PATH,newfilename)
        #newfilename = './outputs/'+newfilename
        df.to_csv(newfilename)        
        logging.info("File saved to {}".format(newfilename))
        return newfilename
    # Individual prediction
    else:
        logging.info("Running Individual Ticket Prediction")
        sorting = (-model_predictions).argsort()
        sorted_ = sorting[0][:5]
        for value in sorted_:   
            predicted_label = labels[value]
            # just some rounding steps
            prob = (model_predictions[0][value]) * 100
            prob = "%.2f" % round(prob,2)
            top5_pred_probs.append([prob,predicted_label])
        output = {'headline':headline[0],'category':category[0],'top5_pred_probs': top5_pred_probs}
        with open(os.path.join(OUTPUT_PATH,'activity_predict_output.json'), 'w') as fp:
            json.dump(output, fp)  
        logging.info(output)
        return output

def load_prepare_dataset (dataset):
    import logging
    import sys    
    # loading dataset from csv
    data = pd.read_csv(
        dataset,
        dtype=str
       # encoding='utf-8'
    )  

    logging.info("Preparing dataset: "+dataset)
    logging.info("Training Dataset Shape: ")
    logging.info(data.shape)
    data.dropna(subset=['Headline'], inplace=True)
    logging.info("Training Dataset Shape after nulls headline: ")
    logging.info(data.shape)

    # Data higienization
    data["Headline"] = data["Headline"].str.replace(r'ITIL_Transaction.Description Like', '')
    data["Headline"] = data["Headline"].str.replace(r'ITIL_Transaction.Headline Like', '')
    data["Headline"] = data["Headline"].str.replace(r'ITIL_Transaction.Categorization_Tier_2', '')
    data["Headline"] = data["Headline"].str.replace(r'ITIL_Transaction.Categorization_Tier_1', '')
    
    data["Headline"] = data["Headline"].str.replace(r'\(\( \'', '')
    data["Headline"] = data["Headline"].str.replace(r'\%\'\)\)','')
    data["Headline"] = data["Headline"].str.replace(r'\%',' ')
    data["Headline"] = data["Headline"].str.replace(r'\(\( \= \'',' ')
    data["Headline"] = data["Headline"].str.replace(r'\'\)\)',' ')
    data["Headline"] = data["Headline"].astype(str)
    # Deal with Category multiple column names 
    category = ''
    if 'Category' in data.keys():
        category += data['Category'].astype(str)    
    elif 'Categorization_Tier_1' in data.keys():
        category += data['Categorization_Tier_1'].astype(str)
    if 'Categorization_Tier_2' in data.keys():
        category += ' '+data['Categorization_Tier_2'].astype(str)
    if 'Categorization_Tier_3' in data.keys():
        category += ' '+data['Categorization_Tier_3'].astype(str)
    data["Category"] = category
    headline = []
    for sen in data['Headline']:
        headline.append(preprocess_text(sen))
    data["Headline"] = headline
    # current date and time
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    data.to_csv (r'./outputs/DataCleanedup'+str(timestamp)+'.csv', index = None, header=True) 
    return data

def preprocess_text(sen):
    import re
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)  
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence
