import os
import pandas
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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
from config import myapp_config

logging.basicConfig(
    filename=os.path.join(myapp_config.LOG_PATH, myapp_config.CLASSIFIER_LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# self.load_model(*args, **kwargs)
# It's a dataframe prediction
# if "ShortDescription"in kwargs:
#    self.predict_text(*args, **kwargs)
# else:
#     self.predict_dataset(*args, **kwargs)


class TicketActivityPredict:
    def __init__(self, *args, **kwargs):
        """ Init and Load model """
        self._tokenizer = type("Tokenizer", (), {})
        self.labels = []
        self._model = type("model_from_json", (), {})
        self.load_model()
    def activity_predict(self, *args, **kwargs):
        """This function is responsible for loading and model and classification classes and based on the input data it returns one or more predictions"""
        """ Inputs """
        """         If ShortDescription is a true parameter we consider this as single request classification """
        # Load Prediction Input file
        if "ShortDescription" not in kwargs:
            with open(
                os.path.join(
                    myapp_config.OUTPUT_PATH, myapp_config.DEFAULT_PREDICT_INPUT_FILE
                ),
                "r",
            ) as fp:
                input_data = json.load(fp)
        else:
            # Single classification
            short_description = [kwargs["ShortDescription"]]
            # Category is optional
            category = [kwargs.get("Category")]

    def load_model(self, *args, **kwargs):
        """This functions loads any model"""

        logging.info("Loading Model")
        # load classes
        class_file = kwargs.pop("class_file", "classes.txt")
        logging.info("Loading classes file: " + class_file)
        text_file = open(os.path.join(myapp_config.MODEL_PATH, class_file), "r")
        for line in text_file:
            line = line.replace("\n", "")
            self.labels.append(line)

        # load json model
        model_name = kwargs.pop("model_name", "Activity_Classifier")
        model_file_name = "model_" + model_name + ".json"

        logging.info("Loading model called: " + model_file_name)
        json_file = open(os.path.join(myapp_config.MODEL_PATH, model_file_name))
        loaded_model_json = json_file.read()
        json_file.close()

        model = model_from_json(loaded_model_json)
        model.load_weights(
            os.path.join(myapp_config.MODEL_PATH, "model_" + model_name + ".h5")
        )
        self._model = model
        # load tockenizers
        token_model_file = "tokenizer" + model_name + ".pickle"
        logging.info("Loading model tokenizer: " + token_model_file)
        import pickle

        with open(
            os.path.join(myapp_config.MODEL_PATH, token_model_file), "rb"
        ) as handle:
            tokenizer = pickle.load(handle)
            self._tokenizer = tokenizer
        self._tokenizer = tokenizer
        logging.info("Loading model tokenizer loaded")

    def predict_dataset(self, *args, **kwargs):
        """ Predicts an entire dataset """
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
        df["Prediction"] = predictions
        df["Score"] = scores
        timestamp = datetime.timestamp(datetime.now())
        if "filename" in kwargs:
            # Using absolute paths
            filename = kwargs.pop("filename")
            newfilename = os.path.split(filename)[1]
            newfilename = os.path.splitext(newfilename)[0] + str(timestamp) + ".csv"
        else:
            newfilename = "Predictions_" + str(timestamp) + ".csv"

        # Saves new file
        newfilename = os.path.join(myapp_config.OUTPUT_PATH, newfilename)
        df.to_csv(newfilename)
        logging.info("File saved to {}".format(newfilename))
        return newfilename

    def predict_text(self, *args, **kwargs):
        """ Individual sample prediction"""
        # Single classification
        short_description = [kwargs["ShortDescription"]]
        # Category is optional
        category = [kwargs.get("Category")]
        logging.info("ShortDescription " + str(short_description))
        logging.info("Category " + str(category))
        predictions = []
        scores = []
        top5_pred_probs = []
        # ShortDescriptions
        x_pred = self._tokenizer.texts_to_matrix(short_description, mode="tfidf")
        # Categorias
        y_pred = self._tokenizer.texts_to_matrix(category, mode="tfidf")

        model_predictions = self._model.predict(
            {"main_input": x_pred, "cat_input": y_pred}
        )

        logging.info("Running Individual Ticket Prediction")
        sorting = (-model_predictions).argsort()
        sorted_ = sorting[0][:5]
        for value in sorted_:
            predicted_label = self.labels[value]
            # just some rounding steps
            prob = (model_predictions[0][value]) * 100
            prob = "%.2f" % round(prob, 2)
            top5_pred_probs.append([prob, predicted_label])
        output = {
            "short_description": short_description[0],
            "category": category[0],
            "top5_pred_probs": top5_pred_probs,
        }
        with open(
            os.path.join(myapp_config.OUTPUT_PATH, "activity_predict_output.json"), "w"
        ) as fp:
            json.dump(output, fp)
        logging.info(output)
        return output
