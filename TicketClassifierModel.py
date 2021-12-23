import pandas as pd
import numpy as np
import os
import pickle
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras import utils
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import model_from_json, Model
from tensorflow.keras.utils import to_categorical

from keras.layers import Activation, Dense, Dropout
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import sklearn.datasets as skds
import re
import logging
import sys
from config import myapp_config

logging.basicConfig(
    filename=os.path.join(myapp_config.LOG_PATH, myapp_config.MODEL_LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


class TicketClassifierModel:
    def __init__(self, *args, **kwargs):
        """Init and Load model"""
        self._tokenizer = type("Tokenizer", (), {})
        self.labels = []
        self._model = type("model_from_json", (), {})
        self.accuracy = 0
        self._train_data = self.load_prepare_dataset(*args, **kwargs)
        # Create model
        if kwargs.get("recreate_model") == True:
            self.labels = self.create_model(myapp_config.MODEL_NAME)
            self.test_data = self.validate_model(*args, **kwargs)

    def load_prepare_dataset(self, *args, **kwargs):
        """
        loading dataset from csv and removal of null keys which are mandatory for training:
            TicketShortDesc and Activity
        """
        dataset=kwargs.get("training_dataset")
        logging.info("Preparing to read csv dataset: " + str(dataset))
        df = pd.read_csv(os.path.join(myapp_config.DATASETS_PATH, dataset), dtype=str)

        logging.info(
            "DS Shape before ShortDescription and Activity cleanup: " + str(df.shape)
        )
        drop_if_na = ["ShortDescription", "Activity"]
        for i in range(0, len(drop_if_na)):
            logging.info("Removing nulls from column " + str(drop_if_na[i]))
            df.dropna(subset=[drop_if_na[i]], inplace=True)
        logging.info(
            "DS Shape after ShortDescription and Activity cleanup: " + str(df.shape)
        )
        logging.info(
            df.head()
        )
        return df

    def preprocess_text(self, sen):
        """
        1) Remove punctuations and number
        2)  Single character removal
        3) Removing Multiple Spaces
        """
        sentence = re.sub("[^a-zA-Z]", " ", sen)
        sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)
        sentence = re.sub(r"\s+", " ", sentence)

        return sentence

    def create_model(self, *args, **kwargs):
        """
        Model creation
        """
        data = self._train_data
        headlines = []
        for sen in data["ShortDescription"]:
            headlines.append(self.preprocess_text(sen))
        # lets take 80% data as training and remaining 20% for test.
        train_size = int(len(data) * 0.9)
        test_size = int(len(data) * 0.4)

        train_headlines = headlines
        train_category = data["Category"]
        train_activities = data["Activity"]

        test_headlines = headlines[:test_size]
        test_category = data["Category"][:test_size]
        test_activities = data["Activity"][:test_size]
        # data.to_csv (os.path.join('outputs','TrainingDataCleaned.csv'), index = None, header=True)

        X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
            data["ShortDescription"], data["Category"], data["Activity"], test_size=0.15
        )
        logging.info("ShortDescription Train len " + str(len(X_train)))
        logging.info("Category Train len " + str(len(Y_train)))
        logging.info("Activity Train len " + str(len(Z_train)))

        logging.info("ShortDescription Test len " + str(len(X_test)))
        logging.info("Category Test len " + str(len(Y_test)))
        logging.info("Activity Test len " + str(len(Z_train)))

        # define Tokenizer with Vocab Sizes
        vocab_size1 = 10000
        vocab_size2 = 10000
        tokenizer = Tokenizer(num_words=vocab_size1)
        tokenizer2 = Tokenizer(num_words=vocab_size2)

        tokenizer.fit_on_texts(X_train)
        tokenizer2.fit_on_texts(Y_train)

        x_train = tokenizer.texts_to_matrix(X_train, mode="tfidf")
        x_test = tokenizer.texts_to_matrix(X_test, mode="tfidf")

        y_train = tokenizer2.texts_to_matrix(Y_train, mode="tfidf")
        y_test = tokenizer2.texts_to_matrix(Y_test, mode="tfidf")

        # Create classes file
        encoder = LabelBinarizer()
        encoder.fit(Z_train)
        text_labels = encoder.classes_
        with open(os.path.join(myapp_config.OUTPUT_PATH, "classes.txt"), "w") as f:
            for item in text_labels:
                f.write("%s\n" % item)
        z_train = encoder.transform(Z_train)
        z_test = encoder.transform(Z_test)
        num_classes = len(text_labels)
        logging.info("Numbers of classes found: " + str(num_classes))

        # Model creation and summarization
        batch_size = 100
        input1 = Input(shape=(vocab_size1,), name="main_input")
        x1 = Dense(512, activation="relu")(input1)
        x1 = Dropout(0.5)(x1)
        input2 = Input(shape=(vocab_size2,), name="cat_input")
        main_output = Dense(num_classes, activation="softmax", name="main_output")(x1)
        model = Model(inputs=[input1, input2], outputs=[main_output])
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        model.summary()

        # Model Evaluation
        history = model.fit(
            [x_train, y_train],
            z_train,
            batch_size=batch_size,
            epochs=10,
            verbose=1,
            validation_split=0.10,
        )
        score = model.evaluate(
            [x_test, y_test], z_test, batch_size=batch_size, verbose=1
        )

        logging.info("Test accuracy:", str(score[1]))
        self.accuracy = score[1]
        # serialize model to JSON
        model_json = model.to_json()
        with open(
            os.path.join(
                myapp_config.OUTPUT_PATH, "model_" + myapp_config.MODEL_NAME + ".json"
            ),
            "w",
        ) as json_file:
            json_file.write(model_json)
        # creates a HDF5 file 'my_model.h5'
        model.save(
            os.path.join(
                myapp_config.OUTPUT_PATH, "model_" + myapp_config.MODEL_NAME + ".h5"
            )
        )

        # Save Tokenizer i.e. Vocabulary
        with open(
            os.path.join(
                myapp_config.OUTPUT_PATH,
                "tokenizer" + myapp_config.MODEL_NAME + ".pickle",
            ),
            "wb",
        ) as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Predict a few samples
        predictions = model.predict([x_test, y_test])
        for i in range(5):
            import random

            j = int(random.uniform(0, len(x_test)))
            prediction = predictions[j]
            predicted_label = text_labels[np.argmax(prediction)]
            ihighest_score = np.argmax(prediction)
            predicted_score = prediction[ihighest_score]
            logging.debug("\n" + test_headlines[j])
            logging.debug("Actual label:" + test_activities.iloc[j])
            logging.debug("Predicted label: " + predicted_label)
            logging.debug("Number of label: " + str(ihighest_score))
            logging.debug("Predicted score: " + str(predicted_score))
            logging.debug("Actual category:" + test_category.iloc[j])
        return text_labels

    def validate_model(self, *args, **kwargs):
        """
        Load Models and predicts test data using the labels
        """
        test_data = self.load_prepare_dataset(*args, **kwargs)
        labels = self.labels
        # load json and create model
        json_file = open(
            os.path.join(
                myapp_config.OUTPUT_PATH, "model_" + myapp_config.MODEL_NAME + ".json"
            ),
            "r",
        )
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(
            os.path.join(
                myapp_config.OUTPUT_PATH, "model_" + myapp_config.MODEL_NAME + ".h5"
            )
        )
        logging.info("Loaded model from disk")

        test_data["ShortDescription"] = (
            test_data["ShortDescription"] + " " + test_data["Description"]
        )
        headlines = test_data["ShortDescription"].astype(str)
        categories = test_data["Category"].astype(str)

        # loading
        with open(
            os.path.join(
                myapp_config.OUTPUT_PATH,
                "tokenizer" + myapp_config.MODEL_NAME + ".pickle",
            ),
            "rb",
        ) as handle:
            tokenizer = pickle.load(handle)
        # ShortDescriptions
        x_pred = tokenizer.texts_to_matrix(headlines, mode="tfidf")
        # Categorias
        y_pred = tokenizer.texts_to_matrix(categories, mode="tfidf")
        predictions = []
        scores = []
        logging.info(len(x_pred))
        logging.info(len(y_pred))
        predictions_vetor = model.predict({"main_input": x_pred, "cat_input": y_pred})
        for i in range(len(predictions_vetor)):
            prediction = predictions_vetor[i]
            predicted_label = labels[np.argmax(prediction)]
            ihighest_score = np.argmax(prediction)
            predicted_score = prediction[ihighest_score]
            predictions.append(predicted_label)
            scores.append(predicted_score)

        test_data["Prediction"] = predictions
        test_data["Score"] = scores
        test_data["Equal"] = np.where(
            test_data["Activity"].str.strip() == test_data["Prediction"].str.strip(),
            "yes",
            "no",
        )
        file_name = myapp_config.MODEL_NAME + "_Testing_Predictions.csv"
        test_data.to_csv(
            os.path.join("outputs", file_name), index=None, header=True
        )  # Don't forget to add '.csv' at the end of the path
        return test_data

    def evaluate_model(self, *args, **kwargs):
        """
        Load a model, validate the accuracy generating a success/failure report
        """
        test_data = self.test_data
        logging.info("Number total of records: " + str(len(test_data)))
        logging.info("Number of values in each column: " + str(test_data.count))
        logging.info(
            "Number of distinct Activities: " + str(len(test_data.Activity.unique()))
        )

        yes = test_data.Equal.value_counts()["yes"]
        logging.info("\nNumber of distinct Successfull classifications: " + str(yes))
        no = test_data.Equal.value_counts()["no"]
        logging.info("\nNumber of distinct Failed classifications: " + str(no))
        success_rate = (yes * 100) / len(test_data)
        logging.info("\nSuccess rate: " + str(round(success_rate)) + "%")

        if len(test_data) == (yes + no):
            logging.info("\nNumber of records in test and compared are the same. Fine")
        else:
            logging.info(
                "\nNumber of records in test and compared are NOT the same. Please verify!"
            )

        test_data_success = test_data[test_data.Equal == "yes"]
        file_name = myapp_config.MODEL_NAME + "_Testing_Predictions_Success.csv"
        logging.info("\nSaving successful to dataset: " + file_name)
        test_data_success.to_csv(
            os.path.join("outputs", file_name), index=None, header=True
        )  # Don't forget to add '.csv' at the end of the path

        test_data_failed = test_data[test_data.Equal == "no"]
        file_name = myapp_config.MODEL_NAME + "_Testing_Predictions_Failed.csv"
        logging.info("\nSaving failed to dataset: " + file_name)
        test_data_failed.to_csv(
            os.path.join("outputs", file_name), index=None, header=True
        )  # Don't forget to add '.csv' at the end of the path

        logging.info("\nTop 10 Successfull classification by Activities")
        logging.info(
            test_data_success.groupby("Activity")["Equal"]
            .value_counts()
            .sort_values(ascending=False)[0:10]
        )
        logging.info("\nTop 10 Failed classification by Activities")
        logging.info(
            test_data_failed.groupby("Activity")["Equal"]
            .value_counts()
            .sort_values(ascending=False)[0:10]
        )

        logging.info("\nTop 10 Successfull classification by Company")
        logging.info(
            test_data_success.groupby("Company")["Equal"]
            .value_counts()
            .sort_values(ascending=False)[0:10]
        )
        logging.info("\nTop 10 Failed classification by Company")
        logging.info(
            test_data_failed.groupby("Company")["Equal"]
            .value_counts()
            .sort_values(ascending=False)[0:10]
        )

        failed_cw = test_data_failed.loc[test_data_failed["Company"] == ""]
        file_name = myapp_config.MODEL_NAME + "_Testing_Predictions_Failed_CW.csv"
        logging.info("\nSaving failed CW to dataset: " + file_name)
        failed_cw.to_csv(
            os.path.join("outputs", file_name), index=None, header=True
        )  # Don't forget to add '.csv' at the end of the path
