# TicketActivityClassifier - Purporse 

One of collest things about working with software is the largest number of opensource APIs available for utilization. Anyone can leverage some complex tool which abstracts a lot complexity, without a deep knowledge in the area, and then gain some knowledge about the complex process. That was my intention when I spent weekends exploring Keras, Keras is an highlevel Tensorflow API part, as i'm not a data scientist Keras was was a perfect initiation tool to understand how AI/NLP, Neural Networks and CNNs works in practice. 
# Another Keras Ticket Classification Model

That's just another ticket classification model, using an Neural network Classification, doing basic Machine Learning stuff like the below:

- Get/Prepare dataset
- Word vectors and embedding layers
Unique words - each one is assigned to a unique index to identify the workds during training
We need to represent the with numeric values

- Model creation
- Model evaluation
- Dataset Prediction

# How use it
## Creating the Model
```
from TicketClassifierModel import TicketClassifierModel
training_dataset = 'TicketTrainingData.csv'
testing_dataset = 'TicketTestingData.csv'

ticket_model = TicketClassifierModel(training_dataset=training_dataset,
                                    testing_dataset=testing_dataset,
                                    recreate_model=True)
ticket_model.evaluate_model(testing_dataset=testing_dataset)
```
### Making Predictions
```
from ActivityClassify import TicketActivityPredict
classifier = TicketActivityPredict()
# Return top 5 prediction scores 
prediction = classifier.predict_text(ShortDescription='Unlock of an Active Directory Admin or Server Account account or account',
                        Category='Account Update Account Administration')
print(prediction)
#{'short_description': 'Unlock of an Active Directory Admin or Server Account account or account', 'category': 'Account Update Account Administration', 'top5_pred_probs': [['87.09', 'AD User Isse'], ['12.90', 'Password reset'], ['0.00', 'Application Access'], ['0.00', 'Script Execution'], ['0.00', 'DB Connection']]})
```
# Reference Links

For more information about Keras Text classification I can recommend the follow links



- https://realpython.com/python-keras-text-classification/
- https://keras.io/examples/structured_data/structured_data_classification_from_scratch/