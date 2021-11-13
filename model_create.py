from TicketClassifierModel import TicketClassifierModel
training_dataset = 'TicketTrainingData.csv'
testing_dataset = 'TicketTestingData.csv'

ticket_model = TicketClassifierModel(training_dataset=training_dataset,
                                    testing_dataset=testing_dataset,
                                    recreate_model=True)
ticket_model.evaluate_model(testing_dataset=testing_dataset)