from unittest import TestCase
from TicketClassifierModel import TicketClassifierModel


class TryTesting(TestCase):
    def test_activity_model(self):
        training_dataset = 'TicketTrainingData.csv'
        testing_dataset = 'TicketTestingData.csv'
        ticket_model = TicketClassifierModel(training_dataset=training_dataset,
                                    testing_dataset=testing_dataset,
                                    recreate_model=True)
        self.assertTrue(100*ticket_model.accuracy >= 80)

