from unittest import TestCase
from ActivityClassify import TicketActivityPredict
class TryTesting(TestCase):
    def test_activity_predict(self):
        classifier = TicketActivityPredict()
        prediction = classifier.predict_text(Headline='CPU is too high',
                        Category='Functionality Issue')
       
        self.assertTrue(prediction['top5_pred_probs'][0][1]==
                        '(Self Heal) High CPU utilization')

