from ActivityClassify import TicketActivityPredict
import pdb
classifier = TicketActivityPredict()
#print (vars(classifier))
#pdb.set_trace()

#classifier.load_model()
classifier.predict_text(Headline='CPU is too high',
                        Category='(Self Heal) High CPU utilization')
#print (activity_predict(Headline='CPU is too high',Category='(Self Heal) High CPU utilization'))
