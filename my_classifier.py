from TicketActivityClassify import TicketActivityPredict
classifier = TicketActivityPredict()
# Return top 5 prediction scores 
prediction = classifier.predict_text(ShortDescription='Unlock of an Active Directory Admin or Server Account account or account',
                        Category='Account Update Account Administration')
print(prediction)
#{'short_description': 'Unlock of an Active Directory Admin or Server Account account or account', 'category': 'Account Update Account Administration', 'top5_pred_probs': [['87.09', 'AD User Isse'], ['12.90', 'Password reset'], ['0.00', 'Application Access'], ['0.00', 'Script Execution'], ['0.00', 'DB Connection']]}