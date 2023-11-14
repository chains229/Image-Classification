from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def cal_score(labels, preds):
    cm = confusion_matrix(labels, preds)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average = 'macro', zero_division=1)
    precision = precision_score(labels, preds, average = 'macro', zero_division=1)
    recall = recall_score(labels, preds, average = 'macro', zero_division=1)
    return cm, accuracy, f1, precision, recall
