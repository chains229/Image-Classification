    
import os
import torch
from data_utils.data_loader import getDataloader
from evaluate.evaluate import cal_score
from model.model import CNN_Model
from tqdm import tqdm
from sklearn.metrics import classification_report

class Predicting():
    def __init__(self,config):
        self.save_path=config['save_path']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN_Model(config).to(self.device)
        self.test_loader = getDataloader(config).get_test()
    
    def main(self):
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'), map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('havent trained yet...')


        self.model.eval()
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for batch, (it,item) in enumerate(tqdm(self.test_loader)):
                images, labels = item['image'].to(self.device), item['label'].to(self.device)
                logits = self.model(images)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(logits.argmax(-1).cpu().numpy())
        
        cm, test_acc, test_f1, test_precision, test_recall = cal_score(true_labels,pred_labels)
        print(f"test acc: {test_acc:.4f} | test f1: {test_f1:.4f} | test precision: {test_precision:.4f} | test recall: {test_recall:.4f}")
        print("confusion matrix:\n")
        print(cm)
        print('classification report:\n')
        print(classification_report(true_labels, pred_labels))
        
