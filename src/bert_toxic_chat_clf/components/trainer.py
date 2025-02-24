import pandas as pd
import numpy as np
import torch
import transformers

from pathlib import Path
from torch.utils.data import DataLoader
from torch import cuda
from bert_toxic_chat_clf.components.setup_model import BertClass
from sklearn import metrics
from bert_toxic_chat_clf.entity.config_entity import TrainingConfig
from bert_toxic_chat_clf.components.custom_dataset import CustomDataset

class Trainer:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = BertClass()


    def setup_device(self):
        self.device = 'cuda' if cuda.is_available() else 'cpu'


    def setup_model(self):
        self.model.load_state_dict(torch.load(self.config.model_path, weights_only=False))


    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config.params_learning_rate)


    def load_data(self):

        df = pd.read_csv(self.config.training_data_path)

        training_set = CustomDataset(df, self.config.params_max_len)

        train_params = {
            'batch_size': self.config.params_train_batch_size,
            'shuffle': self.config.params_train_shuffle,
            'num_workers': self.config.params_train_num_workers
        }

        self.training_loader = DataLoader(training_set, **train_params)


    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    

    def train(self):
        
        self.model.to(self.device)

        epochs = self.config.params_epochs

        for epoch in range(epochs):

            self.model.train()
            
            for _,data in enumerate(self.training_loader, 0):
                
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(self.device, dtype = torch.long)
                targets = data['targets'].to(self.device, dtype = torch.float)

                outputs = self.model(ids, mask, token_type_ids)

                self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                if _%5000==0:
                    print(f'Epoch: {epoch}, Loss:  {loss.item()}')
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        self.save_model(path=self.config.trained_model_path, model=self.model)
            

    
    @staticmethod
    def save_model(path: Path, model: BertClass):
        torch.save(model.state_dict(), path)
