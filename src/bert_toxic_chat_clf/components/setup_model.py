import transformers
import torch

from transformers import BertModel, BertConfig
from bert_toxic_chat_clf.entity.config_entity import SetupModelConfig
from pathlib import Path

class BertClass(torch.nn.Module):

    def __init__(self, classes: int = 3, dropout: float = 0.3):
        super(BertClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(dropout)
        self.l3 = torch.nn.Linear(768, classes)

    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class SetupModel:

    def __init__(self, config: SetupModelConfig):
        self.config = config

    
    def get_model(self):
        
        self.model = BertClass(
            classes=self.config.params_classes,
            dropout=self.config.params_dropout
        )
        
        self.save_model(path=self.config.model_path, model=self.model)
    

    @staticmethod
    def save_model(path: Path, model: BertClass):
        torch.save(model.state_dict(), path)
