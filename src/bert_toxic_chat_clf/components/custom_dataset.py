import torch
import transformers

from torch.utils.data import Dataset
from transformers import BertTokenizer

class CustomDataset(Dataset):

    def __init__(self, dataframe, max_len):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.data = dataframe
        self.message = dataframe.message
        self.targets = self.data.target
        self.max_len = max_len
    

    def __len__(self):
        return len(self.message)


    def __getitem__(self, index):
        message = str(self.message[index])
        message = " ".join(message.split())

        inputs = self.tokenizer.encode_plus(
            message,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(eval(self.targets[index]), dtype=torch.float)
        }
    