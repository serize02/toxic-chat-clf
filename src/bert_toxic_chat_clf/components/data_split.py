import pandas as pd

from pathlib import Path
from bert_toxic_chat_clf.entity.config_entity import DataSplitConfig

class DataSplit:

    def __init__(self, config: DataSplitConfig):
        
        self.config = config


    def load_data(self):

        self.df = pd.read_csv(self.config.data_path)


    def split(self):

        df = self.df
        train_size = self.config.params_train_size

        train_dataset=df.sample(frac=train_size,random_state=42)
        test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
        train_dataset = train_dataset.reset_index(drop=True)

        self.save_data(self.config.train_data_path, train_dataset)
        self.save_data(self.config.test_data_path, test_dataset)
        

    @staticmethod
    def save_data(path: Path, data: pd.DataFrame):
        data.to_csv(path, index=False)
