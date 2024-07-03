import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import sys
sys.path.append('..')
from utils.utils import *
from transformers import logging
logging.set_verbosity_error()
#去除未使用pooler计算损失的警告


class Dataset(Dataset):
    def __init__(self, list_IDs, labels, df_cci, d_tokenizer):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_cci

        self.d_tokenizer = d_tokenizer

    'tokenizer转化'

    def convert_data(self, drug_data_1, drug_data_2):
        d_inputs_1 = self.d_tokenizer(drug_data_1, return_tensors="pt")
        d_inputs_2 = self.d_tokenizer(drug_data_2, return_tensors="pt")

        drug_input_ids_1 = d_inputs_1['input_ids']
        drug_attention_mask_1 = d_inputs_1['attention_mask']
        drug_inputs_1 = {'input_ids': drug_input_ids_1, 'attention_mask': drug_attention_mask_1}

        drug_input_ids_2 = d_inputs_2['input_ids']
        drug_attention_mask_2 = d_inputs_2['attention_mask']
        drug_inputs_2 = {'input_ids': drug_input_ids_2, 'attention_mask': drug_attention_mask_2}

        return drug_inputs_1, drug_inputs_2

    '使用预训练好的tokenizer将smiles和sequence转化为模型需要的格式'

    def tokenize_data(self, drug_data_1, drug_data_2):
        tokenize_drug_1 = ['[CLS]'] + self.d_tokenizer.tokenize(drug_data_1) + ['[SEP]']
        tokenize_drug_2 = ['[CLS]'] + self.d_tokenizer.tokenize(drug_data_2) + ['[SEP]']

        return tokenize_drug_1, tokenize_drug_2

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        drug_data_1 = self.df.iloc[index]['smiles_1']
        drug_data_2 = self.df.iloc[index]['smiles_2']

        d_inputs_1 = self.d_tokenizer(drug_data_1, padding='max_length', max_length=510, truncation=True,
                                      return_tensors="pt")
        d_inputs_2 = self.d_tokenizer(drug_data_2, padding='max_length', max_length=510, truncation=True,
                                      return_tensors="pt")

        d_input_ids_1 = d_inputs_1['input_ids'].squeeze()
        d_attention_mask_1 = d_inputs_1['attention_mask'].squeeze()
        d_input_ids_2 = d_inputs_2['input_ids'].squeeze()
        d_attention_mask_2 = d_inputs_2['attention_mask'].squeeze()

        labels = torch.as_tensor(self.labels[index], dtype=torch.float)

        dataset = [d_input_ids_1, d_attention_mask_1, d_input_ids_2, d_attention_mask_2, labels]
        return dataset


class DataModule(pl.LightningDataModule):
    def __init__(self, task_name, drug_model_name, num_workers, batch_size,
                 traindata_rate=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_name = task_name
        self.traindata_rate = traindata_rate

        self.d_tokenizer = AutoTokenizer.from_pretrained(drug_model_name)

        self.df_train = None
        self.df_val = None
        self.df_test = None

        self.load_testData = True

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def get_task(self, task_name):
        if task_name.lower() == 'stitch':
            return 'data/STITCH/process'

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from
        # a single process in distributed settings.
        dataFolder = self.get_task(self.task_name)

        self.df_train = pd.read_csv(dataFolder + '/data_train.csv')
        self.df_val = pd.read_csv(dataFolder + '/data_val.csv')

        ## -- Data Lenght Rate apply -- ##
        traindata_length = int(len(self.df_train) * self.traindata_rate)
        validdata_length = int(len(self.df_val) * self.traindata_rate)

        self.df_train = self.df_train[:traindata_length]
        self.df_val = self.df_val[:validdata_length]

        if self.load_testData is True:
            self.df_test = pd.read_csv(dataFolder + '/data_test.csv')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Dataset(self.df_train.index.values, self.df_train.Label.values, self.df_train,
                                         self.d_tokenizer)
            self.valid_dataset = Dataset(self.df_val.index.values, self.df_val.Label.values, self.df_val,
                                         self.d_tokenizer)

        if self.load_testData is True:
            self.test_dataset = Dataset(self.df_test.index.values, self.df_test.Label.values, self.df_test,
                                        self.d_tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

