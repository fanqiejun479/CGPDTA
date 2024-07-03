import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import esm
import sys

sys.path.append('..')
from utils.utils import *
from transformers import logging,AutoTokenizer

logging.set_verbosity_error()


# 去除未使用pooler计算损失的警告


class ESM_Dataset(Dataset):
    def __init__(self, list_IDs, labels, df_ppi, p_tokenizer, prot_maxLength):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_ppi

        self.p_tokenizer = p_tokenizer

        self.prot_maxLength = prot_maxLength

    'tokenizer转化'

    def convert_data(self, prot_data_1, prot_data_2):
        if len(prot_data_1) > self.prot_maxLength:
            prot_data_1 = prot_data_1[:self.prot_maxLength]
        if len(prot_data_2) > self.prot_maxLength:
            prot_data_2 = prot_data_2[:self.prot_maxLength]

        _, _, p_inputs_1 = self.p_tokenizer([("", prot_data_1)])
        _, _, p_inputs_2 = self.p_tokenizer([("", prot_data_2)])

        pad_1 = nn.ZeroPad2d(padding=(0, self.prot_maxLength - int(p_inputs_1.shape[1]), 0, 0))
        pad_2 = nn.ZeroPad2d(padding=(0, self.prot_maxLength - int(p_inputs_2.shape[1]), 0, 0))
        p_inputs_1 = pad_1(p_inputs_1)
        p_inputs_2 = pad_2(p_inputs_2)

        return p_inputs_1, p_inputs_2

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        prot_data_1 = self.df.iloc[index]['prot_1']
        prot_data_2 = self.df.iloc[index]['prot_2']

        p_inputs_1, p_inputs_2 = self.convert_data(prot_data_1, prot_data_2)
        p_inputs_1 = p_inputs_1.squeeze()
        p_inputs_2 = p_inputs_2.squeeze()

        labels = torch.as_tensor(self.labels[index], dtype=torch.float)

        dataset = [p_inputs_1, p_inputs_2, labels]
        return dataset

class PB_Dataset(Dataset):
    def __init__(self, list_IDs, labels, df_dti, p_tokenizer, prot_maxLength):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti

        self.p_tokenizer = p_tokenizer

        self.prot_maxLength = prot_maxLength

    'tokenizer转化'

    def convert_data(self, prot_data_1, prot_data_2):
        prot_data_1 = ' '.join(list(prot_data_1))
        prot_data_2 = ' '.join(list(prot_data_2))

        p_inputs_1 = self.p_tokenizer(prot_data_1, return_tensors="pt")
        p_inputs_2 = self.p_tokenizer(prot_data_2, return_tensors="pt")

        prot_input_ids_1 = p_inputs_1['input_ids']
        prot_attention_mask_1 = p_inputs_1['attention_mask']
        prot_inputs_1 = {'input_ids': prot_input_ids_1, 'attention_mask': prot_attention_mask_1}

        prot_input_ids_2 = p_inputs_2['input_ids']
        prot_attention_mask_2 = p_inputs_2['attention_mask']
        prot_inputs_2 = {'input_ids': prot_input_ids_2, 'attention_mask': prot_attention_mask_2}

        return prot_inputs_1, prot_inputs_2

    '使用预训练好的token将smiles和sequence转化为模型需要的格式'

    def tokenize_data(self, prot_data_1, prot_data_2):
        prot_data_1 = ' '.join(list(prot_data_1))
        prot_data_2 = ' '.join(list(prot_data_2))

        tokenize_prot_1 = ['[CLS]'] + self.p_tokenizer.tokenize(prot_data_1) + ['[SEP]']
        tokenize_prot_2 = ['[CLS]'] + self.p_tokenizer.tokenize(prot_data_2) + ['[SEP]']

        return tokenize_prot_1, tokenize_prot_2

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        prot_data_1 = self.df.iloc[index]['prot_1']
        prot_data_1 = ' '.join(list(prot_data_1))
        prot_data_2 = self.df.iloc[index]['prot_2']
        prot_data_2 = ' '.join(list(prot_data_2))

        p_inputs_1 = self.p_tokenizer(prot_data_1, padding='max_length', max_length=self.prot_maxLength, truncation=True,
                                    return_tensors="pt")
        p_inputs_2 = self.p_tokenizer(prot_data_2, padding='max_length', max_length=self.prot_maxLength, truncation=True,
                                    return_tensors="pt")

        p_input_ids_1 = p_inputs_1['input_ids'].squeeze()
        p_attention_mask_1 = p_inputs_1['attention_mask'].squeeze()
        p_input_ids_2 = p_inputs_2['input_ids'].squeeze()
        p_attention_mask_2 = p_inputs_2['attention_mask'].squeeze()

        labels = torch.as_tensor(self.labels[index], dtype=torch.float)

        dataset = [p_input_ids_1, p_attention_mask_1, p_input_ids_2, p_attention_mask_2, labels]
        return dataset



class DataModule(pl.LightningDataModule):
    def __init__(self, task_name, pmodel_type,prot_model_name, num_workers, batch_size, prot_maxLength,
                 traindata_rate=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_name = task_name
        self.prot_maxLength = prot_maxLength
        self.traindata_rate = traindata_rate
        self.pmodel_type = pmodel_type

        if pmodel_type == 'esm1' or pmodel_type == 'esm-1b':
            p_model, alphabet = esm.pretrained.load_model_and_alphabet_local(prot_model_name)
            self.p_tokenizer = alphabet.get_batch_converter()
            num_esm_layers = len(p_model.layers)
        elif pmodel_type == 'protbert' or pmodel_type == 'protbert-bfd':
            self.p_tokenizer = AutoTokenizer.from_pretrained(prot_model_name)

        self.df_train = None
        self.df_val = None
        self.df_test = None

        self.load_testData = True

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def get_task(self, task_name):
        if task_name.lower() == 'intact':
            return 'data/intAct/process'

    def prepare_data(self):
        dataFolder = self.get_task(self.task_name)

        self.df_train = pd.read_csv(dataFolder + '/data_train.csv')
        self.df_val = pd.read_csv(dataFolder + '/data_val.csv')

        traindata_length = int(len(self.df_train) * self.traindata_rate)
        validdata_length = int(len(self.df_val) * self.traindata_rate)

        self.df_train = self.df_train[:traindata_length]
        self.df_val = self.df_val[:validdata_length]

        if self.load_testData is True:
            self.df_test = pd.read_csv(dataFolder + '/data_test.csv')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self.pmodel_type == 'esm1' or self.pmodel_type == 'esm-1b':
                self.train_dataset = ESM_Dataset(self.df_train.index.values, self.df_train.Label.values, self.df_train,
                                             self.p_tokenizer, self.prot_maxLength)
                self.valid_dataset = ESM_Dataset(self.df_val.index.values, self.df_val.Label.values, self.df_val,
                                             self.p_tokenizer, self.prot_maxLength)
            elif self.pmodel_type == 'protbert' or self.pmodel_type == 'esm-protbert-bfd':
                self.train_dataset = PB_Dataset(self.df_train.index.values, self.df_train.Label.values, self.df_train,
                                             self.p_tokenizer, self.prot_maxLength)
                self.valid_dataset = PB_Dataset(self.df_val.index.values, self.df_val.Label.values, self.df_val,
                                             self.p_tokenizer, self.prot_maxLength)

        if self.load_testData is True:
            if self.pmodel_type == 'esm1' or self.pmodel_type == 'esm-1b':
                self.test_dataset = ESM_Dataset(self.df_test.index.values, self.df_test.Label.values, self.df_test,
                                        self.p_tokenizer, self.prot_maxLength)
            elif self.pmodel_type == 'protbert' or self.pmodel_type == 'esm-protbert-bfd':
                self.test_dataset = PB_Dataset(self.df_test.index.values, self.df_test.Label.values, self.df_test,
                                            self.p_tokenizer, self.prot_maxLength)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

