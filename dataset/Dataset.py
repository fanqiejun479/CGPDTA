import ast
from torch_geometric.data import DataLoader
import torch_geometric
from torch_geometric.data import InMemoryDataset
from transformers import AutoTokenizer
import pytorch_lightning as pl
import pandas as pd
import sys
import esm
from tqdm import tqdm
import torch.nn as nn

from utils.utils import *

RDLogger.DisableLog('rdApp.*')
import warnings

warnings.filterwarnings(action='ignore')


class Dataset(InMemoryDataset):
    def __init__(self,k_fold, root=None, dataset=None, data=None,data_indices=None, smile_graph=None, clique_type=None, split_type=None,
                 d_tokenizer=None,
                 p_tokenizer=None, drug_maxLength=None, prot_maxLength=None, pock_maxLength=None, transform=None):

        super(Dataset, self).__init__(root, transform)
        self.k_fold = k_fold
        self.dataset = dataset
        self.data = data
        self.data_indices = data_indices
        self.clique_type = clique_type
        self.split_type = split_type
        self.d_tokenizer = d_tokenizer
        self.p_tokenizer = p_tokenizer
        self.drug_maxLength = drug_maxLength
        self.prot_maxLength = prot_maxLength
        self.pock_maxLength = pock_maxLength
        if 'test' in self.dataset:
            if os.path.isfile(self.processed_paths[1]):
                print('Preprocessed data found: {}, loading ...'.format(self.processed_paths[1]))
                self.data, self.slices = torch.load(self.processed_paths[1])
            else:
                print('Preprocessed data {} not found, doing preprocessing...'.format(self.processed_paths[1]))
                self.process(data,data_indices, smile_graph, drug_maxLength, prot_maxLength, pock_maxLength, clique_type,self.processed_paths[1])
                self.data, self.slices = torch.load(self.processed_paths[1])
        else:
            if os.path.isfile(self.processed_paths[0]):
                print('Preprocessed data found: {}, loading ...'.format(self.processed_paths[0]))
                self.data, self.slices = torch.load(self.processed_paths[0])
            else:
                print('Preprocessed data {} not found, doing preprocessing...'.format(self.processed_paths[0]))
                self.process(data,data_indices, smile_graph, drug_maxLength, prot_maxLength, pock_maxLength, clique_type,self.processed_paths[0])
                self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + str(self.k_fold) + self.clique_type + self.split_type + '.pt',self.dataset + self.clique_type + self.split_type + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, data,data_indices, smile_graph, drug_maxLength, prot_maxLength, pock_maxLength, clique_type,path):
        xd = data['compound_iso_smiles']
        xt = data['target_sequence']
        xp = data['target_pocket']
        pdb_list = data['PDB_id']
        y = data['Label']
        assert (len(xd) == len(xt) and len(xt) == len(xp) and len(xp) == len(pdb_list) and len(pdb_list) == len(
            y)), "The five lists must be the same length!"
        data_list = []
        for i in tqdm(data_indices, desc='Generating Graph file...', total=len(data_indices)):
            smiles = xd[i]
            prot = xt[i]
            pocket = xp[i]
            pdb = pdb_list[i]
            pdb = str(pdb).lower()
            label = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size = smile_graph[smiles][0]
            features = np.array(smile_graph[smiles][1])
            edge_index = smile_graph[smiles][2]
            edge_attr = np.array(smile_graph[smiles][3])

            # make the graph ready for PyTorch Geometrics GCN algorithms:
            try:
                edge_index = torch.LongTensor(edge_index).transpose(1, 0)
            except:
                edge_index = torch.LongTensor(edge_index)

            smiles_input = self.d_tokenizer(smiles, padding='max_length', max_length=drug_maxLength, truncation=True,
                                            return_tensors="pt")
            if len(prot) > prot_maxLength:
                prot = prot[:prot_maxLength]
            _, _, protein_input = self.p_tokenizer([("", prot)])
            pad = nn.ZeroPad2d(padding=(0, self.prot_maxLength - int(protein_input.shape[1]), 0, 0))
            protein_input = pad(protein_input)
            pocket = seq_cat(pocket, pock_maxLength)

            GCNData = torch_geometric.data.Data(x=torch.Tensor(features),
                                                edge_index=edge_index,
                                                edge_attr=torch.Tensor(edge_attr),
                                                y=torch.FloatTensor([label]))
            GCNData.smiles_input_id = smiles_input['input_ids']
            GCNData.smiles_attention_mask = smiles_input['attention_mask']
            GCNData.smiles = smiles
            GCNData.protein = protein_input
            GCNData.protein_seq = prot
            GCNData.pocket_seq = pocket
            GCNData.pocket = torch.LongTensor([pocket])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), path)


class DataModule(pl.LightningDataModule):
    def __init__(self, task_name,k_fold, num_workers, batch_size, clique_type, split_type, drug_model_name, prot_model_name,
                 drug_maxLength, prot_maxLength, pock_maxlength,
                 traindata_rate=1.0):
        super().__init__()
        self.task_name = task_name
        self.k_fold = k_fold
        self.dataFolder = self.get_task(self.task_name)
        self.batch_size = batch_size
        self.clique_type = clique_type
        self.split_type = split_type
        self.num_workers = num_workers
        self.drug_maxLength = drug_maxLength
        self.prot_maxLength = prot_maxLength
        self.pock_maxlength = pock_maxlength
        self.traindata_rate = traindata_rate

        self.d_tokenizer = AutoTokenizer.from_pretrained(drug_model_name)
        esm_bert, alphabet = esm.pretrained.load_model_and_alphabet_local(prot_model_name)
        self.p_tokenizer = alphabet.get_batch_converter()

        self.df_train = None
        self.df_val = None
        self.df_test = None

        self.load_testData = True

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def get_task(self, task_name):
        if task_name.lower() == 'pdbbind':
            return 'data/PDBBind/process'

    def get_k_fold_data(self, k, i, data, split_type):  ###此过程主要是步骤（1）
        # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
        assert k > 1
        fold_size = len(data) // k  # 每份的个数:数据总条数/折数（组数）

        if split_type == 'random':
            np.random.seed(42)
            indices = data.index.tolist()
            val_idx = indices[fold_size * (i - 1):fold_size * i]
            train_idx = indices[:fold_size * (i - 1)] + indices[fold_size * i:]

        if split_type == 'cold_d':
            stype_list = data['ligand_name']
            all_stype = {}
            for j, stype in enumerate(stype_list):
                if stype not in all_stype:
                    all_stype[stype] = [j]
                else:
                    all_stype[stype].append(j)

            # sort from largest to smallest sets
            # all_stype = {key: sorted(value) for key, value in all_stype.items()}
            # all_stype_sets = [
            #     stype_set for (stype, stype_set) in sorted(
            #         all_stype.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=False)
            # ]  # 根据id列表的长度和列表中最大id进行升序排列

            long_li = []
            short_li = []
            for key, value in all_stype.items():
                if len(value) >= 10:
                    long_li.append(key)
                else:
                    short_li.append(key)
            print(len(long_li))
            print(len(short_li))
            all_stype_sets = []
            long_num = 0
            key_li = [key for key in all_stype.keys()]
            print(len(key_li))
            key_a = []
            for index in range(len(key_li)):
                if index - long_num == len(short_li):
                    break
                else:
                    if index < 150 * (long_num + 1):
                        key_a.append(short_li[index - long_num])
                        all_stype_sets.append(all_stype[short_li[index - long_num]])
                    else:
                        key_a.append(long_li[long_num])
                        all_stype_sets.append(all_stype[long_li[long_num]])
                        long_num += 1
            all_stype_sets.append(all_stype[list(set(key_li) - set(key_a))[0]])
            # get train, valid test indices
            val_cutoff = 0.2 * len(stype_list)
            train_idx, val_idx = [], []
            loc = 0
            for idx in range(len(all_stype_sets)):
                stype_set = all_stype_sets[idx]
                if idx == 0:
                    loc = 0
                else:
                    loc += len(all_stype_sets[idx - 1])
                if len(val_idx) + len(
                        stype_set) > val_cutoff and loc > val_cutoff * i:  # 如果存在某个骨架的id列表长度加已创建的训练集大于训练集需要的数据个数
                    train_idx.extend(stype_set)  # 一次性添加列表中多个值
                elif len(val_idx) + len(stype_set) < val_cutoff and loc < val_cutoff * (i - 1):
                    train_idx.extend(stype_set)  # 一次性添加列表中多个值
                else:
                    val_idx.extend(stype_set)

            assert len(set(train_idx).intersection(set(val_idx))) == 0  # 检测是否存在交集
            assert len(val_idx) != 0
            assert len(train_idx) + len(val_idx) == len(data)

        if split_type == 'cold_p':
            stype_list = data['target_name']
            all_stype = {}
            for j, stype in enumerate(stype_list):
                if stype not in all_stype:
                    all_stype[stype] = [j]
                else:
                    all_stype[stype].append(j)

            # sort from largest to smallest sets
            all_stype = {key: sorted(value) for key, value in all_stype.items()}
            all_stype_sets = [
                stype_set for (stype, stype_set) in sorted(
                    all_stype.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
            ]  # 根据id列表的长度和列表中最大id进行升序排列

            # get train, valid test indices
            val_cutoff = 0.2 * len(stype_list)
            train_idx, val_idx = [], []
            loc = 0
            for idx in range(len(all_stype_sets)):
                stype_set = all_stype_sets[idx]
                if idx == 0:
                    loc = 0
                else:
                    loc += len(all_stype_sets[idx - 1])
                if len(val_idx) + len(
                        stype_set) > val_cutoff and loc > val_cutoff * i:  # 如果存在某个骨架的id列表长度加已创建的训练集大于训练集需要的数据个数
                    train_idx.extend(stype_set)  # 一次性添加列表中多个值
                elif len(val_idx) + len(stype_set) < val_cutoff and loc < val_cutoff * (i - 1):
                    train_idx.extend(stype_set)  # 一次性添加列表中多个值
                else:
                    val_idx.extend(stype_set)

            assert len(set(train_idx).intersection(set(val_idx))) == 0  # 检测是否存在交集
            assert len(val_idx) != 0
            assert len(train_idx) + len(val_idx) == len(data)

        return train_idx, val_idx

    def prepare_data(self):
        # Use this method to do things that might write to disk or that need to be done only from
        # a single process in distributed settings.

        ## 前处理数据,获取graph_dict
        self.df = pd.read_csv(self.dataFolder + '/data_all.csv')
        self.df_train_indices,self.df_val_indices = self.get_k_fold_data(5, self.k_fold, self.df,self.split_type)
        compound_iso_smiles = list(self.df['compound_iso_smiles'])
        pdb = list(self.df['target_name'])

        processed_path = 'data/' + self.task_name + '/process/processed/' + self.task_name + '_train_'+ str(self.k_fold) + self.clique_type + self.split_type + '.pt'
        graph_path = 'data/' + self.task_name + '/process/processed/' + self.task_name + '_' + self.clique_type + '.txt'
        if os.path.isfile(processed_path):
            print('Preprocessed data found: {}, loading ...'.format(processed_path))
            smile_graph = {}
        else:
            if os.path.isfile(graph_path):
                print('Graph_dict data found: {}, loading ...'.format(graph_path))
                smile_graph = {}
                with open(graph_path, 'r') as f:
                    for line in f:
                        line = line.split(':')
                        s = line[0]
                        g = ast.literal_eval(line[1])
                        smile_graph[s] = g
            else:
                print('Graph_dict data  not found: {}, doing generating ...'.format(graph_path))
                pdb_dict = {}
                for i in range(len(compound_iso_smiles)):
                    pdb_dict[compound_iso_smiles[i]] = pdb[i]
                compound_iso_smiles = set(compound_iso_smiles)
                smile_graph = {}
                for smile in tqdm(list(compound_iso_smiles), desc='generating graph dict...',
                                  total=len(list(compound_iso_smiles))):
                    pdb = pdb_dict[smile]
                    if self.clique_type == 'None':
                        g = smile_to_graph(smile, pdb)
                    else:
                        g = smile_to_clique_graph(smile, pdb, self.clique_type)
                    smile_graph[smile] = g
                # graph_dict保存为txt
                filename = open(graph_path, 'w')
                for s, g in smile_graph.items():
                    filename.write(s + ':' + str(g))
                    filename.write('\n')
                filename.close()

        self.smile_graph = smile_graph


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Dataset(self.k_fold,self.dataFolder, str(self.task_name) + '_train_', self.df,self.df_train_indices,
                                         self.smile_graph,
                                         self.clique_type, self.split_type, self.d_tokenizer, self.p_tokenizer,
                                         self.drug_maxLength, self.prot_maxLength, self.pock_maxlength)
            self.valid_dataset = Dataset(self.k_fold,self.dataFolder, str(self.task_name) + '_val_', self.df, self.df_val_indices,
                                         self.smile_graph,
                                         self.clique_type, self.split_type, self.d_tokenizer, self.p_tokenizer,
                                         self.drug_maxLength, self.prot_maxLength, self.pock_maxlength)

        if self.load_testData is True:
            self.test_dataset = Dataset(self.k_fold,self.dataFolder, str(self.task_name) + '_val_', self.df, self.df_val_indices,
                                        self.smile_graph,
                                        self.clique_type, self.split_type, self.d_tokenizer, self.p_tokenizer,
                                        self.drug_maxLength, self.prot_maxLength, self.pock_maxlength)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

