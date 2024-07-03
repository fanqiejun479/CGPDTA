import pytorch_lightning as pl
import pandas as pd
import openpyxl
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim import lr_scheduler

from utils.metrics import *
from model.Smi_Encoder import Smi_Encoder
from model.Protein_Encoder import Protein_Encoder
from model.Graph_Encoder import Graph_Encoder
from model.Pocket_Encoder import Pocket_Encoder


class Feature_fusion(nn.Module):
    def __init__(self, channels, inter_channels):
        super().__init__()

        self.d_attention = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.p_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 自适应平均池化，对最后一维转换为指定维度
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, d_input, p_input):
        x = d_input + p_input
        xd = self.d_attention(x)
        xp = self.p_attention(x)
        x = xd + xp
        weight = self.sigmoid(x)

        dp_output = 2 * d_input * weight + 2 * p_input * (1 - weight)

        return dp_output


class Model(pl.LightningModule):
    def __init__(self, ge_type, pe_type, dmodel_type, pmodel_type, lr, loss_fn, batch_size, pock_len, drug_model_name,
                 load_d_model, d_pretrained,
                 prot_model_name, load_p_model, p_pretrained):
        super().__init__()
        self.ge_type = ge_type
        self.pe_type = pe_type
        self.dmodel_type = dmodel_type
        self.pmodel_type = pmodel_type
        self.lr = lr
        self.loss_fn = loss_fn
        self.criterion = torch.nn.SmoothL1Loss()
        self.batch_size = batch_size
        self.pock_len = pock_len
        self.li = nn.Linear(26,2)
        # self.sigmoid = nn.Sigmoid()

        '''---drug branch---'''
        # feature branch
        self.smi_encoder = Smi_Encoder(drug_model_name=drug_model_name, load_d_model=load_d_model,
                                       d_pretrained=d_pretrained)

        # graph branch
        self.graph_encoder = Graph_Encoder(encoder_type=self.ge_type, layer_num=5, in_channels=130, basic_channels=32,
                                           dropout=0.2)

        '''---target branch---'''
        # ---sequence branch---
        self.protein_encoder = Protein_Encoder(pmodel_type=self.pmodel_type, prot_model_name=prot_model_name,
                                               load_p_model=load_p_model,
                                               p_pretrained=p_pretrained)

        # ---pocket branch---
        self.pocket_encoder = Pocket_Encoder(encoder_type=self.pe_type, layer_num=5, vocab_size=26, embedding_size=72,
                                             basic_channels=32, kernel_size=3,
                                             stride=1, padding=0, innum_layers=2, batch_size=self.batch_size,
                                             pock_len=self.pock_len)

        '''---combined layers---'''
        self.cancate_block = Feature_fusion(channels=32, inter_channels=64)

        '''---decoder---'''
        self.decoder = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1))

        self.save_hyperparameters()

    def forward(self, input_id, attention_mask, x, edge_index, edge_attr, batch, prot_data, pock_data):
        d_outs_1 = self.smi_encoder(input_id, attention_mask)# torch.Size([batch_size, 32, 24])
        d_outs_2 = self.graph_encoder(x, edge_index, edge_attr, batch)  # torch.Size([batch_size, 32, 2])
        p_outs_1 = self.protein_encoder(prot_data)  # torch.Size([batch_size, 32, 24])
        p_outs_2 = self.pocket_encoder(pock_data)  # torch.Size([batch_size, 32, 2])

        d_outs = torch.cat((d_outs_1, d_outs_2), 2)
        d_outs = self.li(d_outs)
        p_outs = torch.cat((p_outs_1, p_outs_2), 2)
        p_outs = self.li(p_outs)
        # ---feature cancatenate---
        dp_outs = self.cancate_block(d_outs, p_outs)
        dp_outs = dp_outs.view(dp_outs.size(0), -1)  # torch.Size([batch_size, 64])

        outs = self.decoder(dp_outs)
        logits = outs.squeeze(dim=1)

        return logits

    def training_step(self, batch, batch_idx):
        input_id, attention_mask, x, edge_index, edge_attr, batch_index, prot_data, pock_data = batch.smiles_input_id, batch.smiles_attention_mask, batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.protein, batch.pocket
        labels = batch.y

        logits = self(input_id, attention_mask, x, edge_index, edge_attr, batch_index, prot_data, pock_data)

        loss = self.criterion(logits, labels)

        self.log("train_loss", loss, on_step=False, on_epoch=True,batch_size=self.batch_size)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        input_id, attention_mask, x, edge_index, edge_attr, batch_index, prot_data, pock_data = batch.smiles_input_id, batch.smiles_attention_mask, batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.protein, batch.pocket
        labels = batch.y

        logits = self(input_id, attention_mask, x, edge_index, edge_attr, batch_index, prot_data, pock_data)
        loss = self.criterion(logits, labels)

        self.log("valid_loss", loss, on_step=False, on_epoch=True, logger=True, batch_size=self.batch_size)
        return {"logits": logits, "labels": labels}

    def validation_step_end(self, outputs):
        return {"logits": outputs['logits'], "labels": outputs['labels']}

    def validation_epoch_end(self, outputs):
        preds = self.convert_outputs_to_preds(outputs)
        labels = torch.as_tensor(torch.cat([output['labels'] for output in outputs], dim=0), dtype=torch.int)

        mse, rmse, CI, r2, rm2, pearson, spearman = self.regress_score(preds, labels)

        print(f'Val MSE_score : {mse}')
        print(f'Val RMSE_score : {rmse}')
        print(f'Val CI_score : {CI}')
        print(f'Val r2_score : {r2}')
        print(f'Val rm2_score : {rm2}')
        print(f'Val pearson_score : {pearson}')
        print(f'Val spearman_score : {spearman}')

        self.log("valid_mse", mse, on_step=False, on_epoch=True, logger=True)
        self.log("valid_rmse", rmse, on_step=False, on_epoch=True, logger=True)
        self.log("valid_CI", CI, on_step=False, on_epoch=True, logger=True)
        self.log("valid_r2", r2, on_step=False, on_epoch=True, logger=True)
        self.log("valid_rm2", rm2, on_step=False, on_epoch=True, logger=True)
        self.log("valid_pearson", pearson, on_step=False, on_epoch=True, logger=True)
        self.log("valid_spearman", spearman, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        input_id, attention_mask, x, edge_index, edge_attr, batch_index, prot_data, pock_data = batch.smiles_input_id, batch.smiles_attention_mask, batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.protein, batch.pocket
        smiles = batch.smiles
        labels = batch.y
        logits = self(input_id, attention_mask, x, edge_index, edge_attr, batch_index, prot_data, pock_data)

        return {"smiles": smiles, "logits": logits, "labels": labels}

    def test_step_end(self, outputs):
        return {"smiles": outputs['smiles'],"logits": outputs['logits'],"labels": outputs['labels']}

    def test_epoch_end(self, outputs):
        smiles_li = []
        prot_li = []
        pock_li = []
        preds_li = []
        labels_li = []
        for i in range(len(outputs)):
            smiles = outputs[i]['smiles']
            preds = outputs[i]['logits']
            labels = outputs[i]['labels']
            device = preds.device
            preds = preds.tolist()
            labels = labels.tolist()
            smiles_li.extend(smiles)
            preds_li.extend(preds)
            labels_li.extend(labels)

        df = pd.DataFrame(columns=['compound_iso_smiles','target_sequence','target_pocket','Label','predict_score'])
        df['predict_score'] = preds_li
        df['compound_iso_smiles'] = smiles_li
        df['Label'] = labels_li

        # 保存到本地excel
        pd.DataFrame(df).to_csv('',  encoding='utf-8')


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
        )

        return optimizer

    def convert_outputs_to_preds(self, outputs):
        logits = torch.cat([output['logits'] for output in outputs], dim=0)
        return logits

    def regress_score(self, preds, labels):
        y_pred = preds.detach().cpu().numpy()
        y_label = labels.detach().cpu().numpy()

        mse = get_mse(y_label, y_pred)
        rmse = get_rmse(y_label, y_pred)
        CI = get_cindex(y_label, y_pred)
        r2 = r_squared_error(y_label, y_pred)
        rm2 = get_rm2(y_label, y_pred)
        pearson = get_pearson(y_label, y_pred)
        spearman = get_spearman(y_label, y_pred)

        return mse, rmse, CI, r2, rm2, pearson, spearman
