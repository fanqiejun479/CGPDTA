from model.PPI import PPI_ESM
from model.PPI_protbert import PPI_ProtBert
from transformers import AutoConfig, BertModel
import esm
import torch.nn as nn


class Protein_Encoder(nn.ModuleDict):
    def __init__(self, pmodel_type,prot_model_name, load_p_model, p_pretrained):
        super().__init__()
        self.pmodel_type = pmodel_type
        if pmodel_type == 'esm1' or pmodel_type == 'esm-1b':
            if p_pretrained == "True":
                model = PPI_ESM.load_from_checkpoint(load_p_model)
                self.p_model = nn.Sequential(model.p_model)[0]
            else:
                self.p_model, _ = esm.pretrained.load_model_and_alphabet_local(prot_model_name)
            self.num_esm_layers = 12
        elif pmodel_type == 'protbert' or pmodel_type == 'protbert-bfd':
            prot_config = AutoConfig.from_pretrained(prot_model_name)
            if p_pretrained == "True":
                model = PPI_ProtBert.load_from_checkpoint(load_p_model)
                self.p_model = nn.Sequential(model.p_model)[0]
            else:
                self.p_model = BertModel.from_pretrained(prot_model_name,
                                                         output_hidden_states=True,
                                                         output_attentions=True)

    def forward(self, prot_inputs):
        if self.pmodel_type == 'esm1' or self.pmodel_type == 'esm-1b':
            prot_outputs = self.p_model(prot_inputs, repr_layers=[self.num_esm_layers])
            prot_outputs = prot_outputs["representations"][self.num_esm_layers]
            prot_outputs = prot_outputs[:, 1: prot_outputs.shape[1]].mean(1)
            prot_outputs = prot_outputs.view(prot_outputs.size(0), 32, -1)
        elif self.pmodel_type == 'protbert' or self.pmodel_type == 'protbert-bfd':
            prot_outputs = ''
        return prot_outputs
