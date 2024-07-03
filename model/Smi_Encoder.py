from model.CCI import CCI
from transformers import RobertaModel
import torch.nn as nn


class Smi_Encoder(nn.ModuleDict):
    def __init__(self, drug_model_name, load_d_model, d_pretrained):
        super().__init__()
        if d_pretrained == "True":
            model = CCI.load_from_checkpoint(load_d_model)
            self.d_model = nn.Sequential(model.d_model)[0]
        else:
            self.d_model = RobertaModel.from_pretrained(drug_model_name, num_labels=2,
                                                        output_hidden_states=True,
                                                        output_attentions=True)

    def forward(self, input_id, attention_mask):
        drug_outputs = self.d_model(input_id, attention_mask)
        drug_outputs = drug_outputs.last_hidden_state[:, 0]
        drug_outputs = drug_outputs.view(drug_outputs.size(0), 32, -1)

        return drug_outputs
