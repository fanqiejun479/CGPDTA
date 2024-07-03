import wandb
import sys
from transformers import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner

sys.path.append('..')
from model.model import Model
from dataset.Dataset import DataModule
from utils.utils import *
from rdkit import RDLogger

logging.set_verbosity_error()
# 去除未使用pooler计算损失的警告
RDLogger.DisableLog('rdApp.*')
import warnings

warnings.filterwarnings(action='ignore')


def main_wandb(config=None):
    try:
        if config is not None:
            wandb.init(config=config, project=project_name)
        else:
            wandb.init(settings=wandb.Settings(console='off'))

        config = wandb.config
        pl.seed_everything(seed=config.num_seed)

        dm = DataModule(config.task_name,config.k_fold,
                        config.num_workers, config.batch_size, config.clique_type, config.split_type,
                        config.d_model_name,
                        config.p_model_name, config.drug_maxlength, config.prot_maxlength, config.pock_maxlength,
                        config.traindata_rate)
        dm.prepare_data()
        dm.setup()

        model_type = str(config.pretrained['chem']) + "To" + str(config.pretrained['prot'])
        model_logger = WandbLogger(project=project_name)

        # 设置早停
        early_stop_callback = EarlyStopping(monitor="valid_rmse", patience=10, mode="min")

        checkpoint_callback = ModelCheckpoint(
            f"{config.name}_k-fold-{config.k_fold}_{config.clique_type}_{config.split_type}_{config.graph_e_type}_{config.pocket_e_type}_{model_type}_{config.lr}_{config.num_seed}",
            save_top_k=1, monitor="valid_rmse", mode="min",
            filename='{epoch}_{valid_loss:.3f}_{valid_rmse:.4f}')


        trainer = pl.Trainer(devices=config.gpu_ids,
                             max_epochs=config.max_epoch,
                             min_epochs=10,
                             precision=16,
                             logger=model_logger,
                             callbacks=[checkpoint_callback,early_stop_callback],
                             accelerator='gpu',
                             strategy='ddp',
                             auto_lr_find=True
                             )

        if config.model_mode == "train":
            model = Model(config.graph_e_type, config.pocket_e_type, config.dmodel_type, config.pmodel_type, config.lr,
                config.loss_fn, config.batch_size,
                config.pock_maxlength,
                config.d_model_name, config.load_d_model, config.pretrained['chem'], config.p_model_name,
                config.load_p_model, config.pretrained['prot'])

            model.train()
            trainer.fit(model, datamodule=dm)

            model.eval()
            trainer.test(model, datamodule=dm)

        else:
            model = Model.load_from_checkpoint(config.load_checkpoint)

            model.eval()
            trainer.test(model, datamodule=dm)

    except Exception as e:
        print(e)


def main_default(config):
    try:
        config = DictX(config)
        pl.seed_everything(seed=config.num_seed)

        dm = DataModule(config.task_name, config.k_fold,
                        config.num_workers, config.batch_size, config.clique_type, config.split_type,
                        config.d_model_name,
                        config.p_model_name, config.drug_maxlength, config.prot_maxlength, config.pock_maxlength,
                        config.traindata_rate)
        dm.prepare_data()
        dm.setup()

        model_type = str(config.pretrained['chem']) + "To" + str(config.pretrained['prot'])
        model_logger = WandbLogger(project=project_name)

        # 设置早停
        early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.005, patience=10, verbose=10, mode="min")

        checkpoint_callback = ModelCheckpoint(
            f"{config.name}_k-fold-{config.k_fold}_{config.clique_type}_{config.split_type}_{config.graph_e_type}_{config.pocket_e_type}_{model_type}_{config.lr}_{config.num_seed}",
            save_top_k=3, monitor="valid_loss", mode="min")

        trainer = pl.Trainer(devices=config.gpu_ids,
                             max_epochs=config.max_epoch,
                             precision=16,
                             logger=model_logger,
                             callbacks=[checkpoint_callback, early_stop_callback],
                             accelerator='gpu',
                             strategy='ddp'
                             )

        if config.model_mode == "train":
            model = Model(config.graph_e_type, config.pocket_e_type, config.dmodel_type, config.pmodel_type, config.lr,
                          config.loss_fn, config.batch_size,
                          config.pock_maxlength,
                          config.d_model_name, config.load_d_model, config.pretrained['chem'], config.p_model_name,
                          config.load_p_model, config.pretrained['prot'])
            model.train()
            trainer.fit(model, datamodule=dm)

            model.eval()
            trainer.test(model, datamodule=dm)

        else:
            model = Model.load_from_checkpoint(config.load_checkpoint)

            model.eval()
            trainer.test(model, datamodule=dm)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    using_wandb = True
    os.environ["WANDB_API_KEY"] = 'b31033b58bceeed82a4032a14b8a1f9dcffbf370'
    os.environ["WANDB_MODE"] = "offline"

    if using_wandb == True:
        # -- hyper param config file Load --##
        # 离线状态下可运行
        TRANSFORMERS_OFFLINE = 1
        config = load_hparams('config/config_hparam.json')
        project_name = config["name"]
        main_wandb(config)
    else:
        config = load_hparams('config/config_hparam.json')
        main_default(config)
