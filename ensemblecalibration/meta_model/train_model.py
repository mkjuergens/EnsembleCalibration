import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from ensemblecalibration.meta_model.losses import SKCELoss, MMDLoss, LpLoss
from ensemblecalibration.config import create_config_binary_classification
from ensemblecalibration.data.experiments import get_experiment
from ensemblecalibration.meta_model.train import train_mlp
from ensemblecalibration.meta_model.mlp_model import MLPCalW, MLPCalWLightning



if __name__ == "__main__":
    config_default = create_config_binary_classification(exp_name="gp")

    loader, dataset = get_experiment(config=config_default, h0=True, batch_size=128)
    model = MLPCalW(in_channels=1, out_channels=1, hidden_dim=8, hidden_layers=1, use_relu=True)
    loss = LpLoss(p=2, bw=0.01)
    model, loss_train = train_mlp(model, dataset, loss, n_epochs=100, lr=0.001,
                                   batch_size=128, optim=torch.optim.Adam, shuffle=True, patience=100)
    print(loss_train)

    # train with pytorch lightning
    model = MLPCalWLightning(loss_fct=loss, lr=0.001, in_channels=1, out_channels=1, hidden_dim=8, hidden_layers=1,
                              use_relu=True, use_scheduler=True)
    logger = WandbLogger(name="mlp_calibration_lp", project="ensemble-calibration-2")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(max_epochs=100, logger=logger, callbacks=[lr_monitor])
    trainer.fit(model, loader)
