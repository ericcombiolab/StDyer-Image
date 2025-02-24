from typing import List, Optional
import os
import os.path as osp
import sys
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler, PyTorchProfiler
from pytorch_lightning.utilities import rank_zero_only
from src import utils
import torch
import torch.distributed as dist

log = utils.get_logger(__name__)

def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    # if config.get("seed"):
        # seed_everything(config.seed, workers=True)
    seed_everything(config.datamodule.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    # if torch.__version__.startswith("2."):
    #     model = torch.compile(model)

    # Init lightning loggers
    if "hkbugpusrv" in os.uname()[1]:
        gpu_idx = os.uname()[1].lstrip("hkbugpusrv0")
        if "comet" in config.logger:
            config.logger.comet.offline = True
    elif os.uname()[1] == "hkbugpudgx01":
        gpu_idx = "dgx1"
        if "comet" in config.logger:
            config.logger.comet.offline = True
    else:
        gpu_idx = None
    logger: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning callbacks
    if "model_checkpoint" in config.callbacks:
        config.callbacks.model_checkpoint.dirpath = osp.join(config.paths.log_dir, f"{logger[0].version}/checkpoints")
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    # profiler = AdvancedProfiler(dirpath="/home/comp/20481195/GMVGAT", filename="advanced_profiler")
    # profiler = SimpleProfiler(dirpath="/home/comp/20481195/GMVGAT", filename="simple_profiler")
    # profiler = PyTorchProfiler(dirpath="/home/comp/20481195/GMVGAT", filename="pytorch_profiler", sort_by_key="cuda_memory_usage", profile_memory=True)
    # profiler = PyTorchProfiler(dirpath="/home/comp/20481195/GMVGAT", filename="pytorch_profiler", sort_by_key="cpu_time_total", profile_memory=False)
    # Init lightning trainer
    reload_dataloader = 1 if config.datamodule.max_dynamic_neigh else 0
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    if model.exp_rec_type == "Gaussian":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_properties = torch.cuda.get_device_properties(device)
            # Check if the GPU supports bf16
            if hasattr(gpu_properties, 'major') and gpu_properties.major >= 8:
                # print("This GPU supports bf16.")
                precision = "bf16-mixed"
            else:
                precision = "16-mixed"
        else:
            precision = "bf16-mixed"
            # import cpuinfo
            # cpu_info = cpuinfo.get_cpu_info()
            # if 'bf16' in cpu_info['flags']:
            #     precision = "bf16-mixed"
            # else:
            #     precision = "16-mixed"
    elif model.exp_rec_type == "NegativeBinomial":
        precision = "16-mixed"
    trainer: Trainer = hydra.utils.instantiate(
        # config.trainer, callbacks=callbacks, logger=logger, precision=16, profiler=profiler, _convert_="partial"
        # config.trainer, callbacks=callbacks, logger=logger, precision="bf16", _convert_="partial"
        # config.trainer, callbacks=callbacks, logger=logger, precision="16-mixed", reload_dataloaders_every_n_epochs=reload_dataloader, _convert_="partial"
        config.trainer, callbacks=callbacks, logger=logger, precision=precision, reload_dataloaders_every_n_epochs=reload_dataloader, _convert_="partial"
        # config.trainer, callbacks=callbacks, logger=logger, precision=16, reload_dataloaders_every_n_epochs=reload_dataloader, _convert_="partial", profiler=profiler
    )
    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model, datamodule=datamodule, min_lr=1e-4, max_lr=1e-1, num_training=1000)
    # config.model.lr = lr_finder.suggestion()
    # Send some parameters from config to all lightning loggers
    if trainer.logger is not None:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(
            config=config,
            model=model,
            datamodule=datamodule,
            trainer=trainer,
            callbacks=callbacks,
            logger=logger,
        )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)
    log_versions = [trainer.model.stored_version]
    ban_ckpt_list = []
    last_normal_epoch_dict = trainer.model.last_normal_epoch_dict.copy()
    if "comet" in config.logger:
        while trainer.model.resume_from_ckpt_and_continue_train is True:
            trainer.model.resume_from_ckpt_and_continue_train = False
            trainer.should_stop = False
            logger: List[Logger] = []
            config.logger.comet.experiment_key = log_versions[0]
            trainer.model.stored_version = None
            if "logger" in config:
                for _, lg_conf in config.logger.items():
                    if "_target_" in lg_conf:
                        log.info(f"Instantiating logger <{lg_conf._target_}>")
                        logger.append(hydra.utils.instantiate(lg_conf))
            trainer.logger = logger[0]
            if len(trainer.model.last_normal_epoch_dict) == 0:
                trainer.fit(model=trainer.model, datamodule=trainer.datamodule)
            else:
                usable_ckpt_epoch_list = []
                for last_normal_epoch, run_times in last_normal_epoch_dict.items():
                    if last_normal_epoch in ban_ckpt_list:
                        continue
                    elif run_times >= 2:
                        ban_ckpt_list.append(last_normal_epoch)
                        continue
                    else:
                        usable_ckpt_epoch_list.append(last_normal_epoch)
                if len(usable_ckpt_epoch_list) == 0:
                    from collections import OrderedDict
                    trainer.model.last_normal_epoch_dict = OrderedDict()
                    last_normal_epoch_dict = OrderedDict()
                    trainer.model.stored_version = None
                    ban_ckpt_list = []
                    continue
                else:
                    trainer.fit(model=trainer.model, datamodule=trainer.datamodule, ckpt_path=osp.join(config.logger.comet.save_dir, log_versions[0], "checkpoints", f"epoch_{usable_ckpt_epoch_list[-1]}.ckpt"))
                    last_normal_epoch_dict[usable_ckpt_epoch_list[-1]] += 1
            if trainer.model.stored_version is None:
                # reached the end of training (like max_epochs)
                break
            log_versions.append(trainer.model.stored_version)

    # Get metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)

    # Test the model
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # log.info("torch rank", dist.get_rank())
    # print("lightning rank", trainer.global_rank)
    # print("lightning is_global_zero", trainer.is_global_zero)

    # Print path to best checkpoint
    # if not config.trainer.get("fast_dev_run"):
    #     log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    return score
