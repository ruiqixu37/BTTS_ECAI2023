import hfai_env
import hfai
from hfai.pl import ModelCheckpointHF
from hfai.pl import HFAIEnvironment
import numpy as np
import torch
from transformers import T5TokenizerFast
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import copy
import os

from data import AmazonDataModule, get_raw_data
from model import (T5ForConditionalGenerationWithExtractor,
                   TextSettrModel)


import warnings
warnings.filterwarnings("ignore")

hfai_env.set_env('barlowtwins')

if __name__ == '__main__':
    """# 1. Prepare Data"""
    print(torch.__version__)
    print(torch.version.cuda)
    # set random seed
    rand_seed = 123
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)

    # hyperparameters
    sent_length = 32
    batch_size = 64
    lambda_factor = 1

    tokenizer = T5TokenizerFast.from_pretrained("./pretrained_model/t5-base")
    raw_data = get_raw_data()

    model = T5ForConditionalGenerationWithExtractor.from_pretrained(
        "./pretrained_model/t5-base-with-extractor")
    model.extractor = copy.deepcopy(model.encoder)
    model.extractor.is_extractor = True
    model.lambda_factor = lambda_factor

    module = AmazonDataModule(raw_data, batch_size, tokenizer, sent_length)

    # training loop
    for lambda_val in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1.5, 2]:
        # 加入幻方AI api
        output_dir = 'hfai_out'
        cb = ModelCheckpointHF(dirpath=output_dir)

        model = TextSettrModel(lambda_val, sent_length, tokenizer)
        # checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=root, filename='{epoch}')
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
        logger = TensorBoardLogger("logs", name="style_transfer")
        # trainer = Trainer(max_epochs = 10, gpus=1, default_root_dir=root, val_check_interval=0.25, precision=32, logger=logger, resume_from_checkpoint = f"{root}/barlow-twins-10-hour.ckpt")
        trainer = Trainer(max_epochs=10, gpus=8, default_root_dir="", val_check_interval=0.25,
                          precision=32, logger=logger, plugins=[HFAIEnvironment()], callbacks=[cb])

        model = hfai.pl.nn_to_hfai(model)  # 替换成幻方算子

        ckpt_path = f'{output_dir}/barlow-twins-lambda-{lambda_val}-{cb.CHECKPOINT_NAME_SUSPEND}.ckpt'
        ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
        trainer.fit(
            model,
            module,
            ckpt_path=ckpt_path  # 自动恢复训练
        )
