import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import os
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint, get_fp32_state_dict_from_zero_checkpoint

from torch import distributed as dist
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.plugins.precision import DeepSpeedPrecisionPlugin, MixedPrecisionPlugin, PrecisionPlugin

import matplotlib.pyplot as plt
import numpy as np

from module.utils.data_module import fMRIDataModule
from module.pl_classifier import LitClassifier
from module.aurora_utils import xpu_intel # Aurora specific - 250306 jubin added
from module.aurora_utils.ddp_intel import MPIDDPStrategy, MPIEnvironment # Aurora specific - 250306 jubin added
from module.aurora_utils.deepspeed_intel import XPUDeepSpeedStrategy # Aurora specific - 250319 jubin added

from deepspeed.profiling.flops_profiler.profiler import get_model_profile

from deepspeed.accelerator import get_accelerator
import gc

# define model
class ModelWrapper(nn.Module):
    def __init__(self, model, mask):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.mask = mask

    def forward(self, x):
        return self.model(x, mask=self.mask)

def calculate_FLOPS_per_epoch(ckpt_path=None, use_deepspeed=False, args=None, output_file=None):
    device='xpu'
    with get_accelerator().device(0):
        if use_deepspeed:
            from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
            NEPTUNE_PROJECT_NAME = args.project_name
            if ckpt_path is not None:
                exp_id = ckpt_path.split("/")[-2]
                API_KEY = os.environ.get("NEPTUNE_API_TOKEN")
                neptune_kwargs = {
                    'api_token':API_KEY,
                    'project':NEPTUNE_PROJECT_NAME,
                    'with_id': exp_id,
                    'capture_stdout': False,
                    'capture_stderr': False,
                    'capture_hardware_metrics': False,
                    'capture_traceback': True,
                    } 
                
                logger = NeptuneLogger(**neptune_kwargs)
                neptune_run = logger.run
                args_orig = neptune_run['training']['hyperparams'].fetch() # dict format
                for key, value in args_orig.items():
                    if 'key' in ['num_nodes', 'devices', 'batch_size']:
                        continue
                    args.key = value
            args.strategy = 'deepspeed_stage_2'
            args.precision = 'bf16'
            # args.batch_size=2 # 250410 added

            data_module = fMRIDataModule(**vars(args))
            data_module.setup()
            data_module.prepare_data()
            train_iter_per_epoch = len(data_module.train_dataset)
            print(f"batch_size:{args.batch_size},train_iter_per_epoch:{train_iter_per_epoch}, patch_size:{args.patch_size}")

            model = LitClassifier(data_module = data_module, **vars(args))

            env = MPIEnvironment()
            # use PBS to manage processes
            if 'deepspeed' in args.strategy:
                precision_plugin = DeepSpeedPrecisionPlugin(precision=args.precision)
                stage = int(args.strategy.split('stage_')[-1][0]) if 'stage' in args.strategy else 1
                offload_kwargs = {}
                if 'offload' in args.strategy and stage >= 2:
                    offload_kwargs = {
                        'offload_optimizer': True,
                        'offload_parameters': True if stage == 3 else False,
                    }
                strategy = XPUDeepSpeedStrategy(
                    accelerator="xpu",
                    cluster_environment=env,
                    precision_plugin=precision_plugin,
                    process_group_backend='ccl',
                    stage=stage,
                    **offload_kwargs
                )
            limit_kwargs={
                'limit_train_batches':1,
                'limit_val_batches':1,
                'limit_test_batches':1
            } 
            trainer = pl.Trainer.from_argparse_args(
                args,
                strategy=strategy, # Aurora specific - 250314 jubin added
                gradient_clip_val=args.gradient_clip_val,
                gradient_clip_algorithm="norm",
                track_grad_norm=-1,
                accumulate_grad_batches=args.gradient_accumulation_steps,
                fast_dev_run=True,
                **limit_kwargs,
            )
            trainer.fit(model, datamodule=data_module) # to initialize deepspeed model
            # model = trainer.model.module

            batch_size = args.batch_size
            seq_len = args.sequence_length
            get_num_tokens = lambda batch_size, args, train_iter_per_epoch: (
                batch_size * (96/args.patch_size[0]) * (96/args.patch_size[1]) * (96/args.patch_size[2]) 
                * (args.sequence_length/args.patch_size[3]) * train_iter_per_epoch
                )
        else:
            if ckpt_path is not None:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                ckpt['hyper_parameters']['limit_validation_samples'] = None
                ckpt['hyper_parameters']['limit_test_samples'] = None

                args = ckpt['hyper_parameters']

            data_module = fMRIDataModule(**args)
            data_module.setup()
            data_module.prepare_data()
            train_iter_per_epoch = len(data_module.train_dataset)
            
            print(f"batch_size:{args['batch_size']},train_iter_per_epoch:{train_iter_per_epoch}, patch_size:{args['patch_size']}")

            model = LitClassifier(**args,data_module=data_module)
            batch_size = args['batch_size']
            seq_len = args['sequence_length']
            get_num_tokens = lambda batch_size, args, train_iter_per_epoch: (
                batch_size * (96/args['patch_size'][0]) * (96/args['patch_size'][1]) * (96/args['patch_size'][2])
                * (args['sequence_length']/args['patch_size'][3]) * train_iter_per_epoch 
                )

        # define mask for the model
        if os.environ['RANK'] == 0:
            import pdb;pdb.set_trace()
        else:
            import time
            time.sleep(10)
        mask = [] 
        mask.append(model.mask_generator().unsqueeze(0))
        mask = torch.vstack(mask).to(device)    # B D//p H//p W//p T//p 
        print(mask.shape)
        wrapped_model = ModelWrapper(model.model, mask)
        flops_per_batch, _, params = get_model_profile(
            model=wrapped_model,
            input_shape=(batch_size,1,96,96,96,seq_len),
            print_profile=True,
            detailed=True,
            warm_up=1,
            as_string=False,
            output_file=output_file,
            )
        FLOPS_per_epoch = flops_per_batch * train_iter_per_epoch 
        num_tokens = get_num_tokens(batch_size, args, train_iter_per_epoch)

        print("FLOPs_per_batch", flops_per_batch)
        print('FLOPS_per_epoch',FLOPS_per_epoch)
        print('num_tokens',num_tokens) 
        print(f"params: {params:.2e}")
        if output_file:
            with open(output_file, 'a') as f:
                f.write(f"ckpt_path: {ckpt_path}\n")
                f.write(f"FLOPS_per_epoch: {FLOPS_per_epoch:.2e}\n")
                f.write(f"num_tokens: {num_tokens:.2e}\n")
                f.write(f"params: {params:.2e}\n")

    try:
        del model
        del wrapped_model
        del mask
        del data_module
        1/0
    except:
        gc.collect()
        torch.cuda.empty_cache()

    return FLOPS_per_epoch, num_tokens, params
    # fwd flops per GPU # parameter
    # FLOPS is G(igabytes), params is M(millions)
    
def make_scientific_notation(num_tokens):
    # Express the number in scientific notation
    scientific_notation = "{:.2e}".format(num_tokens)

    # Replace 'e' with 'x10^' for the desired format
    #scientific_notation = scientific_notation.replace('e+', 'x10^')
    
    return scientific_notation


def cli_main():
    # ------------ args -------------
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=1234, type=int, help="random seeds. recommend aligning this argument with data split number to control randomness")
    parser.add_argument("--downstream_task", type=str, default="sex", help="downstream task")
    parser.add_argument("--downstream_task_type", type=str, default="default", help="select either classification or regression according to your downstream task")
    parser.add_argument("--classifier_module", default="default", type=str, help="A name of lightning classifier module (outdated argument)")
    parser.add_argument("--loggername", default="default", type=str, help="A name of logger")
    parser.add_argument("--project_name", default="default", type=str, help="A name of project (Neptune)")
    parser.add_argument("--resume_ckpt_path", type=str, help="A path to previous checkpoint. Use when you want to continue the training from the previous checkpoints")
    parser.add_argument("--load_model_path", type=str, help="A path to the pre-trained model weight file (.pth)")
    parser.add_argument("--valid_only", action='store_true', help="disable running _evaluate_metrics(mode='test') at validation stage")
    parser.add_argument("--test_only", action='store_true', help="specify when you want to test the checkpoints (model weights)")
    parser.add_argument("--load_ds_ckpt_manually", action='store_true', help="load deepspeed ckpt manually")
    parser.add_argument("--test_ckpt_path", type=str, help="A path to the previous checkpoint that intends to evaluate (--test_only should be True)")
    parser.add_argument("--freeze_feature_extractor", action='store_true', help="Whether to freeze the feature extractor (for evaluating the pre-trained weight)")
    parser.add_argument("--finetune_last_block", action='store_true', help="Whether to freeze the feature extractor (for evaluating the pre-trained weight)")
    parser.add_argument("--neptune_tags", nargs="+", default=[], type=str, help="neptune_tags")
    parser.add_argument("--limit_batches", action='store_true', help="option for debuggings")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="The number of gradient accmulation step. Effective Batch Size = gradient_accumulation_steps * args.batch_size")
    parser.add_argument("--save_every_checkpoints", action='store_true', help="option for save all weights of every epochs")
    parser.add_argument("--every_n_train_steps",type=int, default=None, help="Number of training steps between checkpoints.")
    temp_args, _ = parser.parse_known_args()

    # Set classifier
    Classifier = LitClassifier
    
    # Set dataset
    Dataset = fMRIDataModule
    
    # add two additional arguments
    parser = Classifier.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    _, _ = parser.parse_known_args()  # This command blocks the help message of Trainer class.
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    torch.xpu.manual_seed_all(args.seed) # Aurora specific - 250319 jubin added

    calculate_FLOPS_per_epoch(use_deepspeed=True, args=args, output_file="4.1B_flops_bsz4_output.txt")

if __name__ == "__main__":
    cli_main()