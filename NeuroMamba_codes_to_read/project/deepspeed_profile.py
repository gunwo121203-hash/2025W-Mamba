import gc
import logging
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torch.nn as nn
import intel_extension_for_pytorch as ipex # Aurora specific - 250306 jubin added
import oneccl_bindings_for_pytorch as torch_ccl
import pytorch_lightning as pl
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.profiling.flops_profiler.profiler import *
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint, get_fp32_state_dict_from_zero_checkpoint

from torch import distributed as dist

from module.utils.data_module import fMRIDataModule
from module.pl_classifier import LitClassifier
from module.aurora_utils import xpu_intel # Aurora specific - 250306 jubin added
from module.aurora_utils.ddp_intel import MPIDDPStrategy, MPIEnvironment # Aurora specific - 250306 jubin added
from module.aurora_utils.deepspeed_intel import XPUDeepSpeedStrategy # Aurora specific - 250319 jubin added

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

LOGLEVEL = os.environ.get('LOGLEVEL', 'WARNING').upper()
logging.basicConfig(level=LOGLEVEL, format="%(asctime)s %(message)s")

def get_model_profile(model,
                      input=None,
                      input_shape=None,
                      args=[],
                      kwargs={},
                      print_profile=True,
                      detailed=True,
                      module_depth=-1,
                      top_modules=1,
                      warm_up=1,
                      as_string=True,
                      output_file=None,
                      ignore_modules=None,
                      mode='forward'):
    """Returns the total floating-point operations, MACs, and parameters of a model.

    Example:

    .. code-block:: python

        model = torchvision.models.alexnet()
        batch_size = 256
        flops, macs, params = get_model_profile(model=model, input_shape=(batch_size, 3, 224, 224)))

    Args:
        model ([torch.nn.Module]): the PyTorch model to be profiled.
        input_shape (tuple): input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
        args (list): list of positional arguments to the model.
        kwargs (dict): dictionary of keyword arguments to the model.
        print_profile (bool, optional): whether to print the model profile. Defaults to True.
        detailed (bool, optional): whether to print the detailed model profile. Defaults to True.
        module_depth (int, optional): the depth into the nested modules. Defaults to -1 (the inner most modules).
        top_modules (int, optional): the number of top modules to print in the aggregated profile. Defaults to 3.
        warm_up (int, optional): the number of warm-up steps before measuring the latency of each module. Defaults to 1.
        as_string (bool, optional): whether to print the output as string. Defaults to True.
        output_file (str, optional): path to the output file. If None, the profiler prints to stdout.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.

    Returns:
        The number of floating-point operations, multiply-accumulate operations (MACs), and parameters in the model.
    """
    assert isinstance(model, nn.Module), "model must be a PyTorch module"
    prof = FlopsProfiler(model)
    model.eval()

    if input is not None:
        args = [input]
    elif input is None and input_shape is not None:
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"
        try:
            input = torch.ones(()).new_empty(
                (*input_shape, ),
                dtype=next(model.parameters()).dtype,
                device=next(model.parameters()).device,
            )
        except StopIteration:
            input = torch.ones(()).new_empty((*input_shape, ))

        args = [input]
    assert (len(args) > 0) or (len(kwargs) > 0), "args and/or kwargs must be specified if input_shape is None"

    logger.info("Flops profiler warming-up...")
    for _ in range(warm_up):
        if kwargs:
            if mode == 'forward':
                _ = model(*args, **kwargs)
            if mode == 'generate':
                _ = model.generate(*args, **kwargs)
        else:
            if mode == 'forward':
                _ = model(*args)
            if mode == 'generate':
                _ = model.generate(*args)
    prof.start_profile(ignore_list=ignore_modules)

    if kwargs:
        if mode == 'forward':
            _ = model(*args, **kwargs)
        if mode == 'generate':
            _ = model.generate(*args, **kwargs)
    else:
        if mode == 'forward':
            _ = model(*args)
        if mode == 'generate':
            _ = model.generate(*args)

    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        prof.print_model_profile(profile_step=warm_up,
                                 module_depth=module_depth,
                                 top_modules=top_modules,
                                 detailed=detailed,
                                 output_file=output_file)

    prof.end_profile()
    if as_string:
        return number_to_string(flops), macs_to_string(macs), params_to_string(params)

    return flops, macs, params



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
    parser.add_argument("--use_early_stopping", action='store_true', help="option for save all weights of every epochs")
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

    if "16" in str(args.precision):
        torch.set_float32_matmul_precision("medium")

    #override parameters
    max_epochs = args.max_epochs
    num_nodes = args.num_nodes
    devices = args.devices
    project_name = args.project_name
    image_path = args.image_path

    if temp_args.resume_ckpt_path:
        if 'deepspeed' in args.strategy:
            exp_id = args.resume_ckpt_path.split("/")[-2]
        else:
            # resume previous experiment
            from module.utils.neptune_utils import get_prev_args
            args = get_prev_args(args.resume_ckpt_path, args)
            exp_id = args.id
            # override max_epochs if you hope to prolong the training
            args.project_name = project_name
            args.max_epochs = max_epochs
            args.num_nodes = num_nodes
            args.devices = devices
            args.image_path = image_path       
    else:
        exp_id = None
    
    setattr(args, "default_root_dir", f"output/{args.project_name}")

    # ----- Setting for Aurora ------
    # Aurora specific - 250319 jubin added
    args.device='xpu'
    args.dtype = torch.float32 if str(args.precision) == '32' else torch.bfloat16

    # ------------ data -------------
    args.batch_size=1 # 250410 added
    data_module = Dataset(**vars(args))
    data_module.setup()
    data_module.prepare_data()

    # ------------ model -------------
    model = Classifier(data_module = data_module, **vars(args)) 

    deepspeed.init_distributed(dist_init_required=True)
    local_rank = int(os.environ.get("LOCAL_RANK"))
    get_accelerator().set_device(local_rank)

    device = f'xpu:{local_rank}'
    output_file=f'./{args.model}_embed{args.embed_dim}_brainOnly{args.calc_loss_without_background}_profile.txt'

    train_iter_per_epoch = len(data_module.train_dataset)
    print(f"batch_size:{args.batch_size}, train_iter_per_epoch:{train_iter_per_epoch}, patch_size:{args.patch_size}")

    model = model.to(device, dtype=args.dtype)
    batch_size = args.batch_size
    seq_len = args.sequence_length
    get_num_tokens = lambda batch_size, args, train_iter_per_epoch: (
        batch_size * (96/args.patch_size[0]) * (96/args.patch_size[1]) * (96/args.patch_size[2]) 
        * (args.sequence_length/args.patch_size[3]) * train_iter_per_epoch
        )
    
    if 'simmim' in args.model:
        # define mask for the model
        mask = [] 
        mask.append(model.mask_generator().unsqueeze(0))
        mask = torch.vstack(mask).to(device, dtype=args.dtype)    # B D//p H//p W//p T//p 
        print(mask.shape)

        class ModelWrapper(nn.Module):
            def __init__(self, model, mask):
                super(ModelWrapper, self).__init__()
                self.model = model
                self.mask = mask

            def forward(self, x):
                return self.model(x, mask=self.mask)
        wrapped_model = ModelWrapper(model.model, mask)
    else: 
        wrapped_model = model.model

    DS_CONFIG_PATH = os.environ.get("DS_CONFIG_PATH", './ds_config.json')
    print('DS_CONFIG_PATH:', DS_CONFIG_PATH)
    # model_engine, _, _, _ = deepspeed.initialize(
    #     model = model,
    #     optimizer = None,
    #     lr_scheduler = None,
    #     config_params = DS_CONFIG_PATH
    # )

    if 'fmamba' in args.model:
        batch = next(iter(data_module.val_loader))
        fmri, subj, target_value, tr, sex = batch.values()
        input = fmri.to(dtype=args.dtype, device=device)
    else:
        input = None

    with torch.autocast(device_type='xpu', dtype=args.dtype, enabled=args.dtype == torch.bfloat16):
        flops_per_batch, _, params = get_model_profile(
            model=wrapped_model,
            input=input,
            input_shape=(batch_size,1,96,96,96,seq_len),
            print_profile=True,
            detailed=False,
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
        with open(output_file, 'w') as f:
            f.write(f"FLOPS_per_batch: {flops_per_batch:.2e}\n")
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

if __name__ == "__main__":
    cli_main()