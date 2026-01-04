import os
import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict

import torch
import pytorch_lightning as pl
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint, get_fp32_state_dict_from_zero_checkpoint

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from module.utils.data_module import fMRIDataModule
from module.pl_classifier import LitClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

LOGLEVEL = os.environ.get('LOGLEVEL', 'WARNING').upper()
logging.basicConfig(level=LOGLEVEL, format="%(asctime)s %(message)s")

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

    tags = args.neptune_tags.copy()
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

    # ------------ logger -------------
    if args.loggername == "tensorboard":
        # logger = True  # tensor board is a default logger of Trainer class
        dirpath = args.default_root_dir
        logger = TensorBoardLogger(dirpath)
    elif args.loggername == "neptune":
        API_KEY = os.environ.get("NEPTUNE_API_TOKEN")
        # project_name should be "WORKSPACE_NAME/PROJECT_NAME"

        custom_run_id = os.environ.get("NEPTUNE_CUSTOM_RUN_ID",None) # 250106 neptune custom run id
        neptune_id_pair = ('with_id', exp_id) if custom_run_id == None else ('custom_run_id', custom_run_id) # 250106 neptune custom run id
        neptune_kwargs = {
                'api_token':API_KEY,
                'project':args.project_name,
                'tags':tags,
                'log_model_checkpoints':False,
                 neptune_id_pair[0]: neptune_id_pair[1], # 250106 neptune custom run id
                'capture_stdout': False,
                'capture_stderr': False,
                'capture_hardware_metrics': False,
                'capture_traceback': True,
                } # in order to run only 1 neptune logger - jubin
        logger = NeptuneLogger(**neptune_kwargs)

        setattr(args, "id", logger.run['sys/id'].fetch())        
        dirpath = os.path.join(args.default_root_dir, logger.version) if logger.version is not None else args.default_root_dir
        if custom_run_id != None and os.environ.get("RANK", None) == '0': # 250106 neptune custom run id
            with open(f'./neptune_id/{neptune_id_pair[1]}_id.txt', 'w') as f:
                print(f"ID for {neptune_id_pair[1]}:",args.id, 'saved in', f'./neptune_id/{neptune_id_pair[1]}_id.txt')
                f.write(args.id)

        # log NeptuneUnsupportedType hyperparameter error manually
        if os.environ.get("RANK", None) == '0':
            from collections.abc import Iterable
            #hyper needs the four following aliases to be done manually.
            for key, value in vars(args).items():
                if isinstance(value, Iterable):
                    logger.run[f'training/hyperparams/{key}_str'] = str(value)
                if isinstance(value, type(None)):
                    logger.run[f'training/hyperparams/{key}_str'] = "None"
            orig_hparams = logger.run['training/hyperparams'].fetch()
            for key in orig_hparams:
                del logger.run[f'training/hyperparams/{key}']
    else:
        raise Exception("Wrong logger name.")

    # ------------ data -------------
    data_module = Dataset(**vars(args))

    # ------------ callbacks -------------
    # callback for pretraining task
    save_top_k = 1 if not args.save_every_checkpoints else -1
    if args.pretraining:
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_loss",
            filename="checkpt-{epoch:02d}-{valid_loss:.4f}",
            save_last=True,
            mode="min",
            save_top_k=save_top_k,
        )
    # callback for classification task
    elif args.downstream_task == "sex" or args.downstream_task == "Dummy" or args.downstream_task_type == "classification":
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_acc",
            filename="checkpt-{epoch:02d}-{valid_acc:.4f}",
            save_last=True,
            mode="max",
            save_top_k=save_top_k,
        )
    # callback for regression task
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_mse",
            filename="checkpt-{epoch:02d}-{valid_mse:.4f}",
            save_last=True,
            mode="min",
            save_top_k=save_top_k,
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    if args.use_early_stopping:
        if args.downstream_task_type == "classification":
            monitor="valid_acc"
            mode='max'
        else:
            monitor="valid_mse"
            mode='min'
        early_stopping = EarlyStopping(monitor, patience=10, mode=mode, log_rank_zero_only=True)
        callbacks.append(early_stopping)

    if args.every_n_train_steps is not None:
        checkpoint_step_callback = ModelCheckpoint(
            dirpath=dirpath,
            filename="checkpt-{epoch:02d}-{step:03d}",
            save_last=True,
            save_top_k=-1,
        )
        callbacks.append(checkpoint_step_callback)
        
    minimum_dataset_len = min((data_module.train_loader.dataset.total_len,
                               data_module.val_loader.dataset.total_len,
                               data_module.test_loader.dataset.total_len))
    log_every_n_steps = int(minimum_dataset_len / (args.batch_size * int(num_nodes) * int(devices))) - 1
    log_every_n_steps = 1 if log_every_n_steps <= 0 else log_every_n_steps
    log_every_n_steps = 50 if log_every_n_steps > 50 else log_every_n_steps
    print("log_every_n_steps:",log_every_n_steps)
    # ------------ trainer -------------
    limit_kwargs={
        'limit_train_batches':4,
        'limit_val_batches':4,
        'limit_test_batches':4
        } if args.limit_batches else {}
    if args.grad_clip:
        print('using gradient clipping')
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm="norm",
            track_grad_norm=-1,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            log_every_n_steps=log_every_n_steps,
            **limit_kwargs
        )
    else:
        print('not using gradient clipping')
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            check_val_every_n_epoch=1,
            callbacks=callbacks,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            log_every_n_steps=log_every_n_steps,
            **limit_kwargs
        )

    # ------------ model -------------
    model = Classifier(data_module = data_module, **vars(args)) 

    if args.load_model_path:
        if 'deepspeed' in args.strategy and not args.load_ds_ckpt_manually:
            # see https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html for loading deepspeed checkpoint
            print(f"loading model from {args.load_model_path}")
            assert args.load_model_path.endswith(".ckpt")
            tag = 'checkpoint'
            model = load_state_dict_from_zero_checkpoint(model, args.load_model_path, tag=tag)
        else:
            print(f'loading model from {args.load_model_path}')
            path = args.load_model_path
            if path.endswith('pt') or 'deepspeed' in args.strategy:
                # see https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html for loading deepspeed checkpoint
                state = torch.load(path) if 'deepspeed' not in args.strategy else get_fp32_state_dict_from_zero_checkpoint(path)
                new_state = OrderedDict()
                for k, v in state.items():
                    new_state[k.removeprefix("_forward_module.")] = v
                ckpt = {'state_dict' : new_state}
            else:
                ckpt = torch.load(path)
            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                if 'model.decoder' in k or 'model.mask_token' in k or 'output_head' in k or 'output_layer' in k:
                    continue
                else:
                    new_state_dict[k.removeprefix("model.")] = v
            model.model.load_state_dict(new_state_dict, strict=False)

    if args.freeze_feature_extractor:
        # layers are frozen by using eval()
        model.model.eval()
        # freeze params
        for name, param in model.model.named_parameters():
            if 'output_head' not in name: # unfreeze only output head
                param.requires_grad = False
                print(f'freezing layer {name}')
    elif args.finetune_last_block:
        print("Fine-tuning last block: Freezing earlier layers/modules AND setting them to eval mode.")
        modules_to_freeze = [model.model.patch_embed]
        # Add pos_embeds and layers for stages 0 to num_layers-2
        num_layers_to_freeze = model.model.num_layers - 1 # SwinTransformer4D.num_layers == len(depths) == 4
        for i in range(num_layers_to_freeze):
            if i < len(model.model.pos_embeds): # Check if pos_embeds exist for this layer
                modules_to_freeze.append(model.model.pos_embeds[i])
            if i < len(model.model.layers): # Check if layer exists
                modules_to_freeze.append(model.model.layers[i])

        for module in modules_to_freeze:
            module.eval() # Set module to evaluation mode
            for param in module.parameters():
                param.requires_grad = False # Freeze parameters within the module
        print(f"Froze and set to eval: patch_embed, pos_embeds[0..{num_layers_to_freeze-1}], layers[0..{num_layers_to_freeze-1}]")

    # ------------ run -------------
    if args.test_only:
        if args.test_ckpt_path and not args.load_ds_ckpt_manually:
            trainer.test(model, datamodule=data_module, ckpt_path=args.test_ckpt_path)
        else:
            if args.load_ds_ckpt_manually:
                # manually load ckpt path to apply removeprefix and so on
                # using load_state_dict_from_zero_checkpoint ignores unmatched state_dict key/values
                print(f'loading model from {args.test_ckpt_path}')
                state = get_fp32_state_dict_from_zero_checkpoint(args.test_ckpt_path)
                new_state = OrderedDict()
                for k, v in state.items():
                    new_state[k.removeprefix("_forward_module.")] = v
                ckpt = {'state_dict' : new_state}
                new_state_dict = OrderedDict()
                for k, v in ckpt['state_dict'].items():
                    if 'model.decoder' in k or 'model.mask_token' in k or 'output_head' in k or 'output_layer' in k:
                        continue
                    else:
                        new_state_dict[k.removeprefix("model.")] = v
                model.model.load_state_dict(new_state_dict, strict=False)
            # else: # to check randomly initialized model
            trainer.test(model, datamodule=data_module)
    else:
        if not args.resume_ckpt_path:
            # New run
            trainer.fit(model, datamodule=data_module)
        else:
            # Resume existing run
            if os.environ.get("NEPTUNE_CUSTOM_RUN_ID", None) and args.loggername == 'neptune': # 250106 neptune custom run id
                with open(f'./neptune_id/{neptune_id_pair[1]}_id.txt', 'r') as f:
                    exp_id = f.read()
                resume_ckpt_path = args.resume_ckpt_path.split('/')
                resume_ckpt_path[-2] = exp_id
                args.resume_ckpt_path = '/'.join(resume_ckpt_path)
                print(f"Load {args.resume_ckpt_path} from NEPTUNE_CUSTOM_RUN_ID \"{neptune_id_pair[1]}\" whose ID was saved in ./neptune_id/{neptune_id_pair[1]}_id.txt")
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

        trainer.test(model, dataloaders=data_module, ckpt_path='best') 

if __name__ == "__main__":
    cli_main()