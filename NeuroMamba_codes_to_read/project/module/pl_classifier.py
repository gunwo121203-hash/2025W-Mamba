import os
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import monai.transforms as monai_t
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import balanced_accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchmetrics.classification import BinaryAccuracy, AUROC
from torchmetrics.regression import R2Score
from torchmetrics import PearsonCorrCoef  # Accuracy,

from .models.load_model import load_model
from .utils.masking_generator import RandomMaskingGenerator, simmim_MaskGenerator
from .utils.metrics import Metrics
from .utils.losses import NTXentLoss
from .utils.lr_scheduler import (
    CosineAnnealingWarmUpRestarts,
    CosineAnnealingWarmUpRestartsMup,
)
from .utils.parser import str2bool

DEVICE='cuda'

class LitClassifier(pl.LightningModule):
    def __init__(self,data_module, **kwargs):
        super().__init__()
        self.save_hyperparameters(kwargs) # save hyperparameters except data_module (data_module cannot be pickled as a checkpoint)
        self.valid_only = kwargs.get("valid_only", False)
        init_model = lambda model_name, hparams: load_model(model_name, hparams) if not self.hparams.use_MuTransfer else self._set_mup_model(model_name, hparams)

        # Swin4DTransformer 
        self.model = init_model(self.hparams.model, self.hparams)            

        # Heads
        self.output_head = None
        if not self.hparams.pretraining: ## Finetuning model for downstream tasks
            print(f"Downstream task:{self.hparams.downstream_task} {self.hparams.downstream_task_type}")
            if 'fmamba' in self.hparams.model:
                if self.hparams.clf_head_version == 'v1':
                    mlp_name = 'fmamba_mlp'
                elif self.hparams.clf_head_version == 'conv':
                    mlp_name = 'fmamba_conv_mlp'
                elif self.hparams.clf_head_version == 'mamba':
                    mlp_name = 'fmamba_mamba_head'
                elif self.hparams.clf_head_version == 'mamba_fn':
                    mlp_name = 'fmamba_mamba_head_fn'
            elif self.hparams.use_layer_embedding_mlp:
                print("Using Layer-wise MLP heads.")
                mlp_name = 'layer_embedding_mlp'
            else:
                print("Using single output head.")
                if self.hparams.downstream_task == 'sex' or self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
                    mlp_name = "clf_mlp" 
                elif self.hparams.downstream_task == 'age' or self.hparams.downstream_task == 'int_total' or self.hparams.downstream_task == 'int_fluid' or self.hparams.downstream_task_type == 'regression':
                    mlp_name = "reg_mlp"
                else:
                    raise ValueError(f"Unsupported downstream task/type: {self.hparams.downstream_task} / {self.hparams.downstream_task_type}")
            self.output_head = init_model(mlp_name, self.hparams)

            if self.hparams.downstream_task == 'age' or self.hparams.downstream_task == 'int_total' or self.hparams.downstream_task == 'int_fluid' or self.hparams.downstream_task_type == 'regression':
                # you should define target_values at the Dataset classes
                target_values = data_module.train_dataset.target_values
                if self.hparams.label_scaling_method == 'standardization':
                    scaler = StandardScaler()
                    normalized_target_values = scaler.fit_transform(target_values)
                    print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')
                elif self.hparams.label_scaling_method == 'minmax': 
                    scaler = MinMaxScaler()
                    normalized_target_values = scaler.fit_transform(target_values)
                    print(f'target_max:{scaler.data_max_[0]},target_min:{scaler.data_min_[0]}')
                self.scaler = scaler

        ## Pretraining a model
        elif self.hparams.use_contrastive:
            self.output_head = init_model("emb_mlp", self.hparams)
        elif self.hparams.use_mim:
            img_size = self.hparams.img_size 
            patch_size = self.hparams.patch_size 
            mask_patch_size = self.hparams.mask_patch_size 
            self.mask_generator = simmim_MaskGenerator(input_size=img_size,
                                                       mask_patch_size=mask_patch_size,
                                                       model_patch_size=patch_size,
                                                       mask_ratio=self.hparams.mask_ratio,
                                                       masking_type=self.hparams.masking_type)
        elif self.hparams.use_autoregressive:
            pass
        else:
            raise NotImplementedError("output head should be defined")

        self.metric = Metrics()
        if self.hparams.adjust_thresh:
            self.threshold = 0

    def _set_mup_model(self, model_name, hparams): # Helper function for setting up MuTransfer model
        print(f"Using MuTransfer during training for {model_name}")
        from mup import set_base_shapes, make_base_shapes
        orig_embed_dim = hparams.embed_dim
        mup_filename = f'output/base_shapes_{orig_embed_dim}_{str(hparams.num_heads)}_{self.hparams.model}.yaml'
        if hparams.depths != [2, 2, 18, 2]:
            orig_depth = hparams.depths
            str_depth = '_'.join([str(d) for d in orig_depth])
            mup_filename = mup_filename.replace('.yaml', f'_depth{str_depth}.yaml')
        if not self.hparams.pretraining:
            mup_filename = mup_filename.replace('.yaml', '_downstream.yaml')
        if 'mlp' in model_name:
            mup_filename = mup_filename.replace('.yaml', f'_{model_name}.yaml')
        if not os.path.exists(mup_filename):
            # for base model 
            if hparams.depths != [2, 2, 18, 2]:
                setattr(hparams, "depth", [2, 2, 6, 2])
            else:
                setattr(hparams, "embed_dim", 36)
            base_model = load_model(model_name, hparams)
            base_model.set_MuReadout_layer()
            # for delta model
            if hparams.depths != [2, 2, 18, 2]:
                setattr(hparams, "depth", [2, 2, 18, 2])
            else:
                setattr(hparams, "embed_dim", 72)
            delta_model = load_model(model_name, hparams)
            delta_model.set_MuReadout_layer()
            # for target model
            if hparams.depths != [2, 2, 18, 2]:
                setattr(self.hparams, "depth", orig_depth)
            else:
                setattr(self.hparams, "embed_dim", orig_embed_dim)
            model = load_model(model_name, self.hparams)
            model.set_MuReadout_layer()
            set_base_shapes(model, base_model, delta=delta_model)
            make_base_shapes(base_model, delta_model, mup_filename)
            print("Save mup base shape at", mup_filename)
            model.mup_init_weights()

            del base_model, delta_model
            import gc 
            try:
                1/0
            except:
                gc.collect()
        else:
            model = load_model(model_name, hparams)
            model.set_MuReadout_layer()
            set_base_shapes(model, mup_filename)
            print("Load mup base shape at", mup_filename)
            model.mup_init_weights()        
        return model

    def forward(self, x):
        return self.output_head(self.model(x))
        
    def augment(self, img):
        B, C, H, W, D, T = img.shape

        device = img.device
        img = rearrange(img, 'b c h w d t -> b t c h w d')

        rand_affine = monai_t.RandAffine(
            prob=0.5 if self.hparams.augment_strength <= 1 else 0.8,
            # 0.175 rad = 10 degrees
            rotate_range=(0.175, 0.175, 0.175) if self.hparams.augment_strength <= 1 else (0.175*self.hparams.augment_strength,)*3,
            scale_range = (0.1, 0.1, 0.1),
            mode = "bilinear",
            padding_mode = "border",
            device = device
        )
        rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1) if self.hparams.augment_strength <= 1 else monai_t.RandGaussianNoise(prob=0.8, std=0.1*self.hparams.augment_strength)
        rand_smooth = monai_t.RandGaussianSmooth(sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5), sigma_z=(0.0, 0.5), prob=0.1)
        if self.hparams.augment_only_intensity:
            comp = monai_t.Compose([rand_noise, rand_smooth])
        else:
            comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth]) 

        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            for b in range(B):
                aug_seed = torch.randint(0, 10000000, (1,)).item()
                # set augmentation seed to be the same for all time steps
                for t in range(T):
                    if self.hparams.augment_only_affine:
                        rand_affine.set_random_state(seed=aug_seed)
                        img[b, t, :, :, :, :] = rand_affine(img[b, t, :, :, :, :])
                    else:
                        comp.set_random_state(seed=aug_seed)
                        img[b, t, :, :, :, :] = comp(img[b, t, :, :, :, :])

        img = rearrange(img, 'b t c h w d -> b c h w d t')
            
        return img
    
    def _save_embeddings(self, feature, subj, tr, brain_mask = None):
        if max(tr) > 200:
            return
        
        if not hasattr(self, 'embedding_save_dir'):
            if self.hparams.embedding_save_dir is None:
                from datetime import datetime
                self.embedding_save_dir = datetime.now().strftime("%y%m%d_%H%M%S")
                if not self.hparams.test_ckpt_path:
                    self.embedding_save_dir = self.embedding_save_dir + "_scratch"
                self.embedding_save_dir = os.path.join("output", "embeddings", self.embedding_save_dir)
            else:
                self.embedding_save_dir = self.hparams.embedding_save_dir
            os.makedirs(self.embedding_save_dir, exist_ok=True)
            print(f"Saving embeddings to: {self.embedding_save_dir}")

        if not self.hparams.use_layer_embedding_mlp:
            layer_embeddings = {'layer_3': feature}
        else:
            layer_embeddings = feature
            
        for layer_name, layer_embedding_dict in layer_embeddings.items():
            for i, feature_i in enumerate(layer_embedding_dict):
                save_path = os.path.join(self.embedding_save_dir, f"{layer_name}_{subj[i]}_{tr[i]}.pt")
                try:
                    torch.save(feature_i.detach().cpu(), save_path)
                except Exception as e:
                    print(f"Error saving tensor for subj {subj[i]}, tr {tr[i]}: {e}")

        if brain_mask != None and (tr == 0).sum():
            # find the tr == 0
            tr_0_indices = (tr == 0).nonzero(as_tuple=True)[0]
            for idx in tr_0_indices:
                save_path = os.path.join(self.embedding_save_dir, f"brain_mask_{subj[idx]}_{tr[idx]}.pt")
                try:
                    torch.save(brain_mask[idx].detach().cpu(), save_path)
                except Exception as e:
                    print(f"Error saving brain mask for subj {subj[idx]}, tr {tr[idx]}: {e}")

    def _compute_logits(self, batch, augment_during_training=None, mode=None):
        fmri, subj, target_value, tr, sex = batch.values()
       
        if augment_during_training and mode == 'train':
            fmri = self.augment(fmri)

        feature = self.model(fmri)
        # Save fMRI embedding to analyze model
        if self.hparams.save_embedding and mode == 'test':
            if 'fmamba' in self.hparams.model:
                orig_feature, feature = self.model(fmri, return_z=True)
            else:
                orig_feature = feature
            self._save_embeddings(feature, subj, tr)  # Call the new function
            feature = orig_feature
                            
        outputs = self.output_head(feature) # outputs is a dict if use_layer_embedding_mlp is True, else a single tensor

        # Prepare target (handle classification vs regression)
        if self.hparams.downstream_task_type == 'classification':
            target = target_value.long() # CrossEntropyLoss & AUROC expects long targets
        elif self.hparams.downstream_task_type == 'regression':
            unnormalized_target = target_value.float()# .unsqueeze(-1) # Ensure shape (B, 1) # TODO: CHECK if needed
            # Normalize target
            if self.hparams.label_scaling_method == 'standardization':
                target = (unnormalized_target - self.scaler.mean_[0]) / (self.scaler.scale_[0])
            elif self.hparams.label_scaling_method == 'minmax':
                target = (unnormalized_target - self.scaler.data_min_[0]) / (self.scaler.data_max_[0] - self.scaler.data_min_[0])
            # target = target.squeeze(-1) # Back to shape (B) for loss calculation

        else: # Pretraining or other tasks
             target = target_value # Pass target as is

        return subj, outputs, target
    
    def _calculate_loss(self, batch, batch_idx, mode):
        if self.hparams.pretraining:
            fmri, subj, target_value, tr, sex = batch.values()
            
            cond1 = (self.hparams.in_chans == 1 and not self.hparams.with_voxel_norm)
            assert cond1, "Wrong combination of options"
            loss = 0

            if self.hparams.use_contrastive:
                assert self.hparams.contrastive_type != "none", "Contrastive type not specified"

                # B, C, H, W, D, T = image shape
                y, diff_y = fmri

                batch_size = y.shape[0]
                if (len(subj) != len(tuple(subj))) and mode == 'train':
                    print('Some sub-sequences in a batch came from the same subject!')
                criterion = NTXentLoss(device=DEVICE, batch_size=batch_size,
                                        temperature=self.hparams.temperature,
                                        use_cosine_similarity=True).cuda()
                criterion_ll = NTXentLoss(device=DEVICE, batch_size=2,
                                            temperature=self.hparams.temperature,
                                            use_cosine_similarity=True).cuda()
                
                # type 1: IC
                # type 2: LL
                # type 3: IC + LL
                if self.hparams.contrastive_type in [1, 3]:
                    out_global_1 = self.output_head(self.model(self.augment(y)),"g")
                    out_global_2 = self.output_head(self.model(self.augment(diff_y)),"g")
                    ic_loss = criterion(out_global_1, out_global_2)
                    loss += ic_loss

                if self.hparams.contrastive_type in [2, 3]:
                    out_local_1 = []
                    out_local_2 = []
                    out_local_swin1 = self.model(self.augment(y))
                    out_local_swin2 = self.model(self.augment(y))
                    out_local_1.append(self.output_head(out_local_swin1, "l"))
                    out_local_2.append(self.output_head(out_local_swin2, "l"))

                    out_local_swin1 = self.model(self.augment(diff_y))
                    out_local_swin2 = self.model(self.augment(diff_y))
                    out_local_1.append(self.output_head(out_local_swin1, "l"))
                    out_local_2.append(self.output_head(out_local_swin2, "l"))

                    ll_loss = 0
                    # loop over batch size
                    for i in range(out_local_1[0].shape[0]):
                        # out_local shape should be: BS, n_local_clips, D
                        ll_loss += criterion_ll(torch.stack(out_local_1, dim=1)[i],
                                                torch.stack(out_local_2, dim=1)[i])
                    loss += ll_loss

                result_dict = {
                    f"{mode}_loss": loss,
                }

            elif self.hparams.use_mim: 
                assert self.hparams.contrastive_type != None, "masking type not specified"
                if 'simmim' in self.hparams.model: # assume we only use simmim
                    Bsz, C, D, H, W, T = fmri.shape
                    background_value = fmri.amin((1,2,3,4,5)) if self.trainer.datamodule.hparams.input_scaling_method == 'znorm_minback' else torch.zeros(Bsz) # consider znorm_zeroback or minmax scaling
                    background_value = background_value.to(fmri.device)
                    if self.hparams.calc_loss_without_background: # filter tokens that include at least 1 brain voxel (remove background tokens from reconstruction loss calculation)
                        pD, pH, pW, pT = self.model.patch_embed.grid_size 
                        sD, sH, sW, sT = self.hparams.patch_size
                        fmri_patches = fmri.view(Bsz, C, pD, sD, pH, sH, pW, sW, pT, sT)
                        fmri_patches = fmri_patches.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(Bsz,-1, sD * sH * sW * sT * C)
                        valid_token_mask = (fmri_patches!=background_value.view(Bsz,1,1)).any((2)) # found a token that has at least one voxel that has a value larger than background
                        valid_token_mask = valid_token_mask.unsqueeze(-1).repeat_interleave(sD * sH * sW * sT, -1)
                        valid_token_mask = valid_token_mask.view(Bsz, pD, pH, pW, pT, sD, sH, sW, sT, C).permute(0, 9, 1, 5, 2, 6, 3, 7, 4, 8).contiguous()
                        valid_token_mask = valid_token_mask.view(Bsz, C, D, H, W, T)
                    
                    if self.hparams.augment_during_training and mode == 'train':
                        fmri = self.augment(fmri)
                    
                    # generate mask per batch
                    mask = [] 
                    for B in range(fmri.shape[0]):
                        mask.append(self.mask_generator().unsqueeze(0))
                    mask = torch.vstack(mask)    # B D//p H//p W//p T//p 
                    mask = mask.to(fmri.device)
                    
                    # forward 
                    fmri_rec, mask = self.model(fmri, mask)
                    if self.hparams.loss_type.upper() == "L1":
                        loss_recon = F.l1_loss(fmri, fmri_rec, reduction='none')
                    elif self.hparams.loss_type == "smoothL1":
                        loss_recon = F.smooth_l1_loss(fmri, fmri_rec, reduction='none')
                    elif self.hparams.loss_type == "huber":
                        loss_recon = F.huber_loss(fmri, fmri_rec, delta=0.1,reduction='none')
                    elif self.hparams.loss_type == "L2":
                        loss_recon = F.mse_loss(fmri, fmri_rec, reduction='none')
                        
                    if self.hparams.calc_loss_without_background:
                        loss = (loss_recon * mask * valid_token_mask).sum() / ((mask & valid_token_mask).sum() + 1e-5)
                    elif self.hparams.calc_loss_brain_mask:
                        fmri_brain_mask = fmri != background_value.view(Bsz,1,1,1,1,1)
                        loss = (loss_recon * mask * fmri_brain_mask).sum() / ((mask & fmri_brain_mask).sum() + 1e-5)
                    else:
                        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5)
                        
                    if (batch_idx == 0 
                        and self.trainer.is_global_zero 
                        and str(self.trainer.state.stage) != "RunningStage.SANITY_CHECKING" 
                        and self.hparams.save_mim_sample): 
                        # save original image 
                        if not hasattr(self, 'save_id'):
                            try:
                                self.save_id = self.trainer.logger.version
                            except:
                                from datetime import datetime 
                                self.save_id = datetime.now().strftime("%y%m%d_%H%M%S")
                        os.makedirs(f'mim_pred_sample/{self.save_id}',exist_ok=True)
                        np.save(f'mim_pred_sample/{self.save_id}/simmim_img_{self.current_epoch}_{self.hparams.dataset_name[0]}_seq{self.hparams.sequence_length}_timepatch{self.hparams.patch_size[-1]}_masking{self.hparams.mask_ratio}_ape_{mode}.npy', fmri.detach().cpu().float().numpy())
                        # save mask 
                        np.save(f'mim_pred_sample/{self.save_id}/simmim_mask_{self.current_epoch}_{self.hparams.dataset_name[0]}_seq{self.hparams.sequence_length}_timepatch{self.hparams.patch_size[-1]}_masking{self.hparams.mask_ratio}_ape_{mode}.npy', mask.detach().cpu().float().numpy())
                        # save pred image 
                        np.save(f'mim_pred_sample/{self.save_id}/simmim_pred_{self.current_epoch}_{self.hparams.dataset_name[0]}_seq{self.hparams.sequence_length}_timepatch{self.hparams.patch_size[-1]}_masking{self.hparams.mask_ratio}_ape_{mode}.npy', fmri_rec.detach().cpu().float().numpy())
                        print("SAVE SAMPLE DONE")
                logging_str = mode if str(self.trainer.state.fn) != "TrainerFn.TESTING" else "eval_"+mode

                result_dict = {f"{logging_str}_loss": loss.item()}

            elif self.hparams.use_autoregressive: 
                if self.hparams.augment_during_training and mode == 'train':
                    fmri = self.augment(fmri)
                
                if not (self.hparams.save_embedding and mode == 'test'):
                    fmri_pred, fmri_orig = self.model(fmri) 
                else:
                    fmri_pred, fmri_orig, feature = self.model(fmri, return_z=True) 
                
                loss = F.mse_loss(fmri_pred[:,:-1,:], fmri_orig[:,1:,:]) # predict next frame
                # check nan
                if torch.isnan(loss).any():
                    raise ValueError("Loss is NaN. Check your data or model configuration.")
                logging_str = mode if str(self.trainer.state.fn) != "TrainerFn.TESTING" else "eval_"+mode
                result_dict = {f"{logging_str}_loss": loss.item()}

                # Save fMRI embedding to analyze model
                if self.hparams.save_embedding and mode == 'test':
                    Bsz, C, D, H, W, T = fmri.shape
                    background_value = fmri.amin((1,2,3,4,5)) if self.trainer.datamodule.hparams.input_scaling_method == 'znorm_minback' else torch.zeros(Bsz) # consider znorm_zeroback or minmax scaling
                    background_value = background_value.to(fmri.device)

                    pD, pH, pW, pT = self.model.embedder.grid_size 
                    sD, sH, sW, sT = self.hparams.patch_size
                    fmri_patches = fmri.view(Bsz, C, pD, sD, pH, sH, pW, sW, pT, sT)
                    fmri_patches = fmri_patches.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(Bsz,-1, sD * sH * sW * sT * C)
                    valid_token_mask = (fmri_patches!=background_value.view(Bsz,1,1)).any((2)) # found a token that has at least one voxel that has a value larger than background
                    # also save the 
                    self._save_embeddings(feature, subj, tr, brain_mask=valid_token_mask)  # Call the new function

        else:
            subj, outputs, target = self._compute_logits(batch, augment_during_training=self.hparams.augment_during_training, mode=mode)
            # Ensure target dtype matches output dtype for regression loss
            if isinstance(outputs, torch.Tensor): # Single output case
                 target = target.to(outputs.dtype)
            elif isinstance(outputs, dict): # Multi-output case, check first output
                 first_key = next(iter(outputs))
                 target = target.to(outputs[first_key].dtype)

            output_dict = {'single_logit': outputs} if not isinstance(outputs, dict) else outputs
            total_loss = 0.0
            result_dict = {}
            for layer_name, logits in output_dict.items():
                logging_str = mode if 'layer_' not in layer_name else mode + '_' + layer_name # ex) val / val_layer_0
                if self.hparams.downstream_task_type == 'classification': # logits shape = [batch, num_classes], target shape = [batch]
                    loss = F.cross_entropy(logits, target.long(), label_smoothing=0.1)
                    acc = self.metric.get_accuracy(logits, target.float()) 
                    result_dict[f"{logging_str}_loss"] = loss.item()
                    result_dict[f"{logging_str}_acc"] = acc 
                elif self.hparams.downstream_task_type == 'regression': # logits shape = [batch, 1], target shape = [batch]
                    logits_squeezed = logits.squeeze(-1) if logits.ndim > 1 else logits
                    loss = F.mse_loss(logits_squeezed, target)
                    l1 = F.l1_loss(logits_squeezed, target)
                    result_dict[f"{logging_str}_loss"] = loss.item()
                    result_dict[f"{logging_str}_mse"] = result_dict[f"{logging_str}_loss"]
                    result_dict[f"{logging_str}_l1_loss"] = l1
                    if logits.shape[0] > 1: # R2 requires multiple samples
                        r2_score = R2Score().to(target.device)
                        result_dict[f"{logging_str}_r2_score"] = r2_score(logits_squeezed, target)
                    else: 
                        result_dict[f"{logging_str}_r2_score"] = 0.0
                total_loss += loss

            if self.hparams.use_layer_embedding_mlp:
                result_dict[f"{mode}_loss"] = total_loss.item() / len(outputs.keys())
            loss = total_loss

        self.log_dict(result_dict, prog_bar=True, sync_dist=False, add_dataloader_idx=False, on_step=True, on_epoch=True, batch_size=self.hparams.batch_size) # batch_size = batch_size
        return loss

    def _evaluate_metrics(self, aggregated_outputs, mode, best=False):
        mode_str = mode if best == False else 'best_'+mode

        # aggregated_outputs is expected to be {'subjects': ndarray, 'logits': tensor, 'targets': tensor} or {'subjects': ndarray, 'layer_N_logits': tensor, 'targets': tensor}
        subj_array = aggregated_outputs['subjects']
        subjects = np.unique(subj_array)

        logit_keys =  [ key for key in aggregated_outputs.keys() if 'logits' in key ]
        checkpoint_metrics = []
        for logit_key in logit_keys: # logit_key is 'layer_0_logits' (layer_embedding_mlp) or 'logits' (original)
            logging_str = mode_str if 'layer_' not in logit_key else mode_str + '_' + logit_key.replace("_logits", "")            
            subj_avg_logits = []
            subj_targets = []
            for subj in subjects:
                if self.hparams.downstream_task == 'sex'  or self.hparams.downstream_task == 'depression_current' or self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
                    subj_logits = F.softmax(aggregated_outputs[logit_key][subj_array == subj].float(), -1) 
                    subj_logits = torch.mean(subj_logits, 0)    # get mean logits across fMRI seqeunce segment
                    subj_avg_logits.append(F.softmax(subj_logits, -1)) # applying softmax one more time so that mean logits of each class can be represented as probability for each class (before applying softmax once again, sum of logits for class 0 and logist for class 1 is not 1.0)
                elif self.hparams.downstream_task == 'age' or self.hparams.downstream_task == 'int_total' or self.hparams.downstream_task == 'int_fluid' or self.hparams.downstream_task_type == 'regression': 
                    subj_logits = aggregated_outputs[logit_key][subj_array == subj].float()
                    subj_avg_logits.append(torch.mean(subj_logits, 0))
                subj_targets.append(aggregated_outputs['targets'][subj_array == subj][0])
            subj_avg_logits = torch.stack(subj_avg_logits, 0) 
            subj_targets = torch.stack(subj_targets, 0) 
            
            if self.hparams.downstream_task == 'sex' or self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
                if self.hparams.adjust_thresh:
                    # move threshold to maximize balanced accuracy
                    best_bal_acc = 0
                    best_thresh = 0
                    for thresh in np.arange(-5, 5, 0.01):
                        bal_acc = balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=thresh).int().cpu())
                        if bal_acc > best_bal_acc:
                            best_bal_acc = bal_acc
                            best_thresh = thresh
                    self.log(f"{logging_str}_best_thresh", best_thresh, sync_dist=False)
                    self.log(f"{logging_str}_best_balacc", best_bal_acc, sync_dist=False)
                    fpr, tpr, thresholds = roc_curve(subj_targets.cpu(), subj_avg_logits.cpu())
                    idx = np.argmax(tpr - fpr)
                    youden_thresh = thresholds[idx]
                    acc_func = BinaryAccuracy().to('cpu')
                    self.log(f"{logging_str}_youden_thresh", youden_thresh, sync_dist=False)
                    self.log(f"{logging_str}_youden_balacc", balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=youden_thresh).int().cpu()), sync_dist=False)

                    if mode == 'valid':
                        self.threshold = youden_thresh
                    elif mode == 'test':
                        bal_acc = balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits>=self.threshold).int().cpu())
                        self.log(f"{logging_str}_balacc_from_valid_thresh", bal_acc, sync_dist=False)
                else:
                    acc_func = BinaryAccuracy().to('cpu')
                    
                auroc_func = AUROC(task="multiclass", num_classes=2).to('cpu')
                acc = acc_func(torch.argmax(subj_avg_logits, -1), subj_targets)
                bal_acc_sk = balanced_accuracy_score(subj_targets.cpu(), torch.argmax(subj_avg_logits, -1).cpu())
                auroc = auroc_func(subj_avg_logits, subj_targets.long())

                self.log(f"{logging_str}_acc", acc, sync_dist=True) # TODO should add average val_acc/ average val_mse, not layer_0_acc/mse
                self.log(f"{logging_str}_balacc", bal_acc_sk, sync_dist=True)
                self.log(f"{logging_str}_AUROC", auroc, sync_dist=True)

                checkpoint_metrics.append(acc)

            # regression target is normalized
            elif self.hparams.downstream_task == 'age' or self.hparams.downstream_task == 'int_total' or self.hparams.downstream_task == 'int_fluid' or self.hparams.downstream_task_type == 'regression':          
                subj_avg_logits = subj_avg_logits.squeeze(-1) # 250407 - added .squeeze(-1) to logits
                mse = F.mse_loss(subj_avg_logits, subj_targets) 
                mae = F.l1_loss(subj_avg_logits, subj_targets)
                # reconstruct to original scale
                if self.hparams.label_scaling_method == 'standardization': # default
                    adjusted_mse = F.mse_loss((subj_avg_logits * self.scaler.scale_[0] + self.scaler.mean_[0]), subj_targets * self.scaler.scale_[0] + self.scaler.mean_[0])
                    adjusted_mae = F.l1_loss((subj_avg_logits * self.scaler.scale_[0] + self.scaler.mean_[0]), subj_targets * self.scaler.scale_[0] + self.scaler.mean_[0])
                elif self.hparams.label_scaling_method == 'minmax':
                    adjusted_mse = F.mse_loss((subj_avg_logits * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0]), subj_targets * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0])
                    adjusted_mae = F.l1_loss((subj_avg_logits * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0]), subj_targets * (self.scaler.data_max_[0] - self.scaler.data_min_[0]) + self.scaler.data_min_[0])
                if subj_avg_logits.shape[0]>1: # Needs at least two samples to calculate r2 score - 250407 jubin added
                    pearson = PearsonCorrCoef().to('cpu')
                    pearson_coef = pearson(subj_avg_logits, subj_targets) # why squeeze here?
                    r2_score = R2Score().to(subj_targets.device)
                    r2 = r2_score(subj_avg_logits, subj_targets) 
                else:
                    print("_evaluate_metrics warning: There is only one subject. Skip calculating corrcoef and r2")
                    r2=0
                    pearson_coef=0
                
                self.log(f"{logging_str}_corrcoef", pearson_coef, sync_dist=True)
                self.log(f"{logging_str}_r2_score", r2, sync_dist=True)
                self.log(f"{logging_str}_mse", mse, sync_dist=True)
                self.log(f"{logging_str}_mae", mae, sync_dist=True)
                self.log(f"{logging_str}_adjusted_mse", adjusted_mse, sync_dist=True) 
                self.log(f"{logging_str}_adjusted_mae", adjusted_mae, sync_dist=True)  

                checkpoint_metrics.append(mse)

        if len(checkpoint_metrics) > 1:
            checkpoint_metric = sum(checkpoint_metrics)/len(checkpoint_metrics)
            metric_name = f"{mode}_acc" if self.hparams.downstream_task_type == 'classification' else f"{mode}_mse"
            self.log(metric_name, checkpoint_metric, sync_dist=True)


    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mode = "valid" if dataloader_idx == 0 else "test"
        if self.hparams.pretraining:
            return self._calculate_loss(batch, batch_idx, mode=mode)
        else:
            subj, outputs, target = self._compute_logits(batch, mode=mode)
            step_output = {'subjects': subj, 'targets': target.detach().cpu()}
            if self.hparams.use_layer_embedding_mlp: # outputs is a dict {'layer_0': logits_0, ...}
                step_output['layer_logits'] = {name: logits.detach().cpu() for name, logits in outputs.items()}
            else: 
                step_output['logits'] = outputs.detach().cpu() # Original single-head output structure
            return step_output
        
    def _aggregate_epoch_outputs(self, outputs):
        """Helper function to aggregate outputs from step methods."""
        if not outputs or outputs[0] is None: return None

        aggregated = {}
        # Check format based on first output
        first_output = outputs[0]

        # TODO: Check if works well when batch size is 1
        aggregated['subjects'] = np.concatenate([o['subjects'].cpu() if isinstance(o['subjects'], torch.Tensor) else o['subjects'] for o in outputs])
        targets_list = [o['targets'] for o in outputs]
        if len(targets_list[0].shape) == 0: # Targets might be scalars if batch_size=1
            aggregated['targets'] = torch.Tensor(targets_list) # Convert list of scalars to tensor
        else:
            aggregated['targets'] = torch.cat(targets_list, dim=0)

        if 'layer_logits' in first_output: # NEW: Layer-wise case
            # Aggregate logits per layer
            layer_keys = first_output['layer_logits'].keys()
            for layer_name in layer_keys:
                aggregated[f'{layer_name}_logits'] = torch.cat([o['layer_logits'][layer_name] for o in outputs], dim=0)
        elif 'logits' in first_output: # ORIGINAL: Single-head case
            aggregated['logits'] = torch.cat([o['logits'] for o in outputs], dim=0)
        return aggregated
    
    def validation_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            # outputs can be a list (if one dataloader) or list of lists (if multiple)
            outputs_valid = outputs[0] if isinstance(outputs[0], list) else outputs
            outputs_test = outputs[1] if isinstance(outputs[0], list) and len(outputs) > 1 else []
            
            agg_valid = self._aggregate_epoch_outputs(outputs_valid)
            self._evaluate_metrics(agg_valid, mode="valid")

            if not self.valid_only and outputs_test:
                agg_test = self._aggregate_epoch_outputs(outputs_test)
                self._evaluate_metrics(agg_test, mode="test")

        # stop code after 1 epoch has run to save time when submitting max 2hr jobs - 240926 jub
        if (
            str(self.trainer.state.stage) != "RunningStage.SANITY_CHECKING"
            and hasattr(self.trainer, 'ckpt_path') 
            and self.trainer.ckpt_path is not None 
            and self.current_epoch > 0
            and self.hparams.run_only_1epoch == True
        # and self.hparams.embed_dim != 768 # for setting 6
        ):
            print("stop running manually after one epoch, curr epoch:", self.current_epoch)
            self.trainer.should_stop = True
            
    # If you use loggers other than Neptune you may need to modify this
    def _save_predictions(self,total_subjs,total_out, mode):
        self.subject_accuracy = {}
        for subj, output in zip(total_subjs,total_out):
            if self.hparams.downstream_task == 'sex':
                score = torch.sigmoid(output[0]).item()
            else:
                score = output[0].item()

            if subj not in self.subject_accuracy:
                self.subject_accuracy[subj] = {'score': [score], 'mode':mode, 'truth':output[1], 'count':1}
            else:
                self.subject_accuracy[subj]['score'].append(score)
                self.subject_accuracy[subj]['count']+=1
        
        if self.hparams.strategy == None : 
            pass
        elif 'ddp' in self.hparams.strategy and len(self.subject_accuracy) > 0:
            world_size = torch.distributed.get_world_size()
            total_subj_accuracy = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(total_subj_accuracy,self.subject_accuracy) # gather and broadcast to whole ranks     
            accuracy_dict = {}
            for dct in total_subj_accuracy:
                for subj, metric_dict in dct.items():
                    if subj not in accuracy_dict:
                        accuracy_dict[subj] = metric_dict
                    else:
                        accuracy_dict[subj]['score']+=metric_dict['score']
                        accuracy_dict[subj]['count']+=metric_dict['count']
            self.subject_accuracy = accuracy_dict
        if self.trainer.is_global_zero:
            for subj_name,subj_dict in self.subject_accuracy.items():
                subj_pred = np.mean(subj_dict['score'])
                subj_error = np.std(subj_dict['score'])
                subj_truth = subj_dict['truth'].item()
                subj_count = subj_dict['count']
                subj_mode = subj_dict['mode'] # train, val, test

                # only save samples at rank 0 (total iterations/world_size numbers are saved) 
                os.makedirs(os.path.join('predictions',self.hparams.id), exist_ok=True)
                with open(os.path.join('predictions',self.hparams.id,'iter_{}.txt'.format(self.current_epoch)),'a+') as f:
                    f.write('subject:{} ({})\ncount: {} outputs: {:.4f}\u00B1{:.4f}  -  truth: {}\n'.format(subj_name,subj_mode,subj_count,subj_pred,subj_error,subj_truth))

            with open(os.path.join('predictions',self.hparams.id,'iter_{}.pkl'.format(self.current_epoch)),'wb') as fw:
                pickle.dump(self.subject_accuracy, fw)

    def test_step(self, batch, batch_idx):
        if not hasattr(self, 'model_precision_checked'):
            for name, param in self.model.named_parameters():
                model_dtype = str(param.dtype)
                break
            if self.trainer.precision == 'bf16' and model_dtype != 'torch.bfloat16':
                self.model = self.model.to(dtype=torch.bfloat16)
                if hasattr(self, 'output_head') and self.output_head != None:
                    self.output_head = self.output_head.to(dtype=torch.bfloat16)
            else:
                1
                # raise NotImplementedError("do some work for fp16 or else")
            self.model_precision_checked = True

        return self.validation_step(batch, batch_idx, dataloader_idx=1)

    def test_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            # outputs can be a list (if one dataloader) or list of lists (if multiple)
            outputs_test = outputs[0] if isinstance(outputs[0], list) else outputs
            agg_test = self._aggregate_epoch_outputs(outputs_test)
            self._evaluate_metrics(agg_test, mode="test", best=True)
    
    def on_train_epoch_start(self) -> None:
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.total_time = 0
        self.repetitions = 50
        self.gpu_warmup = 10
        self.timings=np.zeros((self.repetitions,1))
        return super().on_train_epoch_start()
    
    def on_train_batch_start(self, batch, batch_idx):
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx-self.gpu_warmup) < self.repetitions:
                self.starter.record()
        return super().on_train_batch_start(batch, batch_idx)
    
    def on_train_batch_end(self, out, batch, batch_idx):
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx-self.gpu_warmup) < self.repetitions:
                self.ender.record()
                torch.cuda.synchronize()
                curr_time = self.starter.elapsed_time(self.ender) / 1000
                self.total_time += curr_time
                self.timings[batch_idx-self.gpu_warmup] = curr_time
            elif (batch_idx-self.gpu_warmup) == self.repetitions:
                mean_syn = np.mean(self.timings)
                std_syn = np.std(self.timings)
                
                Throughput = (self.repetitions*self.hparams.batch_size*int(self.hparams.num_nodes) * int(self.hparams.devices))/self.total_time
                
                self.log(f"Throughput", Throughput, sync_dist=False)
                print("Throughput:", self.hparams.num_nodes, self.hparams.devices, Throughput)
                self.log(f"mean_time", mean_syn, sync_dist=False)
                self.log(f"std_time", std_syn, sync_dist=False)
                print('mean_syn:',mean_syn)
                print('std_syn:',std_syn)

                import csv 

                data = [["num_nodes", "devices","Throughout","mean_time","std_time"], [self.hparams.num_nodes,self.hparams.devices,Throughput,mean_syn,std_syn ]]

                with open(f'scalability_testing_{self.hparams.num_nodes}_{self.hparams.devices}.csv', 'w', newline='') as f:     
                    writer = csv.writer(f)     
                    writer.writerows(data)
                
        return super().on_train_batch_end(out, batch, batch_idx)

    def configure_optimizers(self):
        # scale learning rate using global batch size
        orig_lr = self.hparams.learning_rate
        world_size = int(os.environ.get('SLURM_NTASKS', 384)) if not os.environ.get('WORLD_SIZE') else int(os.environ.get('WORLD_SIZE'))
        global_bsz = self.hparams.batch_size * world_size
        if self.hparams.lr_scaling == 'square': lr_batchsize_multiplier = np.sqrt(global_bsz / 384.0)
        elif self.hparams.lr_scaling == 'linear': lr_batchsize_multiplier = global_bsz / 384.0
        else: lr_batchsize_multiplier = 1

        effective_lr = self.hparams.learning_rate * lr_batchsize_multiplier
        print(f"Global Bsz: {global_bsz}. LR scaling: {self.hparams.lr_scaling} ({lr_batchsize_multiplier:.2f}x). Base LR: {orig_lr:.2e}. Effective LR: {effective_lr:.2e}")
        self.hparams.learning_rate = effective_lr # Update hparams for scheduler

        # --- Select Parameters to Optimize ---
        if not self.hparams.pretraining and self.hparams.freeze_feature_extractor:
             print("Optimizer targeting parameters of the output head ONLY.")
             optim_params = self.output_head.parameters()
             # Also ensure backbone requires_grad is False (done in init)
             for param in self.model.parameters(): assert not param.requires_grad
        else:
            # Optimize all/some parameters (pretraining or fine-tuning model)
            print("Optimizer targeting all/some model parameters.")
            optim_params = self.parameters()
            
        if self.hparams.optimizer == "SGD" : # and 'deepspeed' not in self.hparams.strategy:
            if self.hparams.use_MuTransfer:
                from mup import MuSGD
                optim_class = MuSGD
            else:
                optim_class = torch.optim.SGD
            optim = optim_class(
                optim_params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum
            )
        elif self.hparams.optimizer == "AdamW":
            if self.hparams.use_MuTransfer:
                from mup import MuAdam, MuAdamW
            if 'deepspeed' not in self.hparams.strategy:
                optim_class = torch.optim.AdamW if not self.hparams.use_MuTransfer else MuAdamW
            elif 'deepspeed' in self.hparams.strategy:
                from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
                if not self.hparams.use_MuTransfer:
                    optim_class = FusedAdam if 'offload' not in self.hparams.strategy else DeepSpeedCPUAdam
                else:
                    MuFusedAdam = lambda params, **kwargs: MuAdam(params, impl=FusedAdam, **kwargs)
                    MuDeepSpeedCPUAdam = lambda params, **kwargs: MuAdam(params, impl=DeepSpeedCPUAdam, **kwargs)
                    optim_class = MuFusedAdam if 'offload' not in self.hparams.strategy else MuDeepSpeedCPUAdam
            optim = optim_class(
                optim_params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        else:
            raise RuntimeError("Error: Input a correct optimizer name (default: AdamW)")
        
        
        if self.hparams.use_scheduler:
            total_iterations = int(len(self.trainer.datamodule.train_dataloader()) / (world_size * self.trainer.accumulate_grad_batches) * (self.hparams.max_epochs - self.current_epoch)) 
            gamma = self.hparams.gamma
            base_lr = self.hparams.learning_rate
            min_lr = base_lr / 100.0 # 241121 jubin added
            warmup = int(total_iterations * self.hparams.warmup) if not self.hparams['resume_ckpt_path'] else 0 # adjust the length of warmup here.
            T_0 = int(self.hparams.cycle * total_iterations)
            T_mult = 2
            
            if self.hparams.use_MuTransfer: # 240204 jubin added
                sche = CosineAnnealingWarmUpRestartsMup(optim, first_cycle_steps=T_0, cycle_mult=T_mult, min_lr_scale=0.01, warmup_steps=warmup, gamma=gamma)
            else:
                sche = CosineAnnealingWarmUpRestarts(optim, first_cycle_steps=T_0, cycle_mult=T_mult, max_lr=base_lr, min_lr=min_lr, warmup_steps=warmup, gamma=gamma)
            print('total iterations:', total_iterations * self.hparams.max_epochs)

            scheduler = {
                "scheduler": sche,
                "name": "lr_history",
                "interval": "step",
            }

            return [optim], [scheduler]
        else:
            return optim

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Default classifier")
        ## training related
        group.add_argument("--loss_type", type=str, default="L1", help="", choices=["L1", "smoothL1", "huber", "L2"])
        group.add_argument("--grad_clip", action='store_true', help="whether to use gradient clipping")
        group.set_defaults(grad_clip=False)
        group.add_argument("--optimizer", type=str, default="AdamW", help="which optimizer to use [AdamW, SGD]")
        group.add_argument("--use_scheduler", action='store_true', help="whether to use scheduler")
        group.set_defaults(use_scheduler=False)
        group.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for optimizer")
        group.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate for optimizer")
        group.add_argument("--lr_scaling", type=str, default='linear', help="scaling rule for learning rate")
        group.add_argument("--warmup", type=float, default=0.01, help="warmup in CosineAnnealingWarmUpRestarts (recommend 0.01~0.1 values)")
        group.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        group.add_argument("--gamma", type=float, default=0.5, help="decay for exponential LR scheduler")
        group.add_argument("--cycle", type=float, default=0.3, help="cycle size for CosineAnnealingWarmUpRestarts")
        group.add_argument("--milestones", nargs="+", default=[100, 150], type=int, help="lr scheduler")
        group.add_argument("--adjust_thresh", action='store_true', help="whether to adjust threshold for valid/test")
        
        ## pretraining-related
        group.add_argument("--pretraining", action='store_true', help="whether to use pretraining")
        group.set_defaults(pretraining=False)
        group.add_argument("--augment_during_training", action='store_true', help="whether to augment input images during training")
        group.set_defaults(augment_during_training=False)
        group.add_argument("--augment_only_affine", action='store_true', help="whether to only apply affine augmentation")
        group.add_argument("--augment_only_intensity", action='store_true', help="whether to only apply intensity augmentation")
        group.add_argument("--augment_strength", default=1, type=float, help="probability for randomly applying intensity augmentation")
        # contrastive learning (SwiFT v1)
        group.add_argument("--use_contrastive", action='store_true', help="whether to use contrastive learning (specify --contrastive_type argument as well)")
        group.add_argument("--contrastive_type", default=0, type=int, help="combination of contrastive losses to use [1: Use the Instance contrastive loss function, 2: Use the local-local temporal contrastive loss function, 3: Use the sum of both loss functions]")
        group.add_argument("--temperature", default=0.1, type=float, help="temperature for NTXentLoss")
        # masked image modeling (SwiFT v2)
        group.add_argument("--use_mim", action='store_true', help="whether to use masked image modeling (specify --masking_type argument as well)")
        group.set_defaults(use_mim=False)
        # mamba autoregressive
        group.add_argument("--use_autoregressive", action='store_true', help="whether to use autoregressive modeling")

        # for MAE style
        #group.add_argument("--mask_ratio", default=0.875, type=float, help="mask ratio for masked image modeling", choices=[0.0, 0.25, 0.65, 0.875, 1])
        #group.add_argument("--masking_temporal_stride", type=int, default=1, help="the range of time point that the same spatial masks are applied")
        #group.add_argument("--decoder_type", default='transformer', type=str, help="[linear: Use 1 linear layer, mlp: Use 2 layers mlp, transformer: Use 2 transformer blocks, swin: Use swin blocks", choices=['linear', 'mlp', 'transformer', 'swin'])
        #group.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
        #group.set_defaults(norm_pix_loss=False)
        # for simMIM style
        group.add_argument("--mask_ratio", default=0.6, type=float, help="mask ratio for masked image modeling")
        group.add_argument("--mask_patch_size", nargs="+", default=[6, 6, 6, 2], type=int, help="mask_patch size")
        group.add_argument("--masking_type", default='random', type=str, help="[random: Use random masking, tube: Use tube masking, temporal: Use temporal masking(mask last N TRs)", choices=['random', 'tube', 'temporal'])
        group.add_argument("--calc_loss_without_background", action='store_true', help="exclude background tokens of reconstruction loss")
        group.add_argument("--calc_loss_brain_mask", action='store_true', help="exclude all background values of reconstruction loss")
        group.add_argument("--save_mim_sample", action='store_true', help="option to save sample images for simMIM") # 250724 jubin added

        ## model related
        group.add_argument("--model", type=str, default="none", help="which model to be used")
        group.add_argument("--in_chans", type=int, default=1, help="Channel size of input image")
        group.add_argument("--embed_dim", type=int, default=24, help="embedding size (recommend to use 24, 36, 48)")
        group.add_argument("--window_size", nargs="+", default=[4, 4, 4, 4], type=int, help="window size from the second layers")
        group.add_argument("--first_window_size", nargs="+", default=[2, 2, 2, 2], type=int, help="first window size")
        group.add_argument("--patch_size", nargs="+", default=[6, 6, 6, 1], type=int, help="patch size")
        group.add_argument("--depths", nargs="+", default=[2, 2, 6, 2], type=int, help="depth of layers in each stage of encoder")
        group.add_argument("--num_heads", nargs="+", default=[3, 6, 12, 24], type=int, help="The number of heads for each attention layer")
        group.add_argument("--c_multiplier", type=int, default=2, help="channel multiplier for Swin Transformer architecture")
        group.add_argument("--last_layer_full_MSA", type=str2bool, default=False, help="whether to use full-scale multi-head self-attention at the last layers")
        group.add_argument("--clf_head_version", type=str, default="v1", help="clf head version, v2 has a hidden layer")
        group.add_argument("--attn_drop_rate", type=float, default=0, help="dropout rate of attention layers")
        group.add_argument("--drop_path_rate", type=float, default=0, help="Stochastic depth rate.")
        # SwiFT v2 specific 
        group.add_argument("--decoder_embed_dim", type=int, default=128, help="embedding size (recommend to use 128, 192, 256)")
        group.add_argument("--decoder_depth", type=int, default=2, help="depth of layers in each stage of decoder")
        group.add_argument("--decoder_num_heads", type=int, default=16, help="The number of heads for each attention layer of decoder")
        group.add_argument("--use_flashattn", action='store_true', help="using flash attention in the WindowAttention4D")
        group.set_defaults(use_flashattn=False)
        group.add_argument("--use_post_norm", action='store_true', help="using post normalization like swintransformer v2")
        group.set_defaults(use_post_norm=False)
        group.add_argument("--use_layer_embedding_mlp", action='store_true', help="using layer embedding mlp")
        # Mamba specific
        group.add_argument("--mamba_pe_method", type=str, default='nerf', help="option for fmamba positional encoding. ['none', 'ff', 'nerf', 'cpe'] ")
        group.add_argument("--init_mamba_weights", action='store_true', help="option for weight initialization. If True, use mamba specific weight init function")

        ## others
        group.add_argument("--use_MuTransfer", action='store_true', help="using MuTransfer that can help you search hyperparameters with less training experiments")
        group.set_defaults(use_MuTransfer=False)
        group.add_argument("--scalability_check", action='store_true', help="whether to check scalability")
        group.add_argument("--process_code", default=None, help="Slurm code/PBS code. Use this argument if you want to save process codes to your log")
        group.add_argument("--run_only_1epoch", action='store_true', help="Run only 1 epoch when resuming")
        group.add_argument("--save_embedding", action='store_true', help="Save fMRI embeddings") # 250407 jubin added
        group.add_argument("--embedding_save_dir", type=str, help="Path to save fMRI embeddings. If not given, they will be saved to {project_dir}/output/embeddings/{date +%y%m%d_%H%M%S}") # 250408 jubin added

        return parser