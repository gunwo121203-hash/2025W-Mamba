import torch

from .fmamba import FMamba, fmamba_mlp, fmamba_conv_mlp, fmamba_mamba_head, fmamba_mamba_head_finalnorm

def load_model(model_name, hparams=None):
    #number of transformer stages
    n_stages = len(hparams.depths)

    if str(hparams.precision) in ['16', '16-mixed', 'bf16', 'bf16-mixed']:
        to_float = False
    elif hparams.precision == 32:
        to_float = True

    if model_name == 'fmamba':
        assert len(hparams.depths) == 1, "FMamba only supports a single depth value, which is a num_layers"
        assert hparams.embed_dim % 16 == 0, "FMamba embed_dim must be a multiple of 16"
        net = FMamba(
                embed_dim=hparams.embed_dim,
                num_layers=hparams.depths[0],
                d_state=hparams.embed_dim//4, 
                d_conv=4, 
                expand=2, 
                dropout=0.0,
                img_size=hparams.img_size,
                patch_size=hparams.patch_size, 
                learnable_pos_projection=False, 
                learnable_x_projection=True, 
                pe_method=hparams.get('mamba_pe_method', 'nerf'),
                remove_last_layer=(not hparams.get('pretraining', False)),
                calc_loss_without_background=hparams.get('calc_loss_without_background', False),
                input_scaling_method=hparams.get('input_scaling_method'), 
        )
        return net
    elif model_name == 'fmamba_mlp':
        num_classes = 2 if hparams.downstream_task_type == 'classification' else 1
        net = fmamba_mlp(embed_dim=hparams.embed_dim, num_classes=num_classes)
        return net
    elif model_name == 'fmamba_conv_mlp':
        num_classes = 2 if hparams.downstream_task_type == 'classification' else 1
        net = fmamba_conv_mlp(embed_dim=hparams.embed_dim, num_classes=num_classes)
        return net
    elif model_name == 'fmamba_mamba_head':
        num_classes = 2 if hparams.downstream_task_type == 'classification' else 1
        net = fmamba_mamba_head(embed_dim=hparams.embed_dim, num_classes=num_classes, init_weights=hparams.get("init_mamba_weights", False))
        return net
    elif model_name == 'fmamba_mamba_head_fn':
        num_classes = 2 if hparams.downstream_task_type == 'classification' else 1
        net = fmamba_mamba_head_finalnorm(embed_dim=hparams.embed_dim, num_classes=num_classes)
        return net


    return net
