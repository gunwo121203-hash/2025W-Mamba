import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from dataclasses import dataclass
from einops import rearrange, repeat

from mamba_ssm import Mamba, Mamba2
class Embedder(nn.Module):
    def __init__(self, pe_method, embed_dim, learnable_projection = False):
        super(Embedder, self).__init__()
        assert pe_method in ['none', 'ff', 'nerf', 'cpe']
        self.embed = CoordinateEmbedder(method = pe_method, 
                                        n_continuous_dim = 4, 
                                        target_dim = embed_dim, 
                                        learnable_projection = learnable_projection)

    def forward(self, pos):
        pos_embed = self.embed(pos)  
        return pos_embed
        
class CoordinateEmbedder(nn.Module):
    """
    Three different continuous coordinate embedding methods are merged.
    1. Fourier features
    2. Nerf
    3. Continuous PE
    """
    
    def __init__(self, method = 'cpe', n_continuous_dim = 3, target_dim = 256, learnable_projection = False):
        super(CoordinateEmbedder, self).__init__()
        
        pseudo_input = torch.randn(1, 2, n_continuous_dim)
        
        if method == 'ff':
            self.get_ff(n_continuous_dim)
            out_dim = self.pec.forward(pseudo_input).shape[-1]
            # print('orig output_dim: {}'.format(out_dim))
            self.projection = nn.Parameter(torch.randn(out_dim, target_dim), requires_grad = learnable_projection)
            
        elif method == 'nerf':
            multires = 10
            self.get_nerf(multires, n_continuous_dim)
            out_dim = self.pec.forward(pseudo_input).shape[-1]
            # print('orig output_dim: {}'.format(out_dim))
            self.projection = nn.Parameter(torch.randn(out_dim, target_dim), requires_grad = learnable_projection)
                        
        elif method == 'cpe':
            self.get_cpe(n_continuous_dim, target_dim)
            self.projection = None     
            
        elif method == 'none':
            self.pec = nn.Identity()
            self.projection = nn.Parameter(torch.randn(n_continuous_dim, target_dim), requires_grad = learnable_projection)
            # self.projection = nn.Parameter(torch.randn(3, target_dim), requires_grad = learnable_projection)
            
    def apply_projection(self, tensor):
        return torch.matmul(tensor, self.projection)
    
    def get_ff(self, n_continuous_dim):
        pos2fourier_position_encoding_kwargs = dict(
        num_bands = [12] * n_continuous_dim,
        max_resolution = [20] * n_continuous_dim,
        )
        self.pec = FourierPositionEncoding(**pos2fourier_position_encoding_kwargs)

    def get_cpe(self, n_continuous_dim, target_dim):
        self.pec = PositionEmbeddingCoordsSine(n_dim = n_continuous_dim, d_model = target_dim)
    
    def get_nerf(self, multires, n_continuous_dim):
        embed_kwargs = {
                'include_input': True,
                'n_continuous_dim': n_continuous_dim,
                'max_freq_log2': multires-1,
                'num_freqs': multires,
                'log_sampling': True,
                'periodic_fns': [torch.sin, torch.cos],
            }
        self.pec = NerfEmbedder(**embed_kwargs)

    def forward(self, tensor):
        """
        tensor: b x N_seq x self.n_continuous_dim
        out: b x N_seq x self.target_dim
        """
        out = self.pec.forward(tensor)
        if self.projection is not None:
            out = self.apply_projection(out)
        return out

    
class FourierPositionEncoding():
    """ Fourier (Sinusoidal) position encoding. """

    def __init__(self, num_bands, max_resolution, concat_pos=True, sine_only=False):
        self.num_bands = num_bands
        self.max_resolution = max_resolution
        self.concat_pos = concat_pos
        self.sine_only = sine_only

    def output_size(self):
        """ Returns size of positional encodings last dimension. """
        encoding_size = sum(self.num_bands)
        if not self.sine_only:
            encoding_size *= 2
        if self.concat_pos:
            encoding_size += len(self.max_resolution)
        return encoding_size

    def forward(self, pos=None):
        fourier_pos_enc = generate_fourier_features(
            pos,
            num_bands=self.num_bands,
            max_resolution=self.max_resolution,
            concat_pos=self.concat_pos,
            sine_only=self.sine_only)
        return fourier_pos_enc


def generate_fourier_features(pos, num_bands, max_resolution=(2 ** 10), concat_pos=True, sine_only=False):
    """
    Generate a Fourier feature position encoding with linear spacing.

    Args:
        pos: The Tensor containing the position of n points in d dimensional space.
        num_bands: The number of frequency bands (K) to use.
        max_resolution: The maximum resolution (i.e., the number of pixels per dim). A tuple representing resoltuion for each dimension.
        concat_pos: Whether to concatenate the input position encoding to the Fourier features.
        sine_only: Whether to use a single phase (sin) or two (sin/cos) for each frequency band.
    """
    batch_size = pos.shape[0]
    min_freq = 1.0 
    stacked = []
    for i, (res, num_band) in enumerate(zip(max_resolution, num_bands)):       
        stacked.append(pos[..., i, None] * torch.linspace(start=min_freq, end=res / 2, steps=num_band)[None, :].to(device = pos.device))

    per_pos_features = torch.cat(stacked, dim=-1)  
    per_pos_features = torch.cat([torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)], dim=-1)
    per_pos_features = torch.cat([pos, per_pos_features], dim=-1)
    return per_pos_features


class NerfEmbedder:
    def __init__(self, n_continuous_dim, include_input, max_freq_log2, num_freqs, log_sampling, periodic_fns):
        
        self.n_continuous_dim = n_continuous_dim
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.n_continuous_dim 
        out_dim = 0
        
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    
    
class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
       arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super(PositionEmbeddingCoordsSine, self).__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb


def torchgengrid(steps=(32, 32, 32, 32), bot=(0, 0, 0, 0), top=(1, 1, 1, 1)):
    arrs = []
    for bot_, top_, step_ in zip(bot, top, steps):
        arrs.append(torch.linspace(bot_, top_, steps=step_))
    meshlist = torch.meshgrid(*arrs, indexing='ij')
    mesh = torch.stack(meshlist, dim=len(steps))
    return mesh


class PatchEmbed(nn.Module):
    """ 4D Image to Patch Embedding
    """
 
    def __init__(
        self,
        img_size=(96, 96, 96, 20),
        patch_size=(6, 6, 6, 2),
        in_chans=1,
        embed_dim=24,
        norm_layer=None,
        flatten=False,
        spatial_dims=3,
        learnable_pos_projection=False,
        learnable_x_projection=True,
        pe_method='nerf',
        calc_loss_without_background=False,
        input_scaling_method='znorm_minback'
    ):
        assert len(patch_size) == 4, "you have to give four numbers, each corresponds h, w, d, t"
        #assert patch_size[3] == 1, "temporal axis merging is not implemented yet"
 
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
            img_size[3] // patch_size[3],
        )
        self.embed_dim = embed_dim
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2] * self.grid_size[3]
        self.flatten = flatten
        self.orig_size = in_chans * patch_size[0] * patch_size[1] * patch_size[2] * patch_size[3]
        self.projection = nn.Parameter(torch.randn(self.orig_size, self.embed_dim), requires_grad = learnable_x_projection)
        mesh = torchgengrid(steps=self.grid_size, bot=(0, 0, 0, 0), top=(1, 1, 1, 1)) # (A,B,C,D,4)
        pos = Embedder(pe_method=pe_method, embed_dim=embed_dim, learnable_projection = learnable_pos_projection)(mesh).view(-1, embed_dim) # B, ..., embed_dim
        self.register_buffer('pos', pos.unsqueeze(0)) # (1,*grid_size, C)
        self.calc_loss_without_background = calc_loss_without_background
        self.input_scaling_method = input_scaling_method

    def patchify(self, x):
        B, C, D, H, W, T = x.shape
        pD, pH, pW, pT = self.grid_size
        sD, sH, sW, sT = self.patch_size

        x = x.view(B, C, pD, sD, pH, sH, pW, sW, pT, sT)
        patch_x = x.permute(0, 2, 4, 6, 8, 3, 5, 7, 9, 1).contiguous().view(B, -1, sD * sH * sW * sT * C) # B, num_patches, orig_size
        return patch_x

    def get_brain_token_mask(self, x):
        B, N, C = x.shape
        background_value = x.amin((1,2)) if self.input_scaling_method == 'znorm_minback' else torch.zeros(B, device=x.device) # consider znorm_zeroback or minmax scaling
        # find all tokens that have at least one voxel larger than background, and extend along the batch dimension to use shared maximum brain mask (match sequence length within batch)
        self.brain_token_mask = (x != background_value.view(B,1,1)).any((0,2)) # B, N   find a token that has at least one voxel that has a value larger than background, use maximum brain mask along batch dim

    def project(self, x):
        # Linear Projection
        x = torch.matmul(x, self.projection)
        x = x + self.pos
        return x
    
    def forward(self, x):
        # print(x.shape)
        B, C, D, H, W, T = x.shape
        # assert D == self.img_size[0], f"Input image height ({D}) doesn't match model ({self.img_size[0]})."
        # assert H == self.img_size[1], f"Input image width ({H}) doesn't match model ({self.img_size[1]})."
        # assert W == self.img_size[2], f"Input image width ({W}) doesn't match model ({self.img_size[2]})."
        patch_x = self.patchify(x)
        x = self.project(patch_x)
        if self.calc_loss_without_background:
            self.get_brain_token_mask(patch_x)
            orig_patch_x_shape = patch_x.shape
            patch_x = patch_x[:, self.brain_token_mask]  # keep only brain tokens
            x = x[:, self.brain_token_mask]  # keep only brain tokens
            if not hasattr(self, 'printed_shape'):
                print('shape of patch_x before/after removing background tokens: ', orig_patch_x_shape, patch_x.shape)
                self.printed_shape = True

        return x, patch_x
        

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device= None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * torch.sigmoid(x)


def initialize_fmamba(model: nn.Module, std: float = 0.02):
    """
    Initialize FMamba / Mamba2-style models with Normal(0, std) for weights.
    Rules:
      • nn.Linear / nn.Conv1d weights ~ N(0, std); biases = 0
      • RMSNorm (and any 'norm.weight') scales = 1
      • Mamba-specific 1D parameters:
          - dt_bias = 0
          - A_log   = 0   (yields A = -exp(0) = -1 if used as -exp(A_log))
          - D       = 1
    """
    with torch.no_grad():
        # Module-type based init (covers in_proj, out_proj, conv_xBC, output_layer, etc.)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    init.zeros_(module.bias)
 
            elif isinstance(module, nn.Conv1d):
                init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    init.zeros_(module.bias)

    print(f"✅ FMamba initialized (std={std})")


class FMamba(nn.Module):
    def __init__(self, embed_dim=512, num_layers=12, d_state=64, d_conv=4, expand=2, dropout=0.0,
                 img_size=(96, 96, 96, 20), patch_size=(6, 6, 6, 2), 
                 learnable_pos_projection=False, learnable_x_projection=True, pe_method='nerf',
                 calc_loss_without_background=False, input_scaling_method='znorm_minback',
                 remove_last_layer=False, init_weights=False, skip_last_norm=False,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedder = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
                                   learnable_pos_projection=learnable_pos_projection,
                                   learnable_x_projection=learnable_x_projection,
                                   pe_method=pe_method, 
                                   calc_loss_without_background=calc_loss_without_background,
                                   input_scaling_method=input_scaling_method,)
        self.num_layers = num_layers
        self.mamba_layers = nn.ModuleList(
            [nn.Sequential(RMSNorm(embed_dim), 
                           Mamba2(d_model=embed_dim, d_state=d_state, d_conv=d_conv, expand=expand),
                           nn.Dropout(dropout)) for _ in range(num_layers)]
        )
        self.output_layer = nn.Linear(embed_dim, self.embedder.orig_size) if not remove_last_layer else nn.Identity()
        self.norm = RMSNorm(embed_dim)
        self.remove_last_layer = remove_last_layer
        if init_weights and not remove_last_layer:
            initialize_fmamba(self, std=0.02)  # Initialize model weights
        self.skip_last_norm = skip_last_norm


    def forward(self, x, return_z = False):
        # x: B, C, D, H, W, T        
        x, x_orig_data = self.embedder(x)  # Add slight noise
        for layer in self.mamba_layers:
            z = layer(x)
            if return_z:
                feature = z
            x = z + x           
            
        x = self.norm(x) if not self.skip_last_norm else x

        output = [self.output_layer(x)]
        if not self.remove_last_layer: # use original data only when pretraining
            output.append(x_orig_data) 
        if return_z:
            output.append(feature)

        return output if len(output) > 1 else output[0]


class fmamba_mlp(nn.Module):
    def __init__(self, embed_dim = 96, num_classes=2):
        super(fmamba_mlp, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes)
        initialize_fmamba(self, std=0.02)  # Initialize model weights

    def forward(self, x):
        # x -> (b, num_tokens, C)
        x = self.avgpool(x.transpose(1, 2))  # B L C => B C L => B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class fmamba_conv_mlp(nn.Module):
    def __init__(self, embed_dim = 96,  num_classes=2, dropout_rate=0.0):
        super(fmamba_conv_mlp, self).__init__()
        # A sequence of conv blocks to hierarchically reduce sequence length
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=10, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=10, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=10, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=10, stride=2, padding=1),
            nn.GELU(),
        )
        
        # Final pooling and classification head
        self.final_pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(embed_dim, num_classes)
        initialize_fmamba(self, std=0.02)  # Initialize model weights

    def forward(self, x):
       # Input x shape: (Batch, num_tokens, embed_dim)
        x = x.transpose(1, 2)  # (B, embed_dim, num_tokens)
        x_conv = self.conv_blocks(x) # (B, embed_dim, reduced_num_tokens)
        x_pooled = self.final_pool(x_conv).flatten(1) # (B, embed_dim)
        x_norm = self.norm(x_pooled)
        x_out = self.head(self.dropout(x_norm))
        return x_out
    
class fmamba_mamba_head(nn.Module):
    def __init__(self, embed_dim = 96, num_classes=2, dropout_rate=0.0, init_weights=False):
        super(fmamba_mamba_head, self).__init__()
        # A Mamba block to act as a global information aggregator
        self.mamba_layers = nn.ModuleList(
            [nn.Sequential(Mamba(d_model=embed_dim, d_state=embed_dim//16),
                           nn.Dropout(dropout_rate),
                           RMSNorm(embed_dim)) for _ in range(3)]                           
        )
        
        self.head = nn.Linear(embed_dim, num_classes)
        if init_weights:
            initialize_fmamba(self, std=0.02)  # Initialize model weights

    def forward(self, x):
        # x -> (B, num_tokens, embed_dim)
        # Pass the sequence of tokens through the Mamba aggregator
        for layer in self.mamba_layers:
            z = layer(x)
            x = z + x           
        # Select the output of the LAST token in the sequence. This token's hidden state has seen all previous tokens.
        last_token_features = x[:, -1, :] # Shape: (B, embed_dim)
        
        # classify
        x_out = self.head(last_token_features)
        
        return x_out
    
    
class fmamba_mamba_head_finalnorm(nn.Module):
    def __init__(self, embed_dim = 96, num_classes=2, dropout_rate=0.0):
        super(fmamba_mamba_head_finalnorm, self).__init__()
        # A Mamba block to act as a global information aggregator
        self.mamba_layers = nn.ModuleList(
            [nn.Sequential(RMSNorm(embed_dim), 
                           Mamba(d_model=embed_dim, d_state=embed_dim//16),
                           nn.Dropout(dropout_rate)) for _ in range(3)]                           
        )
        self.norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        initialize_fmamba(self, std=0.02)  # Initialize model weights

    def forward(self, x):
        # x -> (B, num_tokens, embed_dim)
        # Pass the sequence of tokens through the Mamba aggregator
        for layer in self.mamba_layers:
            z = layer(x)
            x = z + x           
        # Select the output of the LAST token in the sequence. This token's hidden state has seen all previous tokens.
        last_token_features = x[:, -1, :] # Shape: (B, embed_dim)
        
        # classify
        x_out = self.head(self.norm(last_token_features))
        
        return x_out