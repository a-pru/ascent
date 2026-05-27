from typing import Optional

import torch
import torch.nn as nn
from timm.layers import DropPath
from torch import Tensor


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        post_norm=False,
        custom_forward=False,
        kdim=None,
        vdim=None,
        kvdim=None
    ):
        super().__init__()
        self.post_norm = post_norm
        self.custom_forward = custom_forward

        if self.custom_forward:
            #self.q_proj = nn.Linear(dim, dim)
            
            if kvdim is not None:
                self.normkv = norm_layer(kvdim)
            elif vdim is not None or kdim is not None:
                self.normk = norm_layer(kdim)
                self.normv = norm_layer(vdim)
                #self.k_proj = nn.Linear(dim, dim)
        self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
            kdim=kdim,
            vdim=vdim
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        src2, attn_weights = self.attn(
            query=src2,
            key=src2,
            value=src2,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src, attn_weights
    
    def forward_custom(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        kv: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
    ):
        assert (k is None and v is None) or (k is not None and v is not None)
        if k is not None:
            q = self.norm1(src)
            k = self.normk(k)
            v = self.normkv(v)
        elif kv is not None:
            q = self.norm1(src)
            k = v = self.normkv(kv) 
        else:
            q = k = v = self.norm1(src) 
            
        attn_output, attn_weights = self.attn(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )

        if torch.isnan(attn_output).sum() > 0:
            print(torch.isnan(attn_output).nonzero(as_tuple=True))
            print(torch.isnan(attn_output).sum(), q.shape, k.shape, v.shape, key_padding_mask.shape)
            print(key_padding_mask)
            assert False
        src = q + self.drop_path1(attn_output) 
        src = src + self.drop_path2(self.mlp(self.norm2(src))) 
        return src, attn_weights
    
    def forward_post(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ):
        src2, attn_weights = self.attn(
            query=src,
            key=src,
            value=src,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )
        src = src + self.drop_path1(self.norm1(src2))
        src = src + self.drop_path2(self.norm2(self.mlp(src)))
        return src, attn_weights

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        kv: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        weights: bool = False
    ):
        if self.custom_forward:
            out, attn_weights = self.forward_custom(
                src=src, kv=kv, mask=mask, k=k, v=v, key_padding_mask=key_padding_mask
            )
        else:
            assert type(kv) == type(None) and type(k) == type(None) and type(v) == type(None) 
            if self.post_norm:
                out, attn_weights = self.forward_post(
                    src=src, mask=mask, key_padding_mask=key_padding_mask
                )
            else:
                out, attn_weights = self.forward_pre(src=src, mask=mask, key_padding_mask=key_padding_mask)
        if not weights:
            return out
        else:
            return out, attn_weights
