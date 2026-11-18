# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited
#
# Adapted from scFoundation (https://github.com/biomap-research/scFoundation),
# licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Modified by InstaDeep Ltd (2026):
# only the encoder half of the "cell" model is kept.
#
# The scFoundation model weights are NOT covered by the Apache License: they are
# distributed under the scFoundation Model License (non-commercial research only),
# see https://github.com/biomap-research/scFoundation/blob/main/MODEL_LICENSE.
import torch
import torch.nn as nn


class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        bin_num: int,
        bin_alpha: float,
        mask_token_id: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha

        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)

        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)

        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mask_idx = (x == self.mask_token_id).nonzero()
        x_pad_idx = (x == self.pad_token_id).nonzero()

        x = self.mlp(x)  # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x)
        x_crosslayer = self.mlp2(x)
        x = self.bin_alpha * x + x_crosslayer
        weight = self.Softmax(x)

        token_emb = self.emb(torch.arange(self.bin_num, device=x.device))  # [H, D]
        x = torch.matmul(weight, token_emb)  # [B, N, D]

        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)
        mask_token_emb = self.emb_mask(tensor0).type(x.dtype)
        x[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = mask_token_emb.repeat(
            x_mask_idx.shape[0], 1
        )
        pad_token_emb = self.emb_pad(tensor0).type(x.dtype)
        x[x_pad_idx[:, 0], x_pad_idx[:, 1], :] = pad_token_emb.repeat(
            x_pad_idx.shape[0], 1
        )
        return x


class pytorchTransformerModule(nn.Module):  # noqa: N801 (original name)
    def __init__(
        self, max_seq_len: int, dim: int, depth: int, heads: int, ff_mult: int = 4
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.depth = depth
        self.transformer_encoder = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * ff_mult,
                batch_first=True,
                norm_first=False,
            )
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] <= self.max_seq_len
        for mod in self.transformer_encoder:
            x = mod(x, src_key_padding_mask=padding_mask)
        return self.norm(x)


class ScFoundationModel(nn.Module):
    """Encoder half of scFoundation's ``MaeAutobin`` (same parameter names)."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        enc = cfg["encoder"]
        self.token_emb = AutoDiscretizationEmbedding2(
            enc["hidden_dim"],
            cfg["seq_len"],
            bin_num=cfg["bin_num"],
            bin_alpha=cfg["bin_alpha"],
            pad_token_id=cfg["pad_token_id"],
            mask_token_id=cfg["mask_token_id"],
        )
        self.pos_emb = nn.Embedding(cfg["seq_len"] + 1, enc["hidden_dim"])
        self.encoder = pytorchTransformerModule(
            max_seq_len=cfg["seq_len"],
            dim=enc["hidden_dim"],
            depth=enc["depth"],
            heads=enc["heads"],
        )


class SCFoundationWrapper(nn.Module):
    """Cell embedding: concat of the last two positions + max + mean pooling.

    Pooling runs over the padded batch tensor, exactly as during training.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.model = ScFoundationModel(cfg)

    def forward(
        self,
        x: torch.Tensor,
        x_padding_mask: torch.Tensor,
        position_gene_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.model.token_emb(torch.unsqueeze(x, 2))
        x += self.model.pos_emb(position_gene_ids)
        gene_emb = self.model.encoder(x, x_padding_mask)
        gene_emb1 = gene_emb[:, -1, :]
        gene_emb2 = gene_emb[:, -2, :]
        gene_emb3, _ = torch.max(gene_emb[:, :-2, :], dim=1)
        gene_emb4 = torch.mean(gene_emb[:, :-2, :], dim=1)
        return torch.concat([gene_emb1, gene_emb2, gene_emb3, gene_emb4], dim=1)
