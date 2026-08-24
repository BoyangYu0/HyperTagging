"""GPT-like/autoregressive model definitions from `graFEI_gpt`."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParticleEmbedder(torch.nn.Module):
    def __init__(
        self,
        n_features=11,
        tr_width=64,
        tr_n_head=8,
        tr_n=4,
        tr_hidden_size=2048,
        pdg_emb=5,
        dim_hyper=3,
        num_pdg=40,
        device="cuda:0"
    ):
        super().__init__()
        self.device = device
        self.pdg_emb = pdg_emb
        self.tr_width = tr_width
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb)
        self.projector = nn.Linear(in_features=n_features + pdg_emb, out_features=tr_width)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=tr_width, nhead=tr_n_head, dim_feedforward=tr_hidden_size, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=tr_n)

        self.particle_phi_v = torch.nn.Linear(in_features=tr_width, out_features=dim_hyper)
        self.particle_phi_n = torch.nn.Linear(in_features=tr_width, out_features=1)
        torch.nn.init.xavier_uniform_(self.particle_phi_n.weight, gain=nn.init.calculate_gain('sigmoid'))

        self.to(device)
        
    def forward(self, dataset):
        pdg = self.pdg_embedder(dataset['pdg'].to(self.device))
        feat = dataset['feature'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        att_input = self.projector(torch.cat([pdg, feat], axis=-1))
        transformed = self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask) 

        particle_v = F.normalize(self.particle_phi_v(transformed), dim=-1)
        particle_p = torch.sigmoid(self.particle_phi_n(transformed))
        particle_emb = particle_v * particle_p

        return particle_emb
        

class EmbLinker(nn.Module):
    def __init__(
        self,
        n_features=4,
        link_width=256,
        link_n_head=4,
        link_n_layers=12,
        link_fc=1024,
        device="cuda:0",
    ):
        super().__init__()
        self.projector = nn.Linear(n_features, link_width)
        self.device = device
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=link_width, nhead=link_n_head, dim_feedforward=link_fc, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=link_n_layers)

        self.to(device)
        
    def forward(self, dataset):
        feat_x = dataset['emb_x'].to(self.device)
        feat_y = dataset['emb_y'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        
        encoder_input = self.projector(torch.cat([feat_x, feat_y], axis=-2))
        att_mask = torch.cat([padding_mask, padding_mask], axis=-1)
        
        encoded = self.encoder(encoder_input, src_key_padding_mask=~att_mask)
        boundary = feat_x.shape[-2]
        return self.corr_matrix(encoded[:,:boundary,:], encoded[:,boundary:,:], padding_mask[...,None])

    def corr_matrix(self, data_x, data_y, mask):
        M_ix = (data_x*mask).unsqueeze(2).repeat(1,1,data_x.shape[1],1)
        M_iy = (data_y*mask).unsqueeze(1).repeat(1,data_y.shape[1],1,1)
        return F.cosine_similarity(M_ix, M_iy, dim=-1, eps=1e-6)
    

class GPTReconstructor(nn.Module):
    def __init__(
        self,
        tr_width=64,
        tr_n_head=8,
        tr_n=4,
        tr_hidden_size=2048,
        dim_hyper=3,
        device="cuda:0"
    ):
        super().__init__()
        self.device = device
        self.projector = nn.Linear(in_features=dim_hyper + 1, out_features=tr_width)
        self.tr_n_head = tr_n_head
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=tr_width, nhead=tr_n_head, dim_feedforward=tr_hidden_size, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=tr_n)

        self.particle_phi_v = torch.nn.Linear(in_features=tr_width, out_features=dim_hyper)
        self.particle_phi_n = torch.nn.Linear(in_features=tr_width, out_features=1)
        torch.nn.init.xavier_uniform_(self.particle_phi_n.weight, gain=nn.init.calculate_gain('sigmoid'))
        
        self.to(device)
        
    def forward(self, dataset):
        emb = dataset['emb'].to(self.device)
        src_mask = dataset['src_mask']
        if src_mask is not None:
            src_mask = src_mask.to(self.device)
            if src_mask.dtype is not torch.bool:
                src_mask = torch.isneginf(src_mask)
            src_mask = src_mask.repeat_interleave(self.tr_n_head, dim=0)
        lvl_code = dataset['lvl_code'].to(self.device)
        
        encoder_input = self.projector(torch.cat([emb, lvl_code.unsqueeze(-1)], axis=-1))
        transformed = self.encoder(encoder_input, mask=src_mask, src_key_padding_mask=~lvl_code.bool())
        particle_v = F.normalize(self.particle_phi_v(transformed), dim=-1)
        particle_p = torch.sigmoid(self.particle_phi_n(transformed))
        particle_emb = particle_v * particle_p

        return particle_emb


class MultiGPT(nn.Module):
    """Combined GPT-like embedding reconstruction and embedding-link model.

    The historical ``graFEI_gpt.models.MultiGPT`` class is not executable as
    written: it references undefined ``num_pdg``/``pdg_emb`` constructor
    values, uninitialized PDG/feature heads, and returns ``particle_emb``
    without assigning it. This migrated class preserves the verified branches
    used by the training script: autoregressive embedding reconstruction and
    link prediction from original/reconstructed embeddings.
    """

    def __init__(
        self,
        rec_width=64,
        rec_n_head=8,
        rec_n=4,
        rec_hidden_size=2048,
        link_width=64,
        link_n_head=8,
        link_n=4,
        link_hidden_size=2048,
        dim_hyper=3,
        device="cuda:0",
    ):
        super().__init__()
        self.device = device
        self.rec_n_head = rec_n_head
        self.link_n_head = link_n_head

        self.rec_projector = nn.Linear(in_features=dim_hyper + 1, out_features=rec_width)
        self.rec_encoder_layer = nn.TransformerEncoderLayer(
            d_model=rec_width,
            nhead=rec_n_head,
            dim_feedforward=rec_hidden_size,
            batch_first=True,
        )
        self.rec_encoder = nn.TransformerEncoder(self.rec_encoder_layer, num_layers=rec_n)
        self.particle_phi_v = nn.Linear(in_features=rec_width, out_features=dim_hyper)
        self.particle_phi_n = nn.Linear(in_features=rec_width, out_features=1)
        nn.init.xavier_uniform_(self.particle_phi_n.weight, gain=nn.init.calculate_gain("sigmoid"))

        self.link_projector = nn.Linear(in_features=dim_hyper + 1, out_features=link_width)
        self.link_encoder_layer = nn.TransformerEncoderLayer(
            d_model=link_width,
            nhead=link_n_head,
            dim_feedforward=link_hidden_size,
            batch_first=True,
        )
        self.link_encoder = nn.TransformerEncoder(self.link_encoder_layer, num_layers=link_n)

        self.to(device)

    def forward(self, dataset):
        emb = dataset["emb"].to(self.device)
        lvl_code = dataset["lvl_code"].to(self.device)
        padding_mask = lvl_code.bool()
        src_mask = dataset.get("src_mask")
        rec_mask = None
        if src_mask is not None:
            rec_mask = src_mask.to(self.device)
            if rec_mask.dtype is not torch.bool:
                rec_mask = torch.isneginf(rec_mask)
            rec_mask = rec_mask.repeat_interleave(self.rec_n_head, dim=0)

        rec_input = torch.cat([emb, lvl_code.unsqueeze(-1)], axis=-1)
        rec_transformed = self.rec_encoder(
            self.rec_projector(rec_input),
            mask=rec_mask,
            src_key_padding_mask=~padding_mask,
        )
        particle_v = F.normalize(self.particle_phi_v(rec_transformed), dim=-1)
        particle_p = torch.sigmoid(self.particle_phi_n(rec_transformed))
        particle_emb = particle_v * particle_p

        link_x = torch.cat([emb, lvl_code.unsqueeze(-1)], axis=-1)
        link_y = torch.cat([particle_emb, lvl_code.unsqueeze(-1)], axis=-1)
        link_input = self.link_projector(torch.cat([link_x, link_y], axis=-2))
        att_mask = torch.cat([padding_mask, padding_mask], axis=-1)
        link_transformed = self.link_encoder(link_input, src_key_padding_mask=~att_mask)
        boundary = emb.shape[-2]
        links = self.corr_matrix(
            link_transformed[:, :boundary, :],
            link_transformed[:, boundary:, :],
            padding_mask[..., None],
        )
        return particle_emb, links

    def corr_matrix(self, data_x, data_y, mask):
        M_ix = (data_x * mask).unsqueeze(2).repeat(1, 1, data_x.shape[1], 1)
        M_iy = (data_y * mask).unsqueeze(1).repeat(1, data_y.shape[1], 1, 1)
        return F.cosine_similarity(M_ix, M_iy, dim=-1, eps=1e-6)


__all__ = ["EmbLinker", "GPTReconstructor", "MultiGPT", "ParticleEmbedder"]
