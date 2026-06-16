import torch
import torch.nn as nn
import torch.nn.functional as F

class InteractingLayer(nn.Module):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Arguments
            - **in_features** : Positive integer, dimensionality of input features.
            - **head_num**: int.The head number in multi-head self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, embedding_size, head_num=2, use_res=True, use_norm=True, scaling=True, seed=1024, device="cuda:0"):
        super(InteractingLayer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.use_norm = use_norm
        self.scaling = scaling
        self.seed = seed
        
        if self.use_norm:
            self.norm = nn.LayerNorm(embedding_size)

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs, mask=None):

        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
            
        if self.use_norm:
            inputs = self.norm(inputs)
            
        # None F D
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(
            inputs * mask[...,None], 
            self.W_key, dims=([-1], [0])
        )
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = F.softmax(inner_product, dim=-1)  # head_num None F F
        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.elu(result)

        return result

class SimpleInteractor(nn.Module):
    def __init__(
        self,
        n_features=11,
        int_n_head=4,
        int_n=3,
        tr_n_head=8,
        tr_n=4,
        tr_hidden_size=2048,
        pdg_emb=5,
        num_pdg=40,
        device="cuda:0"
    ):
        super().__init__()
        self.n_features = n_features + pdg_emb
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb)
        
        self.int_layers = nn.ModuleList(
            [InteractingLayer(self.n_features, int_n_head, device=device) for _ in range(int_n)])
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_features, nhead=tr_n_head, dim_feedforward=tr_hidden_size, 
            norm_first=False, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=tr_n)
        self.device = device
        self.to(device)
        
    def forward(self, dataset):
        padding_mask = dataset['padding_mask'].to(self.device)
        transformed = []
        for ds_pdg, ds_feat in zip(
            [dataset['pdg_x'],dataset['pdg_y']],
            [dataset['feature_x'],dataset['feature_y']]
        ):
            pdg = self.pdg_embedder(ds_pdg.to(self.device))
            feat = ds_feat.to(self.device)       
            att_input = torch.cat([pdg, feat], axis=-1)
            for layer in self.int_layers:
                att_input = layer(att_input, padding_mask)
            # src_key_padding_mask is inversely defined!!! True = Skip, False = Keep
            transformed.append(
                self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask)
            )
        return self.corr_matrix(transformed[0], transformed[1], padding_mask[...,None])
        # output: [batch, max_seq_len, max_seq_len]

    def corr_matrix(self, data_x, data_y, mask):
        M_ix = (data_x*mask).unsqueeze(2).repeat(1,1,data_x.shape[1],1)
        M_iy = (data_y*mask).unsqueeze(1).repeat(1,data_y.shape[1],1,1)
#         return torch.einsum('ixya,ixyb->ixy',M_ix,M_iy) # M_ixy
        return F.cosine_similarity(M_ix, M_iy, dim=-1)

class particleCombiner(nn.Module):
    def __init__(
        self,
        n_features=4,
        tr_width=64,
        tr_n_head=8,
        tr_n=4,
        tr_hidden_size=2048,
        pdg_emb=4,
        num_pdg=13,
        dim_hyper=16,
        device="cuda:0"
    ):
        super().__init__()
        n_features = pdg_emb + n_features + dim_hyper
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb) 
        self.projector = nn.Linear(in_features=n_features, out_features=tr_width)      
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=tr_width, nhead=tr_n_head, dim_feedforward=tr_hidden_size, 
            norm_first=False, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=tr_n)
        self.device = device
        self.to(device)
        
    def forward(self, dataset):
        pdg = self.pdg_embedder(dataset['pdg'].to(self.device))
        feat = dataset['feature'].to(self.device)
        emb = dataset['emb'].to(self.device).unsqueeze(1).repeat(1,feat.shape[1],1)
        padding_mask = dataset['padding_mask'].to(self.device)
        att_input = self.projector(torch.cat([pdg, feat, emb], axis=-1))
        # src_key_padding_mask is inversely defined!!! True = Skip, False = Keep
        transformed = self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask)
        output = self.cal_cos_matrix(transformed * padding_mask[...,None])
        
        return output
    
    def cal_cos_matrix(self, data):
        M_ia = data.unsqueeze(2).repeat(1,1,data.shape[1],1)
        M_ib = data.unsqueeze(1).repeat(1,data.shape[1],1,1)
        return F.cosine_similarity(M_ia, M_ib, dim=-1) # M_iab

class pretrain_HTR(torch.nn.Module):
    def __init__(
        self,
        n_features=11,
        int_n_head=4,
        int_n=3,
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
        self.n_features = n_features + pdg_emb
        self.pdg_emb = pdg_emb
        self.tr_width = tr_width
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb)
        
        self.int_layers = nn.ModuleList(
            [InteractingLayer(self.n_features, int_n_head, device=device) for _ in range(int_n)])
        self.projector = nn.Linear(in_features=self.n_features, out_features=tr_width)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=tr_width, nhead=tr_n_head, dim_feedforward=tr_hidden_size, 
            norm_first=False, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=tr_n)
        self.phi_vector = torch.nn.Linear(in_features=tr_width, out_features=dim_hyper)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.phi_norm = torch.nn.Linear(in_features=tr_width, out_features=1)
        torch.nn.init.xavier_uniform_(self.phi_norm.weight, gain=torch.nn.init.calculate_gain('sigmoid'))
        self.to(device)
        
    def forward(self, dataset):
        # pretrained - feature representation
        pdg = self.pdg_embedder(dataset['pdg'].to(self.device))
        feat = dataset['feature'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        att_input = torch.cat([pdg, feat], axis=-1)
        for layer in self.int_layers:
            att_input = layer(att_input, padding_mask)
        
        # main 
        # src_key_padding_mask is inversely defined!!! True = Skip, False = Keep
        att_input = self.projector(att_input)
        transformed = self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask).permute(0,2,1)
        post_norm = nn.LayerNorm([self.tr_width, padding_mask.shape[1]]).to(self.device)
        features = self.gap(post_norm(transformed)).squeeze() # euclidean output: [batch, n_features + pdg_emb]

        v = F.normalize(self.phi_vector(features))
        p = torch.sigmoid(self.phi_norm(features))
        return p * v # hyperbolic output: [batch, dim_hyper]        

# hst
class HypDecoder(nn.Module):
    def __init__(
        self,
        n_features=4,
        tr_n_head=8,
        tr_n=4,
        tr_hidden_size=2048,
        pdg_emb=5,
        dim_hyper=3,
        KD_n_head=1,
        KD_n=3,
        KD_hidden_size=256,
        device="cuda:0",
        num_pdg=40
    ):
        super().__init__()
        self.n_units = n_features + pdg_emb
        self.pdg_emb = pdg_emb
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb)
        self.device = device
        # Event embedding - HTR
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_units, nhead=tr_n_head, dim_feedforward=tr_hidden_size, 
            norm_first=False, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=tr_n)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.phi_dir = nn.Linear(in_features=self.n_units, out_features=dim_hyper)
        # Knowledge transfer
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_hyper, nhead=KD_n_head, dim_feedforward=KD_hidden_size, 
            norm_first=False, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=KD_n)
        self.to(device)
        
    def forward(self, dataset):
        # prepare
        pdg = self.pdg_embedder(dataset['pdg'].to(self.device))
        feat = dataset['feature'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        r = 0.6 * torch.sqrt(1-dataset['E_Rec'].to(self.device)) + 0.3
        att_input = torch.cat([pdg, feat], axis=-1)
        # HTR
        # src_key_padding_mask is inversely defined!!! True = Skip, False = Keep
        transformed = self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask).permute(0,2,1)
        post_norm = nn.LayerNorm([self.n_units, padding_mask.shape[1]]).to(self.device)
        features = self.gap(post_norm(transformed)).squeeze() # euclidean output: [batch, n_features + pdg_emb]
        new = self.phi_dir(features)
        # KT
        memory = dataset['emb_fsp'].float().to(self.device)
        out = r[:,None].float() * F.normalize(self.transformer_decoder(memory, new))
        return out # hyperbolic output: [batch, dim_hyper]    
    
class Generator(nn.Module):
    def __init__(
        self,
        n_features=4,
        int_n_head=4,
        int_n=3,
        gen_tr_width=64,
        gen_encoder_n_head=8,
        gen_encoder_n_layers=4,
        gen_encoder_fc=2048,
        gen_decoder_n_head=8,
        gen_decoder_n_layers=4,
        gen_decoder_fc=2048,
        pdg_emb=5,
        dim_hyper=3,
        device="cuda:0",
        num_pdg=540
    ):
        super().__init__()
        self.n_units = n_features + pdg_emb
        self.pdg_emb = pdg_emb
        self.gen_tr_width = gen_tr_width
        self.num_pdg = num_pdg
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb)
        self.device = device
        # Encoder
        self.int_layers = nn.ModuleList(
            [InteractingLayer(self.n_units, int_n_head, device=device) for _ in range(int_n)]) 
        self.projector = nn.Linear(in_features=self.n_units, out_features=gen_tr_width)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=gen_tr_width, nhead=gen_encoder_n_head, dim_feedforward=gen_encoder_fc, 
            norm_first=False, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=gen_encoder_n_layers)
        self.phi_vector = torch.nn.Linear(in_features=gen_tr_width, out_features=dim_hyper)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.phi_norm = torch.nn.Linear(in_features=gen_tr_width, out_features=1)
        
        # Decoder
        self.projector_decoder = nn.Linear(in_features=self.n_units+dim_hyper, out_features=gen_tr_width)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=gen_tr_width, nhead=gen_decoder_n_head, dim_feedforward=gen_decoder_fc, 
            norm_first=False, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=gen_decoder_n_layers)
        
        # Multi-task heads
        # PDG Head 
        self.pdg_head = nn.Sequential(
            nn.Linear(gen_tr_width, 128), nn.ReLU(), 
            nn.Linear(128,32), nn.ReLU(),
            nn.Linear(32,num_pdg+1)
        )         
        self.combined_head = nn.Linear(gen_tr_width+num_pdg+1, 128)
        self.p_head = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(128,32), nn.ReLU(),
            nn.Linear(32,3)
        )         
        self.e_head = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(128,16), nn.ReLU(),
            nn.Linear(16,1), nn.ReLU()
        )
        
        
        self.to(device)
        
    def forward(self, dataset):
        # prepare
        pdg = self.pdg_embedder(dataset['pdg_x'].to(self.device))
        feat = dataset['feature_x'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        particle_info = torch.cat([pdg, feat], axis=-1)
        att_input = particle_info
        for layer in self.int_layers:
            att_input = layer(att_input, padding_mask)
        att_input = self.projector(att_input)
        # main 
        # src_key_padding_mask is inversely defined!!! True = Skip, False = Keep
        encoded_particle = self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask)
        post_norm = nn.LayerNorm([self.gen_tr_width, padding_mask.shape[1]]).to(self.device)
        global_features = self.gap(post_norm(encoded_particle.permute(0,2,1))).squeeze(-1) 
        v = F.normalize(self.phi_vector(global_features))
        p = torch.sigmoid(self.phi_norm(global_features))
        emb = p * v
        
        # Decoder
        att_input = torch.cat([particle_info,emb.unsqueeze(1).repeat((1,padding_mask.shape[1],1))],axis=-1)
        att_input = self.projector_decoder(att_input)
        decoded_particle = self.transformer_decoder(
            att_input, encoded_particle, 
            tgt_key_padding_mask=~padding_mask, memory_key_padding_mask=~padding_mask
        )
        # PDG head
        pdg_out = self.pdg_head(decoded_particle)
        # feature head
        combined = self.combined_head(torch.cat([decoded_particle,pdg_out.detach()],axis=-1))
        feat_out = torch.cat(
            [
                self.p_head(combined),
                (
                    padding_mask/padding_mask.sum(dim=-1,keepdim=True) * self.e_head(combined).squeeze()
                    ).cumsum(dim=-1).flip(dims=[-1]).unsqueeze(-1)
                ], axis=-1
            )
        
        return pdg_out, feat_out
        # output: [batch, max_seq_len, num_pdg+1], [batch, max_seq_len, 4]
        
class Linker(nn.Module):
    def __init__(
        self,
        n_features=4,
        link_width=256,
        link_n_head=4,
        link_n_layers=12,
        link_fc=1024,
        pdg_emb=8,
        device="cuda:0",
        num_pdg=540
    ):
        super().__init__()
        self.pdg_emb = pdg_emb
        self.num_pdg = num_pdg
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb)
        self.projector = nn.Linear(n_features + pdg_emb, link_width)
        self.device = device
        # Encoders
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=link_width, nhead=link_n_head, dim_feedforward=link_fc, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=link_n_layers)

        self.to(device)
        
    def forward(self, dataset):
        # prepare
        pdg_x = self.pdg_embedder(dataset['pdg_x'].to(self.device))
        feat_x = dataset['feature_x'].to(self.device)
        pdg_y = self.pdg_embedder(dataset['pdg_y'].to(self.device))
        feat_y = dataset['feature_y'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        
        encoder_input_x = self.projector(torch.cat([pdg_x, feat_x], axis=-1))
        encoder_input_y = self.projector(torch.cat([pdg_y, feat_y], axis=-1))
        # Encoders
        encoded_x = self.encoder(encoder_input_x, src_key_padding_mask=~padding_mask)
        encoded_y = self.encoder(encoder_input_y, src_key_padding_mask=~padding_mask)
        return self.corr_matrix(encoded_x, encoded_y, padding_mask[...,None])
        # output: [batch, max_seq_len, max_seq_len]

    def corr_matrix(self, data_x, data_y, mask):
        M_ix = (data_x*mask).unsqueeze(2).repeat(1,1,data_x.shape[1],1)
        M_iy = (data_y*mask).unsqueeze(1).repeat(1,data_y.shape[1],1,1)
#         return torch.einsum('ixya,ixyb->ixy',M_ix,M_iy) # M_ixy
        return F.cosine_similarity(M_ix, M_iy, dim=-1, eps=1e-6)
    
class linearLinker(nn.Module):
    def __init__(
        self,
        n_features=4,
        link_width=256,
        link_n_head=4,
        link_n_layers=12,
        link_fc=1024,
        pdg_emb=8,
        device="cuda:0",
        num_pdg=540
    ):
        super().__init__()
        self.pdg_emb = pdg_emb
        self.num_pdg = num_pdg
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb)
        self.projector = nn.Linear(n_features + pdg_emb, link_width)
        self.device = device
        # Encoders
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=link_width, nhead=link_n_head, dim_feedforward=link_fc, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=link_n_layers)

        self.to(device)
        
    def forward(self, dataset):
        # prepare
        pdg_x = self.pdg_embedder(dataset['pdg_x'].to(self.device))
        feat_x = dataset['feature_x'].to(self.device)
        pdg_y = self.pdg_embedder(dataset['pdg_y'].to(self.device))
        feat_y = dataset['feature_y'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        
        pdg = torch.cat([pdg_x, pdg_y], axis=-2)
        feat = torch.cat([feat_x, feat_y], axis=-2)
        att_mask = torch.cat([padding_mask, padding_mask], axis=-1)
        
        encoder_input = self.projector(torch.cat([pdg, feat], axis=-1))
        # Encoders
        encoded = self.encoder(encoder_input, src_key_padding_mask=~att_mask)
        boundary = int(pdg.shape[-2]/2)
        return self.corr_matrix(encoded[:,:boundary,:], encoded[:,boundary:,:], padding_mask[...,None])
        # output: [batch, max_seq_len, max_seq_len]

    def corr_matrix(self, data_x, data_y, mask):
        M_ix = (data_x*mask).unsqueeze(2).repeat(1,1,data_x.shape[1],1)
        M_iy = (data_y*mask).unsqueeze(1).repeat(1,data_y.shape[1],1,1)
#         return torch.einsum('ixya,ixyb->ixy',M_ix,M_iy) # M_ixy
        return F.cosine_similarity(M_ix, M_iy, dim=-1, eps=1e-6)

class HyperEmbedder(torch.nn.Module):
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
        self.particle_importance = nn.Linear(in_features=tr_width, out_features=1)
        self.phi_vector = torch.nn.Linear(in_features=tr_width, out_features=dim_hyper)

        self.phi_norm = torch.nn.Linear(in_features=tr_width, out_features=1)
        torch.nn.init.xavier_uniform_(self.phi_norm.weight, gain=torch.nn.init.calculate_gain('sigmoid'))
        self.to(device)
        
    def forward(self, dataset):
        pdg = self.pdg_embedder(dataset['pdg'].to(self.device))
        feat = dataset['feature'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        att_input = self.projector(torch.cat([pdg, feat], axis=-1))
        # src_key_padding_mask is inversely defined!!! True = Skip, False = Keep
        transformed = self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask) # features: [batch, n_particles, width]
        particle_weights = F.softmax(
            self.particle_importance(transformed).squeeze(-1).masked_fill_(~padding_mask,-1e9),
            dim=-1)
        features = F.adaptive_avg_pool1d(
            transformed.permute(0,2,1) * particle_weights.unsqueeze(1), 1
            ).squeeze(-1) # euclidean output: [batch, width]
        v = F.normalize(self.phi_vector(features))
        p = torch.sigmoid(self.phi_norm(features))
        return p * v # hyperbolic output: [batch, dim_hyper]        
    
class Reconstructor(nn.Module):
    def __init__(
        self,
        n_features=4,
        gen_tr_width=64,
        gen_encoder_n_head=8,
        gen_encoder_n_layers=4,
        gen_encoder_fc=2048,
        gen_decoder_n_head=8,
        gen_decoder_n_layers=4,
        gen_decoder_fc=2048,
        pdg_emb=5,
        dim_hyper=3,
        device="cuda:0",
        num_pdg=540
    ):
        super().__init__()
        self.pdg_emb = pdg_emb
        self.gen_tr_width = gen_tr_width
        self.num_pdg = num_pdg
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb)
        self.device = device
        self.decoder_broadener = 2
        # Encoder
        self.projector = nn.Linear(in_features=n_features+pdg_emb, out_features=gen_tr_width)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=gen_tr_width, nhead=gen_encoder_n_head, dim_feedforward=gen_encoder_fc, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=gen_encoder_n_layers)
        self.particle_importance = nn.Linear(in_features=gen_tr_width, out_features=1)
        # self.phi_vector = torch.nn.Linear(in_features=gen_tr_width, out_features=dim_hyper)
        # self.phi_norm = torch.nn.Linear(in_features=gen_tr_width, out_features=1)

        decoder_width = self.decoder_broadener * gen_tr_width
        # Decoder
        self.projector_decoder = nn.Linear(in_features=n_features+pdg_emb+gen_tr_width, out_features=decoder_width)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_width, nhead=gen_decoder_n_head, dim_feedforward=gen_decoder_fc, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=gen_decoder_n_layers)
        
        # Multi-task heads
        # PDG Head 
        self.pdg_head = nn.Sequential(
            nn.Linear(decoder_width, 256), nn.ReLU(), 
            nn.Linear(256, 64), nn.ReLU(), 
            nn.Linear(64,num_pdg+1)
        )         
        self.combined_head = nn.Linear(decoder_width+num_pdg+1, 256)
        self.p_head = nn.Sequential(
            nn.Linear(256,64), nn.ReLU(),
            nn.Linear(64,8), nn.ReLU(),
            nn.Linear(8,3)
        )         
        self.e_head = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(256,64), nn.ReLU(),
            nn.Linear(64,8), nn.ReLU(),
            nn.Linear(8,1), nn.ReLU()
        )
        
        self.to(device)
        
    def forward(self, dataset):
        # prepare
        pdg = self.pdg_embedder(dataset['pdg_x'].to(self.device))
        feat = dataset['feature_x'].to(self.device)
        # emb = dataset['emb'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        particle_info = torch.cat([pdg, feat], axis=-1)
        att_input = self.projector(particle_info)
        # main 
        # src_key_padding_mask is inversely defined!!! True = Skip, False = Keep
        encoded_particle = self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask)
        
        # embedding
        particle_weights = F.softmax(
            self.particle_importance(encoded_particle).squeeze(-1).masked_fill_(~padding_mask,-1e9),
            dim=-1)
        features = F.adaptive_avg_pool1d(
            encoded_particle.permute(0,2,1) * particle_weights.unsqueeze(1), 1
            ).squeeze(-1) # euclidean output: [batch, width]
        # v = F.normalize(self.phi_vector(features))
        # p = torch.sigmoid(self.phi_norm(features))
        # emb = p * v

        # Decoder
        att_input = torch.cat([particle_info,features.unsqueeze(1).repeat((1,padding_mask.shape[1],1))],axis=-1)
        att_input = self.projector_decoder(att_input)
        decoded_particle = self.transformer_decoder(
            att_input, encoded_particle.repeat(1,1,self.decoder_broadener), 
            tgt_key_padding_mask=~padding_mask, memory_key_padding_mask=~padding_mask
        )
        
        # PDG head
        pdg_out = self.pdg_head(decoded_particle)
        # feature head
        combined = self.combined_head(torch.cat([decoded_particle,pdg_out.detach()],axis=-1))
        feat_out = torch.cat(
            [self.p_head(combined),self.e_head(combined)], axis=-1
            )
        
        return pdg_out, feat_out
        # output: [batch, max_seq_len, num_pdg+1], [batch, max_seq_len, 4]

class doubleReconstructor(nn.Module):
    def __init__(
        self,
        n_features=4,
        gen_tr_width=64,
        gen_encoder_n_head=8,
        gen_encoder_n_layers=4,
        gen_encoder_fc=2048,
        gen_decoder_n_head=8,
        gen_decoder_n_layers=4,
        gen_decoder_fc=2048,
        pdg_emb=5,
        dim_hyper=3,
        device="cuda:0",
        num_pdg=540
    ):
        super().__init__()
        self.pdg_emb = pdg_emb
        self.gen_tr_width = gen_tr_width
        self.num_pdg = num_pdg
        self.pdg_embedder = nn.Embedding(num_pdg + 1,pdg_emb)
        self.device = device
        # Encoder
        self.projector = nn.Linear(in_features=n_features+pdg_emb, out_features=gen_tr_width)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=gen_tr_width, nhead=gen_encoder_n_head, dim_feedforward=gen_encoder_fc, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=gen_encoder_n_layers)
        self.particle_importance = nn.Linear(in_features=gen_tr_width, out_features=1)
        self.phi_vector = torch.nn.Linear(in_features=gen_tr_width, out_features=dim_hyper)
        self.phi_norm = torch.nn.Linear(in_features=gen_tr_width, out_features=1)

        # PDG Decoder
        self.PDG_projector = nn.Linear(in_features=n_features+pdg_emb+dim_hyper, out_features=gen_tr_width)
        self.PDG_decoder_layer = nn.TransformerDecoderLayer(
            d_model=gen_tr_width, nhead=gen_decoder_n_head, dim_feedforward=gen_decoder_fc, batch_first=True
        )
        self.PDG_decoder = nn.TransformerDecoder(self.PDG_decoder_layer, num_layers=gen_decoder_n_layers)
        self.PDG_head = nn.Linear(gen_tr_width,num_pdg+1)

        # Feature Decoder
        self.Feat_projector = nn.Linear(in_features=n_features+pdg_emb+dim_hyper, out_features=gen_tr_width)
        self.Feat_decoder_layer = nn.TransformerDecoderLayer(
            d_model=gen_tr_width, nhead=gen_decoder_n_head, dim_feedforward=gen_decoder_fc, batch_first=True
        )
        self.Feat_decoder = nn.TransformerDecoder(self.Feat_decoder_layer, num_layers=gen_decoder_n_layers)
        self.p_head = nn.Linear(gen_tr_width+num_pdg+1,3)      
        self.e_head = nn.Linear(gen_tr_width+num_pdg+1,1)
        
        self.to(device)
        
    def forward(self, dataset):
        # prepare
        pdg = self.pdg_embedder(dataset['pdg_x'].to(self.device))
        feat = dataset['feature_x'].to(self.device)
        # emb = dataset['emb'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        particle_info = torch.cat([pdg, feat], axis=-1)
        att_input = self.projector(particle_info)
        # main 
        # src_key_padding_mask is inversely defined!!! True = Skip, False = Keep
        encoded_particle = self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask)
        
        # embedding
        particle_weights = F.softmax(
            self.particle_importance(encoded_particle).squeeze(-1).masked_fill_(~padding_mask,-1e9),
            dim=-1)
        features = F.adaptive_avg_pool1d(
            encoded_particle.permute(0,2,1) * particle_weights.unsqueeze(1), 1
            ).squeeze(-1) # euclidean output: [batch, width]
        v = F.normalize(self.phi_vector(features))
        p = torch.sigmoid(self.phi_norm(features))
        emb = p * v

        # Decoder
        att_input = torch.cat([particle_info,emb.unsqueeze(1).repeat((1,padding_mask.shape[1],1))],axis=-1)
        
        # PDG Decoder
        PDG_decoded_particle = self.PDG_decoder(
            self.PDG_projector(att_input), encoded_particle, 
            tgt_key_padding_mask=~padding_mask, memory_key_padding_mask=~padding_mask
        )
        PDG_out = self.PDG_head(PDG_decoded_particle)

        # Feature Decoder
        Feat_decoded_particle = self.Feat_decoder(
            self.Feat_projector(att_input), encoded_particle, 
            tgt_key_padding_mask=~padding_mask, memory_key_padding_mask=~padding_mask
        )
        Feat_combined = torch.cat([Feat_decoded_particle,PDG_out.detach()],axis=-1)
        Feat_out = torch.cat(
            [self.p_head(Feat_combined), F.relu(self.e_head(Feat_combined))], axis=-1
            )
        
        return PDG_out, Feat_out
        # output: [batch, max_seq_len, num_pdg+1], [batch, max_seq_len, 4]

class compReconstructor(nn.Module):
    def __init__(
        self,
        n_features=4,
        gen_tr_width=64,
        gen_encoder_n_head=8,
        gen_encoder_n_layers=4,
        gen_encoder_fc=2048,
        dim_hyper=3,
        device="cuda:0",
        num_pdg=540
    ):
        super().__init__()
        self.gen_tr_width = gen_tr_width
        self.num_pdg = num_pdg
        self.device = device
        # Encoder
        self.projector = nn.Linear(in_features=num_pdg+1+n_features+dim_hyper, out_features=gen_tr_width)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=gen_tr_width, nhead=gen_encoder_n_head, dim_feedforward=gen_encoder_fc, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=gen_encoder_n_layers)
        self.back_projector = nn.Linear(in_features=gen_tr_width, out_features=num_pdg+1+n_features)

        # Multi-task heads
        # PDG Head 
        # self.pdg_head = nn.Sequential(
        #     nn.Linear(num_pdg+1, 256), nn.ReLU(), 
        #     nn.Linear(256, 64), nn.ReLU(), 
        #     nn.Linear(64,num_pdg+1)
        # )         
        # self.combined_head = nn.Linear(gen_tr_width+num_pdg+1, 256)
        # self.p_head = nn.Sequential(
        #     nn.Linear(256,64), nn.ReLU(),
        #     nn.Linear(64,8), nn.ReLU(),
        #     nn.Linear(8,3)
        # )         
        # self.e_head = nn.Sequential(
        #     nn.ReLU(), 
        #     nn.Linear(256,64), nn.ReLU(),
        #     nn.Linear(64,8), nn.ReLU(),
        #     nn.Linear(8,1), nn.ReLU()
        # )
        
        self.to(device)
        
    def forward(self, dataset):
        # prepare
        pid = dataset['pid_x'].to(self.device)
        feat = dataset['p4_x'].to(self.device)
        emb = dataset['emb'].to(self.device).to(self.device).unsqueeze(1).repeat(1,feat.shape[1],1)
        padding_mask = dataset['padding_mask'].to(self.device)
        att_input = self.projector(torch.cat([pid, feat, emb], axis=-1))
        # main
        # src_key_padding_mask is inversely defined!!! True = Skip, False = Keep
        transformed = self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask)
        
        # # PDG head
        # pdg_out = self.pdg_head(transformed)
        # # feature head
        # combined = self.combined_head(torch.cat([transformed,pdg_out.detach()],axis=-1))
        # feat_out = torch.cat(
        #     [self.p_head(combined),self.e_head(combined)], axis=-1
        #     )
        output = self.back_projector(transformed)
        
        # return pdg_out, feat_out
        return output[...,:self.num_pdg+1], output[...,self.num_pdg+1:]
        # output: [batch, max_seq_len, num_pdg+1], [batch, max_seq_len, 4]

class DNNReconstructor(nn.Module):
    def __init__(
        self,
        n_features=4,
        gen_dnn_width=64,
        gen_dnn_n_layers=4,
        dim_hyper=3,
        device="cuda:0",
        num_pdg=540
    ):
        super().__init__()
        self.num_pdg = num_pdg
        self.device = device
        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(nn.Linear(num_pdg+1+n_features+dim_hyper, gen_dnn_width))
        for i in range(gen_dnn_n_layers):
            self.encoder_layers.append(nn.Linear(gen_dnn_width, gen_dnn_width))
            
        l1_width = gen_dnn_width * 2
        l2_width = int(gen_dnn_width / 2)
        l3_width = int(gen_dnn_width / 4)
        # Multi-task heads
        # PDG Head 
        self.pdg_head = nn.Sequential(
            nn.Linear(gen_dnn_width, l1_width), nn.ReLU(), 
            nn.Linear(l1_width, l2_width), nn.ReLU(), 
            nn.Linear(l2_width, num_pdg+1)
        )         
        self.combined_head = nn.Linear(gen_dnn_width+num_pdg+1, l1_width)
        self.p_head = nn.Sequential(
            nn.Linear(l1_width, l2_width), nn.ReLU(),
            nn.Linear(l2_width, l3_width), nn.ReLU(),
            nn.Linear(l3_width, 3)
        )         
        self.e_head = nn.Sequential(
            nn.ReLU(), 
            nn.Linear(l1_width, l2_width), nn.ReLU(),
            nn.Linear(l2_width, l3_width), nn.ReLU(),
            nn.Linear(l3_width, 1), nn.ReLU()
        )
        
        self.to(device)
        
    def forward(self, dataset):
        # prepare
        pid = dataset['pid_x'].to(self.device)
        feat = dataset['p4_x'].to(self.device)
        emb = dataset['emb'].to(self.device).to(self.device).unsqueeze(1).repeat(1,feat.shape[1],1)
        # padding_mask = dataset['padding_mask'].to(self.device)
        x = torch.cat([pid, feat, emb], axis=-1)
        # main
        for layer in self.encoder_layers:
            x = F.relu(layer(x))
        
        # PDG head
        pdg_out = self.pdg_head(x)
        # feature head
        combined = self.combined_head(torch.cat([x,pdg_out.detach()],axis=-1))
        feat_out = torch.cat(
            [self.p_head(combined),self.e_head(combined)], axis=-1
            )
        
        return pdg_out, feat_out
        # return output[...,:self.num_pdg+1], output[...,self.num_pdg+1:]
        # output: [batch, max_seq_len, num_pdg+1], [batch, max_seq_len, 4]

class DoubleEmbedder(torch.nn.Module):
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

        # self.particle_phi_v = torch.nn.Linear(in_features=tr_width, out_features=tr_width)
        # self.particle_phi_n = torch.nn.Linear(in_features=tr_width, out_features=1)
        # torch.nn.init.xavier_uniform_(self.particle_phi_n.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

        self.particle_importance = nn.Linear(in_features=tr_width, out_features=1)
        self.global_phi_v = torch.nn.Linear(in_features=tr_width, out_features=dim_hyper)
        self.global_phi_n = torch.nn.Linear(in_features=tr_width, out_features=1)
        torch.nn.init.xavier_uniform_(self.global_phi_n.weight, gain=torch.nn.init.calculate_gain('sigmoid'))
        self.to(device)
        
    def forward(self, dataset):
        pdg = self.pdg_embedder(dataset['pdg'].to(self.device))
        feat = dataset['feature'].to(self.device)
        padding_mask = dataset['padding_mask'].to(self.device)
        att_input = self.projector(torch.cat([pdg, feat], axis=-1))
        # src_key_padding_mask is inversely defined!!! True = Skip, False = Keep
        transformed = self.transformer_encoder(att_input, src_key_padding_mask=~padding_mask) 
        # features: [batch, n_particles, width]

        # particle level hyper embedding
        # particle_v = F.normalize(self.particle_phi_v(transformed)) #* self.tr_width
        # particle_p = torch.sigmoid(self.particle_phi_n(transformed))
        # particle_emb = particle_v * particle_p

        # event level hyper embedding
        particle_weights = F.softmax(
            self.particle_importance(transformed).squeeze(-1).masked_fill_(~padding_mask,-1e9),
            dim=-1)
        features = F.adaptive_avg_pool1d(
            transformed.permute(0,2,1) * particle_weights.unsqueeze(1), 1
            ).squeeze(-1) # euclidean output: [batch, width]
        global_v = F.normalize(self.global_phi_v(features))
        global_p = torch.sigmoid(self.global_phi_n(features))
        global_emb = global_v * global_p
        return transformed, global_emb 
        # particle output: [batch, tr_width], global output: [batch, dim_hyper]