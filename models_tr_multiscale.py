import torch,os
from torch import nn
import math
from torch.nn.init import xavier_uniform_

from torch import Tensor
from typing import Optional

from torch.nn import functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossTransformer(nn.Module):
    """
    Cross Transformer layer
    """

    def __init__(self, dropout, d_model=512, n_head=4):
        """
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        """
        super(CrossTransformer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, input_s1, input_s2, input_l1, input_l2, w_s):
        # dif_as_kv
        dif_s = input_s1 - input_s2
        output_s1 = self.cross(input_s1, dif_s)  # (Q,K,V)
        output_s2 = self.cross(input_s2, dif_s)  # (Q,K,V)

        cat_s = input_s1 + input_s2
        output_s11 = self.cross(output_s1, cat_s)  # (Q,K,V)
        output_s22 = self.cross(output_s2, cat_s)

        dif_l = input_l1 - input_l2
        output_l1 = self.cross(input_l1, dif_l)  # (Q,K,V)
        output_l2 = self.cross(input_l2, dif_l)

        cat_l = input_l1 + input_l2
        output_l11 = self.cross(output_l1, cat_l)  # (Q,K,V)
        output_l22 = self.cross(output_l2, cat_l)

        output_rl1 = F.interpolate(output_l11.view(28,28,-1,512).permute(2, 3, 0, 1), size=(w_s, w_s), mode='bilinear', align_corners=True).view(-1, 512, w_s**2).permute(2, 0, 1)
        output_rl2 = F.interpolate(output_l22.view(28,28,-1,512).permute(2, 3, 0, 1), size=(w_s, w_s), mode='bilinear', align_corners=True).view(-1, 512, w_s**2).permute(2, 0, 1)

        return output_s11 + output_rl1, output_s22 + output_rl2#, output_l1, output_l2
    def cross(self, input,dif):
        # RSICCformer_D (diff_as_kv)
        attn_output, attn_weight = self.attention(input, dif, dif)  # (Q,K,V)

        output = input + self.dropout1(attn_output)

        output = self.norm1(output)
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(output))))
        output = output + self.dropout3(ff_output)
        output = self.norm2(output)
        return output

class resblock(nn.Module):
    '''
    module: Residual Block
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
                nn.Conv2d(inchannel,int(outchannel/2),kernel_size = 1),
                # nn.LayerNorm(int(outchannel/2),dim=1),
                nn.BatchNorm2d(int(outchannel/2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(outchannel/2), int(outchannel / 2), kernel_size = 3,stride=1,padding=1),
                # nn.LayerNorm(int(outchannel/2),dim=1),
                nn.BatchNorm2d(int(outchannel / 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(int(outchannel/2),outchannel,kernel_size = 1),
                # nn.LayerNorm(int(outchannel / 1),dim=1)
                nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out += residual
        return F.relu(out)


class MCCFormers_diff_as_Q(nn.Module):
    """
    RSICCFormers_diff
    """

    def __init__(self, feature_dim, dropout, h, w, d_model=512, n_head=4, n_layers=3):
        """
        :param feature_dim: dimension of input features
        :param dropout: dropout rate
        :param d_model: dimension of hidden state
        :param n_head: number of heads in multi head attention
        :param n_layer: number of layers of transformer layer
        """
        super(MCCFormers_diff_as_Q, self).__init__()
        self.d_model = d_model

        # n_layers = 3
        print("encoder_n_layers=", n_layers)

        self.n_layers = n_layers

        self.sw_embedding = nn.Embedding(w, int(d_model / 2))
        self.sh_embedding = nn.Embedding(h, int(d_model / 2))

        self.lw_embedding = nn.Embedding(w*2, int(d_model / 2))
        self.lh_embedding = nn.Embedding(h*2, int(d_model / 2))

        self.embedding_1D = nn.Embedding(h*w, int(d_model))

        self.projection = nn.Conv2d(feature_dim, d_model, kernel_size=1)
        self.projection2 = nn.Conv2d(768, d_model, kernel_size=1)
        self.projection3 = nn.Conv2d(512, d_model, kernel_size=1)
        self.projection4 = nn.Conv2d(256, d_model, kernel_size=1)

        self.transformer = nn.ModuleList([CrossTransformer(dropout, d_model, n_head) for i in range(n_layers)])

        self.resblock = nn.ModuleList([resblock(d_model*2, d_model*2) for i in range(n_layers)])

        self.LN = nn.ModuleList([nn.LayerNorm(d_model*2) for i in range(n_layers)])

        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, img_feat1, img_feat2):
        # img_feat1 (batch_size, feature_dim, h, w)
        batch = img_feat1[0].size(0)
        feature_dim_samll = img_feat1[1].size(1)

        feature_dim_large = img_feat1[0].size(1)

        w_s, h_s = img_feat1[1].size(2), img_feat1[1].size(3)

        w_l, h_l = img_feat1[0].size(2), img_feat1[0].size(3)

        if feature_dim_samll == 1024:
            img_feat_s1 = self.projection(img_feat1[1])
            img_feat_s2 = self.projection(img_feat2[1])

        if feature_dim_large == 512:
            img_feat_l1 = self.projection3(img_feat1[0])
            img_feat_l2 = self.projection3(img_feat2[0])
            '''
        if feature_dim == 768:
            img_feat1 = self.projection2(img_feat1)  
            img_feat2 = self.projection2(img_feat2)  
         
        if feature_dim == 256:
            img_feat1 = self.projection4(img_feat1)  
            img_feat2 = self.projection4(img_feat2)  
            '''
        pos_sw = torch.arange(w_s, device=device).to(device)
        pos_sh = torch.arange(h_s, device=device).to(device)
        embed_sw = self.sw_embedding(pos_sw)
        embed_sh = self.sh_embedding(pos_sh)
        position_embedding_s = torch.cat([embed_sw.unsqueeze(0).repeat(h_s, 1, 1),
                                        embed_sh.unsqueeze(1).repeat(1, w_s, 1)],
                                       dim=-1)
        # (h, w, d_model)
        position_embedding_s = position_embedding_s.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,1)  # (batch, d_model, h, w)
        # 1D_PE
        # position_embedding = self.embedding_1D(torch.arange(h*w, device=device).to(device)).view(h,w,self.d_model).permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,1)

        img_feat_s1 = img_feat_s1 + position_embedding_s  # (batch_size, d_model, h, w)
        img_feat_s2 = img_feat_s2 + position_embedding_s  # (batch_size, d_model, h, w)

        encoder_output_s1 = img_feat_s1.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)
        encoder_output_s2 = img_feat_s2.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)
#--------------------------------------------large------------------------------------------------
        pos_lw = torch.arange(w_l, device=device).to(device)
        pos_lh = torch.arange(h_l, device=device).to(device)
        embed_lw = self.lw_embedding(pos_lw)
        embed_lh = self.lh_embedding(pos_lh)
        position_embedding_l = torch.cat([embed_lw.unsqueeze(0).repeat(h_l, 1, 1),
                                          embed_lh.unsqueeze(1).repeat(1, w_l, 1)],
                                         dim=-1)
        # (h, w, d_model)
        position_embedding_l = position_embedding_l.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,
                                                                                         1)  # (batch, d_model, h, w)
        # 1D_PE
        # position_embedding = self.embedding_1D(torch.arange(h*w, device=device).to(device)).view(h,w,self.d_model).permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1,1)

        img_feat_l1 = img_feat_l1 + position_embedding_l  # (batch_size, d_model, h, w)
        img_feat_l2 = img_feat_l2 + position_embedding_l  # (batch_size, d_model, h, w)

        encoder_output_l1 = img_feat_l1.view(batch, self.d_model, -1).permute(2, 0, 1)  # (h*w, batch_size, d_model)
        encoder_output_l2 = img_feat_l2.view(batch, self.d_model, -1).permute(2, 0, 1)

        output_s1 = encoder_output_s1
        output_s2 = encoder_output_s2

        output_l1 = encoder_output_l1
        output_l2 = encoder_output_l2
        output_s1_list = list()
        output_s2_list = list()
        for l in self.transformer:
            output1, output2 = l(output_s1, output_s2, output_l1, output_l2, w_s )

            output_s1_list.append(output1)
            output_s2_list.append(output2)

        # MBF
        i = 0
        output = torch.zeros((196,batch,self.d_model*2)).to(device)
        for res in self.resblock:
            input = torch.cat([output_s1_list[i],output_s2_list[i]],dim=-1)
            output = output + input
            output = output.permute(1, 2, 0).view(batch, self.d_model*2, 14, 14)
            output = res(output)
            output = output.view(batch, self.d_model*2,-1).permute(2, 0, 1)
            output = self.LN[i](output)
            i=i+1

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.embedding_1D = nn.Embedding(52, int(d_model))
    def forward(self, x):
        # fixed
        x = x + self.pe[:x.size(0), :]
        # learnable
        # x = x + self.embedding_1D(torch.arange(52, device=device).to(device)).unsqueeze(1).repeat(1,x.size(1),  1)
        return self.dropout(x)

class Mesh_TransformerDecoderLayer(nn.Module):

    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, int(nhead), dropout=dropout)
        self.multihead_attn3 = nn.MultiheadAttention(int(d_model), int(nhead), dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)


        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.activation2 = nn.Softmax(dim=-1)

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)


    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        # # cross self-attention
        enc_att, att_weight = self._mha_block2((self_att_tgt),
                                               memory, memory_mask,
                                               memory_key_padding_mask)
     
        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x,att_weight = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout2(x),att_weight
    def _mha_block2(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x ,att_weight= self.multihead_attn2(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout3(x),att_weight
    def _mha_block3(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x ,att_weight= self.multihead_attn3(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=True)
        return self.dropout4(x),att_weight

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout4(x)



class DecoderTransformer(nn.Module):
    """
    Decoder with Transformer.
    """

    def __init__(self, feature_dim, vocab_size, n_head, n_layers, dropout):
        """
        :param n_head: the number of heads in Transformer
        :param n_layers: the number of layers of Transformer
        """
        super(DecoderTransformer, self).__init__()

        # n_layers = 1
        print("decoder_n_layers=",n_layers)

        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # embedding layer
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)  # vocaburaly embedding

        # Transformer layer
        decoder_layer = Mesh_TransformerDecoderLayer(feature_dim, n_head, dim_feedforward=feature_dim * 4,
                                                   dropout=self.dropout)
        self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim)

        # Linear layer to find scores over vocabulary
        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence
        """
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)

        self.wdc.bias.data.fill_(0)
        self.wdc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, memory, encoded_captions, caption_lengths):
        """
        :param memory: image feature (S, batch, feature_dim)
        :param tgt: target sequence (length, batch)
        :param sentence_index: sentence index of each token in target sequence (length, batch)
        """
        tgt = encoded_captions.permute(1, 0)
        tgt_length = tgt.size(0)

        mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)

        tgt_embedding = self.vocab_embedding(tgt)
        tgt_embedding = self.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

        pred = self.transformer(tgt_embedding, memory, tgt_mask=mask)  # (length, batch, feature_dim)
        pred = self.wdc(self.dropout(pred))  # (length, batch, vocab_size)

        pred = pred.permute(1, 0, 2)

        # Sort input data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()

        return pred, encoded_captions, decode_lengths, sort_ind


