from typing import List
import torch as th
import torch.nn as nn

from causal_conv1d import CausalConv1d
from transformers import TFEncoder, TFDecoder, PositionalEncoding
from fsq import FSQ


class SkillTransformer(nn.Module):

    def __init__(
            self,
            action_dim: int,
            encoder_dim: int = 128,
            decoder_dim: int = 128,
            sequence_len: int = 16,
            encoder_heads:int = 4,
            encoder_layers: int = 2,
            decoder_heads: int = 4,
            decoder_layers: int = 4,
            attn_dropout: float = 0.1,
            fsq_level: List[int] = [8, 6, 5],
            kernel_sizes: List[int] = [5, 3, 3],
            strides: List[int] = [2, 2, 1],
    ):
        super().__init__()

        self.decoder_dim = decoder_dim
        self.sequence_len = sequence_len

        self.causal_conv1d = CausalConv1d(
            action_dim,
            encoder_dim, 
            kernel_sizes,
            strides,
        )
        self.tf_encoder = TFEncoder(
            encoder_dim,
            encoder_heads,
            encoder_layers,
            dropout=attn_dropout,
        )
        self.encoder_mlp = nn.Linear(encoder_dim, len(fsq_level))
        self.fsq = FSQ(fsq_level)

        self.pe = PositionalEncoding(decoder_dim, max_len=sequence_len)
        self.tf_decoder = TFDecoder(
            decoder_dim,
            decoder_heads,
            decoder_layers,
            dropout=attn_dropout,
        )
        self.decoder_mlp = nn.Linear(len(fsq_level), decoder_dim)
        self.pred_mlp = nn.Linear(decoder_dim, action_dim)

    def forward(self, action_sequence: th.Tensor):
        # action_sequence: (batch size, sequence length, action dimension)
        # causal conv1d requires shape (batch size, action dimension, sequence length)
        x = th.permute(action_sequence, (0, 2, 1))
        x = self.causal_conv1d(x)

        # seq: (batch size, encoder dimension, reduced sequence length)
        x = th.permute(x, (0, 2, 1))
        x = self.tf_encoder(x)
        x = self.encoder_mlp(x)

        # quantize using FSQ
        x_hat, idx = self.fsq(x)  # x_hat has same size as x
        # return x_hat[:, -1], idx[:, -1]
        return x_hat, idx
    
    def encode(self, action_sequence: th.Tensor):
        x_hat, _ = self(action_sequence)
        return x_hat
    
    def decode(self, x: th.Tensor, use_index: bool = False):
        if use_index:
            # indices = x.unsqueeze(0)
            indices = x
            codes = self.fsq.indices_to_codes(indices)
        else:
            # codes = x.unsqueeze(1)
            codes = x
        codes = self.decoder_mlp(codes)
        batch_size = codes.size(0)
        pe = self.pe(th.zeros((batch_size, self.sequence_len, self.decoder_dim), device=x.device))
        return self.pred_mlp(self.tf_decoder(pe, codes))
    
    def compute_loss(self, act_seq):
        skill = self.encode(act_seq)
        act_seq_pred = self.decode(skill)
        # print(act_seq_pred[0, 0], act_seq[0, 0])
        loss = nn.functional.mse_loss(act_seq_pred, act_seq)
        return loss


if __name__ == '__main__':

    action_sequence = th.randn((2, 32, 4))
    stf = SkillTransformer(4)
    z, i = stf(action_sequence)
    print(z, i)

    a_hat = stf.decode(i, use_index=True)
    print(action_sequence, a_hat)

    n = 0
    for p in stf.parameters():
        n += p.numel()
    print(n)
