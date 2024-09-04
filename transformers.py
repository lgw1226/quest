import torch as th
import torch.nn as nn
import numpy as np


class TFEncoder(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_layers: int,
            dropout: float = 0.5,
    ):
        super().__init__()
        tf_encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True,
        )
        self.tf_encoder = nn.TransformerEncoder(tf_encoder_layer, num_layers)

    def forward(self, x: th.Tensor):
        len = x.size(1)
        mask = th.triu(th.ones((len, len))).to(th.bool)
        return self.tf_encoder(x, mask=mask, is_causal=True)


class TFDecoder(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            num_layers: int,
            dropout: float = 0.5,
    ):
        super().__init__()
        tf_decoder_layer = nn.TransformerDecoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True,
        )
        self.tf_decoder = nn.TransformerDecoder(tf_decoder_layer, num_layers)

    def forward(self, target, memory):
        # memory is a positional encoding
        len = target.size(1)
        mask = th.triu(th.ones((len, len))).to(th.bool)
        return self.tf_decoder(target, memory, tgt_mask=mask, tgt_is_causal=True)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix to hold the positional encodings
        pe = th.zeros(max_len, d_model)
        
        # Calculate the positional encodings for each position
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        # Apply sine to even indices (2i) and cosine to odd indices (2i+1)
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        
        # Add a dimension for batch size (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register buffer allows the model to save the positional encodings without considering them as a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input tensor x
        x = x + self.pe[:, :x.size(1), :]
        return x
    

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Example parameters
    d_model = 512  # Dimension of the model (embedding size)
    seq_len = 50   # Length of the input sequence (number of tokens)
    batch_size = 32  # Batch size

    # Initialize the positional encoding module
    pos_encoding = PositionalEncoding(d_model=d_model, max_len=seq_len)

    # Create random embeddings of size (batch_size, seq_len, d_model)
    input_embeddings = th.randn(batch_size, seq_len, d_model)

    # Add positional encodings to the input embeddings
    encoded_input = pos_encoding(input_embeddings)

    print("Shape of input embeddings with positional encodings:", encoded_input.shape)

    # Plot the positional encodings for the first token position
    plt.figure(figsize=(10, 8))
    pe_values = pos_encoding.pe[0, :, :].numpy()  # Get the positional encodings from the buffer

    # Select up to 20 dimensions for visualization
    plt.plot(pe_values[:, :1])
    plt.title('Positional Encodings')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.show()
