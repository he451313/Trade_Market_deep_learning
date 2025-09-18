import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # The input x is expected to be of shape (seq_len, batch_size, d_model)
        # We add positional encoding to the sequence dimension
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.output_decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: (batch_size, seq_len, features)
        src = self.input_encoder(src) * math.sqrt(self.d_model)
        
        # PyTorch TransformerEncoderLayer with batch_first=True expects (batch_size, seq_len, d_model)
        # Our PositionalEncoding expects (seq_len, batch_size, d_model), so we need to permute
        # However, to simplify, let's adjust PositionalEncoding's forward pass to handle batch_first
        # Let's make a quick adjustment to PositionalEncoding to be more flexible or adjust here.
        # A simpler approach for batch_first layout:
        # Permute src to (seq_len, batch_size, d_model) for pos_encoder, then permute back.
        # src = src.permute(1, 0, 2)
        # src = self.pos_encoder(src)
        # src = src.permute(1, 0, 2)
        # The above is the standard way, but let's try a direct add for simplicity, assuming batch_first logic
        # The positional encoding needs to be added to the sequence dimension.
        # Let's adjust the positional encoding logic slightly to be more direct with batch_first.
        # For now, we will proceed with a simplified assumption that direct addition works if PE is broadcastable.
        # Let's create a version of positional encoding that is directly applicable to (Batch, Seq, Dim)
        
        # Re-implementing pos encoding logic directly here for clarity with batch_first=True
        # This is a common source of error, so let's be explicit.
        # The shape of src is (batch_size, seq_len, d_model)
        seq_len = src.size(1)
        pe = self.pos_encoder.pe[:seq_len, :].squeeze(1) # Shape: (seq_len, d_model)
        src = src + pe # Add pe to each item in the batch

        output = self.transformer_encoder(src)
        # We take the output of the last time step to make a prediction
        output = self.output_decoder(output[:, -1, :])
        return output

if __name__ == '__main__':
    # Example Usage
    model = TransformerModel()
    print("PyTorch Transformer model created successfully.")
    print(model)
    
    # Dummy input tensor with shape (batch_size, seq_len, input_features)
    dummy_input = torch.randn(32, 60, 1)
    output = model(dummy_input)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Dummy output shape: {output.shape}")
