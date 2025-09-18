import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    A Long Short-Term Memory (LSTM) model for time series prediction using PyTorch.
    """
    def __init__(self, input_size=1, hidden_layer_size=50, num_layers=2, output_size=1, dropout=0.2):
        """
        Initializes the LSTM model.
        
        Args:
            input_size (int): The number of input features.
            hidden_layer_size (int): The number of features in the hidden state.
            num_layers (int): The number of recurrent layers.
            output_size (int): The number of output features.
            dropout (float): The dropout probability.
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            batch_first=True, # This is important!
            dropout=dropout
        )
        
        self.linear = nn.Linear(in_features=hidden_layer_size, out_features=output_size)

    def forward(self, input_seq):
        """
        Defines the forward pass of the model.
        
        Args:
            input_seq: The input sequence for the model.
            
        Returns:
            The model's prediction.
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        
        # We only need the output of the last time step, but the LSTM layer returns all outputs
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        
        # Pass the output of the last time step through the linear layer
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

if __name__ == '__main__':
    # Example usage:
    # Create a model instance
    # Using a sequence of 60 days (timesteps) with 1 feature ('Close' price)
    model = LSTMModel()
    print("PyTorch LSTM model created successfully.")
    print(model)
    
    # Create a dummy input tensor to test the model
    # Batch size = 32, sequence length = 60, num_features = 1
    dummy_input = torch.randn(32, 60, 1)
    output = model(dummy_input)
    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Dummy output shape: {output.shape}")
