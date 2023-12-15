import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, fc_units, output_size, target_length, device):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_length = target_length
        self.device = device

        # Encoder
        self.encoder_gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Decoder
        self.decoder_gru = nn.GRU(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_size, fc_units)
        self.decoder_out = nn.Linear(fc_units, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Encoder
        encoder_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
        _, encoder_hidden = self.encoder_gru(x, encoder_hidden)

        # Decoder
        decoder_input = x[:, -1, :].unsqueeze(1)  # Last element of input sequence
        decoder_hidden = encoder_hidden

        outputs = torch.zeros(batch_size, self.target_length, x.size(2), device=self.device)
        for ti in range(self.target_length):
            decoder_output, decoder_hidden = self.decoder_gru(decoder_input, decoder_hidden)
            decoder_output = F.relu(self.decoder_fc(decoder_output))
            decoder_output = self.decoder_out(decoder_output)
            outputs[:, ti:ti+1, :] = decoder_output
            decoder_input = decoder_output  # Use own predictions as inputs for next step

        return outputs

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
