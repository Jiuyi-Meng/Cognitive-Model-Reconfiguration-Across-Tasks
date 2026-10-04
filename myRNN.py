import torch
import torch.nn as nn


class TaskRNN(nn.Module):

    def __init__(
        self,
        input_size=12,      # 5 color + 5 shape + 2 task cue
        hidden_size=64,     # 64 recurrent units
        output_size=5,      # 5 possible responses
        noise_std=0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.noise_std = noise_std

        # input -> recurrent units
        self.W_in = nn.Linear(input_size, hidden_size)

        # recurrent units -> recurrent units
        self.W_rec = nn.Linear(hidden_size, hidden_size)

        # recurrent units -> output
        self.W_out = nn.Linear(hidden_size, output_size)

        # activation function
        self.relu = nn.ReLU()


    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        # initial hidden state
        h = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device
        )

        hidden_activity = []

        # run the RNN through time
        for t in range(seq_len):

            noise = torch.randn_like(h) * self.noise_std

            h = self.relu(
                self.W_in(x[:, t, :])
                + self.W_rec(h)
                + noise
            )

            hidden_activity.append(h)

        # save activity of all recurrent units at all time points
        hidden_activity = torch.stack(
            hidden_activity,
            dim=1
        )

        # output from the final hidden state
        output = self.W_out(h)

        return output, hidden_activity