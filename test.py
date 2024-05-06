import torch
import torch.nn as nn
import torch.optim as optim

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Parameters
input_size = 4  # A, C, G, T
hidden_size = 128  # Number of features in the hidden state
output_size = 4  # Output size is the same as input (predicting one of A, C, G, T)

model = RNNModel(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop
def train(model, data_loader):
    model.train()
    for epoch in range(10):  # Number of epochs
        for batch in data_loader:
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class KmerDataset(Dataset):
    def __init__(self, sequence, k=4):
        self.kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
        self.vocab = {v: k for k, v in enumerate(set(self.kmers))}
        self.encoded = [self.vocab[kmer] for kmer in self.kmers]

    def __len__(self):
        return len(self.encoded) - 1

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx:idx+4]), torch.tensor(self.encoded[idx+1:idx+5])

# Define the RNN model
class RNNKmerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super(RNNKmerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Assuming train_data contains the genomic sequence
train_dataset = KmerDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

vocab_size = len(train_dataset.vocab)
embed_dim = 10
hidden_size = 128
output_size = vocab_size

model = RNNKmerModel(vocab_size, embed_dim, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, data_loader):
    model.train()
    for epoch in range(10):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Example training call
train(model, train_loader)

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.attn(encoder_outputs)
        return F.softmax(attn_energies, dim=1)

class RNNAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super(RNNAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, hidden = self.rnn(x)
        attn_weights = self.attention(hidden, rnn_out)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), rnn_out)
        output = self.fc(attn_applied.squeeze(0))
        return output