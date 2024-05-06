import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Constants
K_MER_SIZE = 4  # Consider k = 4 for k-mers
BATCH_SIZE = 10000
HIDDEN_DIM = 10
NUM_LAYERS = 2
EPOCHS = 10
LEARNING_RATE = 0.005

# Mapping of characters to integers and generating k-mers
char_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
int_to_char = {i: c for c, i in char_to_int.items()}

def create_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

# Encoding k-mers into a unique integer identifier
def get_vocab(data):
    kmers = set()
    for sequence in data:
        kmers.update(create_kmers(sequence, K_MER_SIZE))
    return {kmer: i for i, kmer in enumerate(kmers)}

# Custom Dataset for k-mers
class KmerDataset(Dataset):
    def __init__(self, sequence, vocab):
        self.x, self.y = self.prepare_dataset(sequence, vocab)
    
    def prepare_dataset(self, sequence, vocab):
        kmers = create_kmers(sequence, K_MER_SIZE)
        encoded_kmers = [vocab[kmer] for kmer in kmers if kmer in vocab]
        X = []
        Y = []
        for i in range(len(encoded_kmers) - 1):
            X.append(encoded_kmers[i])
            Y.append(encoded_kmers[i + 1])
        return np.array(X), np.array(Y)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

# Define the RNN model for k-mer level prediction
class KmerRNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers):
        super(KmerRNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(self, x):
        x = self.embed(x)  # Add batch dimension and embed
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x.unsqueeze(1), h0)  # Unsqueeze along the sequence length dimension
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

# Set up CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training and Evaluation
vocab = get_vocab([train_sequence])  # Generate vocabulary from training sequence
vocab_size = len(vocab)
train_dataset = KmerDataset(train_sequence, vocab)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataset = KmerDataset(test_sequence, vocab)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = KmerRNNModel(vocab_size, HIDDEN_DIM, NUM_LAYERS).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop with progress bar
for epoch in range(EPOCHS):
    model.train()
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Evaluate the model
accuracy = evaluate(model, test_loader)
print(f'Accuracy on test set: {accuracy}%')
