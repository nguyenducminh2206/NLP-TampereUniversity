import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import datasets
from tqdm import tqdm
import spacy
from pprint import pprint
from transformers import AutoTokenizer
from collections import Counter
import json

dataset = datasets.load_dataset('ncduy/mt-en-vi')

train_data, valid_data, test_data = (
    dataset["train"].remove_columns('source'),
    dataset["validation"].remove_columns('source'),
    dataset["test"].remove_columns('source')
)

# Define special tokens and parameters
max_length = 1000
lower = True
sos_token = "<sos>"
eos_token = "<eos>"
unk_token = "<unk>"
pad_token = "<pad>"
min_freq = 2
special_tokens = [unk_token, pad_token, sos_token, eos_token]

# Define tokenizer
en_nlp = spacy.load('en_core_web_sm')
vi_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Step 1: Tokenize and Numericalize sentences 
def process_sentence(sentence):
    # Tokenize English
    en_tokens = [token.text for token in en_nlp.tokenizer(sentence["en"])][:max_length]
    vi_tokens = vi_tokenizer.tokenize(sentence["vi"])[:max_length]
    
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        vi_tokens = [token.lower() for token in vi_tokens]
    
    # Add special tokens
    en_tokens = [sos_token] + en_tokens + [eos_token]
    vi_tokens = [sos_token] + vi_tokens + [eos_token]

    # Numericalize tokens
    en_ids = [en_vocab.get(token, en_vocab[unk_token]) for token in en_tokens]
    vi_ids = [vi_vocab.get(token, vi_vocab[unk_token]) for token in vi_tokens]
    
    return {
        "en": sentence["en"],
        "vi": sentence["vi"],
        "en_tokens": en_tokens,
        "vi_tokens": vi_tokens,
        "en_ids": en_ids,
        "vi_ids": vi_ids,
    }

# Step 2: Build Vocabulary
def build_vocab(data, min_freq, specials):
    counter = Counter()
    for tokens in data:
        counter.update(tokens)
    vocab = {token: idx for idx, token in enumerate(specials)}
    sorted_tokens = sorted(token for token, freq in counter.items() if freq >= min_freq)
    for token in sorted_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab

# Step 3: Generate Tokenized Data for Vocabulary Building
tokenized_train_data = train_data.map(
    lambda example: {"en_tokens": [token.text for token in en_nlp.tokenizer(example["en"])][:max_length],
                     "vi_tokens": vi_tokenizer.tokenize(example["vi"])[:max_length]}
)

# Build vocabularies
en_vocab = build_vocab(tokenized_train_data["en_tokens"], min_freq, special_tokens)
vi_vocab = build_vocab(tokenized_train_data["vi_tokens"], min_freq, special_tokens)

# Step 4: Process Full Dataset
train_data = train_data.map(process_sentence)
valid_data = valid_data.map(process_sentence)
test_data = test_data.map(process_sentence)

# Save vocabularies to JSON
with open("en_vocab.json", "w") as f:
    json.dump(en_vocab, f)
with open("vi_vocab.json", "w") as f:
    json.dump(vi_vocab, f)

pprint(train_data[random.randint(0, 2884451)])

# Check for special tokens in both vocabularies
assert en_vocab[unk_token] == vi_vocab[unk_token]
assert en_vocab[pad_token] == vi_vocab[pad_token]

unk_index = en_vocab[unk_token]
pad_index = en_vocab[pad_token]

data_type = "torch"
format_columns = ["en_ids", "vi_ids"]

train_data = train_data.with_format(
    type=data_type, columns=format_columns, output_all_columns=True
)

valid_data = valid_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

test_data = test_data.with_format(
    type=data_type,
    columns=format_columns,
    output_all_columns=True,
)

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_vi_ids = [example["vi_ids"] for example in batch]
        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_vi_ids = nn.utils.rnn.pad_sequence(batch_vi_ids, padding_value=pad_index)
        batch = {
            "en_ids": batch_en_ids,
            "vi_ids": batch_vi_ids,
        }
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

batch_size = 128

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs

input_dim = len(vi_vocab)
output_dim = len(en_vocab)
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
)

model = Seq2Seq(encoder, decoder, device).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["vi_ids"]
        trg = batch["en_ids"]
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = batch["vi_ids"]
            trg = batch["en_ids"]
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

n_epochs = 10
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")

for epoch in tqdm(range(n_epochs)):
    train_loss = train_fn(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        
    )
    valid_loss = evaluate_fn(
        model,
        valid_data_loader,
        criterion,
        
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt")
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")