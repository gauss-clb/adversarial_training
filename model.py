import torch
import torch.nn as nn

class TextCNNConfig:
    def __init__(self, vocab_size=10, num_class=2, max_len=128):
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.out_channels = 128
        self.kernel_sizes = [2, 3, 4]
        self.dropout_rate = 0.5
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_class = num_class

class TextCNN(nn.Module):

    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(1, config.out_channels, (kernel_size, config.embedding_dim)),
            nn.ReLU(),
            nn.MaxPool2d((config.max_len - kernel_size + 1, 1))
        ) for kernel_size in config.kernel_sizes])
        self.dropout = nn.Dropout(config.dropout_rate)
        # self.fc1 = nn.Linear(3 * config.out_channels, config.hidden_dim)
        # self.fc2 = nn.Linear(config.hidden_dim, config.num_class)
        self.fc = nn.Linear(3 * config.out_channels, config.num_class)

    def forward(self, input_ids, attack=None):
        embed_x = self.embedding(input_ids)
        if attack is not None:
            embed_x = embed_x + attack
        embed_x = embed_x.unsqueeze(1) # [bs, 1, seq_len, emb_dim]
        conv_x = [conv(embed_x) for conv in self.convs]
        out = torch.cat(conv_x, dim=1).squeeze() # [bs, 3 * out_channels]
        out = self.dropout(out)
        # out = self.fc1(out)
        # out = self.fc2(out)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    config = TextCNNConfig()
    model = TextCNN(config)
    input_ids = torch.randint(10, (7, config.max_len))
    model(input_ids)