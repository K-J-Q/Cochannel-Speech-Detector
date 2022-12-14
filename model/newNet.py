import wavencoder
import torch.nn as nn
import torch

class AudioClassifier(nn.Module):
    def __init__(self, hidden_size=512, n_classes=3):
        super(AudioClassifier, self).__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True) # Wav2Vec model
        self.transformer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=1) # Transformer layer
        self.attention = wavencoder.layers.SoftAttention(hidden_size, hidden_size) # SoftAttention Layer - (embedding dim, attention_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, n_classes),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.encoder(x) # [batch, 512, Num_Frames]
        x = x.transpose(1,2) # [batch, Num_Frames, 512]
        x =  self.transformer(x) # [batch, Num_Frames, 512]
        attn_output = self.attention(x)  # [batch, 512]
        y_hat = self.classifier(attn_output) # [batch, n_classes]
        return y_hat, attn_output

