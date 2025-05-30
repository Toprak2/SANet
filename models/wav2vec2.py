import torch
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


class Wav2Vec2SpeakerClassifier(nn.Module):
    def __init__(self, num_speakers=10):
        super().__init__()
        # Load the base Wav2Vec2 model without the LM head
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Get the hidden size from the Wav2Vec2 config
        hidden_size = self.wav2vec2.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512, momentum=0.05, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128, momentum=0.05, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_speakers)
        )

    def forward(self, input_values, attention_mask=None):

        outputs = self.wav2vec2(input_values=input_values,
                                attention_mask=attention_mask)

        # Get the last hidden states (batch, seq_len, hidden_size)
        hidden_states = outputs.last_hidden_state

        pooled_output = torch.mean(hidden_states, dim=1)

        logits = self.classifier(pooled_output)

        return logits
