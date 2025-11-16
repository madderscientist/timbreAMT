"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from constants import *
from mel import MelSpectrogram


class BiLSTM(nn.Module):
    inference_chunk_length = 512

    def __init__(self, input_features, recurrent_features):
        super().__init__()
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x:(B, T, input_features)
        if self.training:
            return self.rnn(x)[0]   # (B, T, 2*recurrent_features)
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

            return output


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            # (B, 1, T, input_features)
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # (B, output_features//16, T, input_features)
            # layer 2
            nn.MaxPool2d((1, 2)),   # (B, output_features//16, T, input_features//2)
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # (B, output_features//8, T, input_features//2)
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            # (B, output_features//8, T, input_features//4)
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        # mel: (B, T, n_mels) n_mels = 229
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        # x: (B, T, (output_features//8)*(input_features//4))
        x = self.fc(x)
        # x: (B, T, output_features)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features=N_MELS, output_features=CONFIG.MIDI_MAX-CONFIG.MIDI_MIN+1, model_size=256):
        """
        input_features: number of mel bands in input
        output_features: number of output pitches
        model_size: number of features of bilstm output. In origin paper, lstm hidden size is 128, so model_size is 256
        """
        super().__init__()
        self.mel = MelSpectrogram(N_MELS, CONFIG.CQT.fs, WINDOW_LENGTH, CONFIG.CQT.hop, mel_fmin=MEL_FMIN, mel_fmax=MEL_FMAX)
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
        # only keep the onset and frame stacks, to be compatible with BasicAMT structure
        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )   # (B, T, output_features)
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 2, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )

    def forward(self, audio):
        if audio.dim() == 3:
            if audio.size(1) > 1:
                audio = audio.mean(dim=1, keepdim=False)
            else:
                audio = audio.squeeze(dim=1)

        # audio: (B, T) no channel dimension
        mel = self.mel(audio)
        # mel: (B, T, n_mels)
        onset_pred = self.onset_stack(mel)
        activation_pred = self.frame_stack(mel)
        # (B, T, output_features)
        combined_pred = torch.cat([onset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        # (B, T, output_features)
        onset_pred = onset_pred.transpose(1, 2).contiguous()
        frame_pred = frame_pred.transpose(1, 2).contiguous()
        # (B, output_features, T)
        return onset_pred, frame_pred

    @staticmethod
    def loss(onset, note, midiarray):
        if onset.shape[-1] > midiarray.shape[-1]:
            onset = onset[..., :midiarray.shape[-1]]
            note = note[..., :midiarray.shape[-1]]
        elif onset.shape[-1] < midiarray.shape[-1]:
            midiarray = midiarray[..., :onset.shape[-1]]
        note_ref = (midiarray >= 1).float()
        onset_ref = (midiarray == 2).float()
        frame_loss = F.binary_cross_entropy(note, note_ref)
        onset_loss = F.binary_cross_entropy(onset, onset_ref)
        return onset_loss + frame_loss