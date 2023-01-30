import torch
import torchaudio


class AudioLoader:
    audio_duration = 1
    audio_channels = 1
    audio_sampling = 8000

    def __init__(self):
        pass

    def audio_preprocessing(self, audioIn):
        return self.dcRemoval(self.resample(self.rechannel(audioIn)))

    def dcRemoval(self, aud):
        wav, sr = aud
        wav = torchaudio.functional.dcshift(wav, -wav.mean())
        return wav, sr

    def rechannel(self, aud):
        sig, sr = aud
        if sig.shape[0] == self.audio_channels:
            # Nothing to do
            return aud
        elif self.audio_channels == 1:
            # Convert from stereo to mono by selecting only the first channel
            resignal = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resignal = torch.cat([sig, sig])
        return resignal, sr

    def resample(self, aud):
        sig, sr = aud
        if sr == self.audio_sampling:
            return aud
        resig = torchaudio.transforms.Resample(sr, self.audio_sampling)(sig)
        return resig, self.audio_sampling
