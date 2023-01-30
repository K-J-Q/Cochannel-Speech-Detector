import torch, torchaudio

class AudioLoader():
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
        return wav,sr

    def pad_trunc(self, aud, reduce_only=False):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        target_len = int(sr / 1000) * (self.audio_duration)

        # ((self.audio_duration-1000) if reduce_only else self.audio_duration)

        if (sig_len > target_len):
            start_len = random.randint(0, sig_len - target_len)
            sig = sig[:, start_len:start_len + target_len]
            assert (sig.shape[1] == target_len)

        elif sig_len < target_len and not reduce_only:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, target_len - sig_len)
            pad_end_len = target_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))
            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return sig, sr

    def rechannel(self, aud, showWarning=True):
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
        if showWarning:
            print('rechannel process triggered!')
        return resignal, sr

    def resample(self, aud, showWarning=True):
        sig, sr = aud
        if (sr == self.audio_sampling):
            # Nothing to do
            return aud
        if showWarning:
            print('resampling process triggered!')
        num_channels = sig.shape[0]

        resig = torchaudio.transforms.Resample(sr, self.audio_sampling)(sig)
        return ((resig, self.audio_sampling))