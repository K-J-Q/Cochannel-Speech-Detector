import torchaudio
# from torchaudio.io import StreamReader
from pathlib import Path
import torch
import torch.nn as nn
import ml.machineLearning
from loader.AudioDataset import Augmentor, createDataset, collate_batch
import random
import seaborn as sn
import matplotlib.pyplot as plt
import loader.utils as utils
from torchsummary import summary

# To ensure reproducibility
random.seed(0)
torch.manual_seed(0)

class_map = ['0', '1', '2']


def predictFolder(model, device, folderPath):
    audio_paths = [[], []]
    audio_paths[0] = utils.getAudioPaths(folderPath[0])[0:10]
    audio_paths[1] = utils.getAudioPaths(
        folderPath[1])[0:100]

    audio_test_dataset = createDataset(audio_paths, generateCochannel=False)

    test_dataloader = torch.utils.data.DataLoader(
        audio_test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        collate_fn=collate_batch
    )

    test_loss, test_acc, confusion_matrix = ml.machineLearning.eval(
        model, test_dataloader, torch.nn.CrossEntropyLoss(), device)
    print(f'Validating  | Loss: {test_loss} Accuracy: {test_acc}% \n')
    sn.heatmap(confusion_matrix.cpu(), annot=True,
               xticklabels=class_map, yticklabels=class_map)
    plt.show()


def predictFile(filePath, model, device,labelPath = None):
    augmentor = Augmentor()
    spectrogram = torchaudio.transforms.MelSpectrogram(normalized=True, sample_rate=8000, n_fft=256, n_mels=64)
    sm = torch.nn.Softmax()

    if labelPath:
        gt_x, gt_y = getGroundTruth(labelPath)
        plt.step(gt_x, gt_y)

    pred_graph = [0]

    with torch.no_grad():
        wav, sr = augmentor.resample(augmentor.rechannel(torchaudio.load(filePath)))
        sampleLength = 4 * sr
        wav = wav[0]
        for splitIndex, j in enumerate(range(0, len(wav), sampleLength)):
            trimmed_audio = wav[j:j+sampleLength]
            # print(splitIndex, ' ', len(trimmed_audio))
            if len(trimmed_audio) == sampleLength:
                spectrogram_tensor = (spectrogram(
                    trimmed_audio) + 1e-12).log2()
                spectrogram_tensor = torch.unsqueeze(torch.unsqueeze(
                    spectrogram_tensor, 0), 0).to(device)
                pred = model(spectrogram_tensor)
                pred = sm(pred)

                pred_graph.append(int(class_map[pred.argmax()]))
    plt.step(torch.arange(len(pred_graph))*4,pred_graph)
    plt.ylim(0,2)
    plt.ylabel('Number of speakers')
    plt.xlabel('Time (seconds)')
    plt.yticks(torch.arange(0,2))
    plt.xticks(torch.arange(0, len(pred_graph)*4, 30))
    plt.show()


def predictLive(model, device):
    augmentor = Augmentor()

    streamer = StreamReader(
        src="audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{7BA2F90C-C592-4A85-8B11-73B716179C4A}",
        format="dshow",
    )

    streamer.add_basic_audio_stream(
        frames_per_chunk=44100*4, sample_rate=44100)

    stream_iterator = streamer.stream()
    wav = torch.Tensor([])
    sm = torch.nn.Softmax()

    with torch.no_grad():
        while True:
            (chunk,) = next(stream_iterator)
            # wav = torch.cat((wav, chunk[:, 0]))
            spectrogram = torchaudio.transforms.MelSpectrogram(normalized=True, sample_rate=8000, n_fft=256, n_mels=64)
            wav, sr = augmentor.audio_preprocessing([chunk.T, 44100])

            spectrogram_tensor = (spectrogram(wav) + 1e-12).log2()
            spectrogram_tensor = torch.unsqueeze(spectrogram_tensor, 0).to(device)
            pred = model(spectrogram_tensor)
            pred = sm(pred)
            print(pred)
            # sn.barplot(y=class_map, x=pred[0].cpu().numpy())
            # plt.show()

def getGroundTruth(file):
    label_list = [0]
    time_list = [0]

    with open(file) as f:
        for lbl in f.read().splitlines():
            lbl = lbl.split('\t')

            startTime = int(float(lbl[0]))
            endTime = int(float(lbl[1]))
            num_speakers = int(lbl[2])

            if startTime != time_list[-1]:
                time_list.append(startTime)
                label_list.append(1)
            time_list.append(endTime)
            label_list.append(int(num_speakers))
    return time_list, label_list

if __name__ == "__main__":
    model, device, _ = ml.machineLearning.selectModel()
    summary(model.cuda(), (1, 201, 161))
    a = '/media/jianquan/Data/Original Audio/Singapore Speech Corpus/[P] Part 3 Same BoundaryMic/3003.wav'
    predictFile(a, model, device)

    # predictFolder(model, device, [
    #               'E:\Processed Audio\SPEECH', 'E:\Processed Audio\ENV'])

    # predictLive(model, device)

