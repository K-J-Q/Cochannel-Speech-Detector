import torchaudio
# from torchaudio.io import StreamReader
from pathlib import Path
import torch
import ml.machineLearning
from loader.AudioDataset import *
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
    audio_paths, _ = utils.getAudioPaths(folderPath, percent=0.01)

    audio_test_dataset = createDataset(audio_paths)
    
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
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()


def predictFile(filePath, model, device):
    augmentor = Augmentor()
    sm = torch.nn.Softmax()

    def IoU(predicted, ground_truth):
        print(ground_truth)
        print(predicted)


    labelPath = filePath[0:-3]+'txt'

    if os.path.exists(labelPath):
        gt_x, gt_y = getGroundTruth(labelPath)
        plt.step(gt_x, gt_y)
        plt.legend(['Ground Truth', 'Predicted'])

    pred = []
    pred_graph = [0]

    batch_size = 20
    windowLength = 1
    

    with torch.no_grad():
        wav, sr = augmentor.resample(augmentor.rechannel(torchaudio.load(filePath)))
        sampleLength = windowLength * sr
        wav = wav[0]
        batch_length = batch_size*sampleLength
        for batch in range(0, len(wav)-sampleLength, batch_length):
            spectrogram_tensor = torch.zeros([batch_size, 1,  257, 63])
            for splitIndex, j in enumerate(range(batch, batch + batch_length, sampleLength)):
                trimmed_audio = wav[j:j+sampleLength]
                # print(splitIndex, ' ', len(trimmed_audio))
                if len(trimmed_audio) == sampleLength:
                    spectrogram_tensor[splitIndex][0] = generateSpec(trimmed_audio)
                    plt.imshow(spectrogram_tensor[splitIndex][0])
                    plt.show()

            # print(spectrogram_tensor)
            pred = model(spectrogram_tensor.to(device))
            pred = sm(pred)
            pred = pred.argmax(dim=1)
            pred_graph += list(pred.cpu().numpy())
            print(pred)

    plt.step(torch.arange(len(pred_graph))*windowLength,pred_graph)
    plt.ylim(0,2)
    plt.ylabel('Number of speakers')
    plt.xlabel('Time (seconds)')
    plt.yticks(torch.arange(0,2))
    plt.xticks(torch.arange(0, len(pred_graph)*windowLength, 30))
    plt.show()

    # IoU(pred, (gt_x, gt_y))


def predictLive(model, device):
    augmentor = Augmentor()

    streamer = StreamReader(
        src="audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{7BA2F90C-C592-4A85-8B11-73B716179C4A}",
        format="dshow",
    )

    streamer.add_basic_audio_stream(
        frames_per_chunk=44100*1, sample_rate=44100)

    stream_iterator = streamer.stream()
    wav = torch.Tensor([])
    sm = torch.nn.Softmax()

    with torch.no_grad():
        while True:
            (chunk,) = next(stream_iterator)
            # wav = torch.cat((wav, chunk[:, 0]))
            spectrogram = torchaudio.transforms.MelSpectrogram(normalized=True, sample_rate=8000, n_fft=256, n_mels=64)
            wav, sr = augmentor.audio_preprocessing([chunk.T, 44100])

            spectrogram_tensor = generateSpec(wav)
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
    model, device, _ = ml.machineLearning.selectModel(setCPU=False, modelIndex=3)
    # summary(model, (1, 201, 161))
    a = '/media/jianquan/Data/Original Audio/Singapore Speech Corpus/[P] Part 3 Same BoundaryMic/3003.wav'
    b = '../OneDrive/Desktop/untitled.wav'
    predictFile(b, model, device)

    # predictFolder(model, device, '/media/jianquan/Data/Processed Audio/')

    # predictLive(model, device)

