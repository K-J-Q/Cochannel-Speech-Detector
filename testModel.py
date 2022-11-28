from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
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
import enum
import numpy as np

# To ensure reproducibility
random.seed(0)
torch.manual_seed(0)

class_map = ['0', '1', '2']


def predictFolder(model, device, folderPath):
    audio_paths, _ = utils.getAudioPaths(folderPath, percent=0.01)

    audio_test_dataset = createDataset(audio_paths, outputAudio=True)

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
        gt_vec = torch.zeros(len(predicted))
        predicted = torch.tensor(predicted)
        for i, startTime in enumerate(range(0, (len(predicted)-1)*windowLength, windowLength)):
            gt_vec[i] = percentageMode((get_percentage_in_window(ground_truth, startTime, startTime+windowLength)))
        print(f'Accuracy: {torch.sum(gt_vec == predicted)/len(predicted)*100}%')
            

    labelPath = filePath[0:-3]+'txt'

    pred = []
    pred_graph = [0]

    # config
    batch_size = 20
    windowLength = 2

    with torch.no_grad():
        wav, sr = augmentor.resample(
            augmentor.rechannel(torchaudio.load(filePath)))
        sampleLength = windowLength * sr
        wav = wav[0]
        batch_length = batch_size*sampleLength
        for batch in range(0, len(wav)-sampleLength, batch_length):
            data_tensor = torch.zeros([batch_size, 1,  8000 * windowLength])
            for splitIndex, j in enumerate(range(batch, batch + batch_length, sampleLength)):
                trimmed_audio = wav[j:j+sampleLength]
                # print(splitIndex, ' ', len(trimmed_audio))
                if len(trimmed_audio) == sampleLength:
                    data_tensor[splitIndex][0] = trimmed_audio
                    # plt.imshow(spectrogram_tensor[splitIndex][0])
                    # plt.show()

            # print(spectrogram_tensor)
            pred = model(data_tensor.to(device))
            # for dim in pred['conv2'][-2]:
            #     plt.imshow(dim)
            #     plt.show()
            pred = sm(pred)
            pred = pred.argmax(dim=1)
            pred_graph += list(pred.cpu().numpy())
            
    if os.path.exists(labelPath):
        gt_x, gt_y = getGroundTruth(labelPath, windowLength, len(wav)/sr)
        IoU(pred_graph, (gt_x, gt_y))
        plt.step(gt_x, gt_y)
        plt.legend(['Ground Truth', 'Predicted'])

    plt.step(torch.arange(len(pred_graph))*windowLength, pred_graph)
    plt.ylim(0, 2)
    plt.ylabel('Number of speakers')
    plt.xlabel('Time (seconds)')
    plt.yticks(torch.arange(0, 2))
    plt.xticks(torch.arange(0, len(pred_graph)*windowLength, 30))
    plt.show()

    


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
            spectrogram = torchaudio.transforms.MelSpectrogram(
                normalized=True, sample_rate=8000, n_fft=256, n_mels=64)
            wav, sr = augmentor.audio_preprocessing([chunk.T, 44100])

            spectrogram_tensor = generateSpec(wav)
            spectrogram_tensor = torch.unsqueeze(
                spectrogram_tensor, 0).to(device)
            pred = model(spectrogram_tensor)
            pred = sm(pred)
            print(pred)
            # sn.barplot(y=class_map, x=pred[0].cpu().numpy())
            # plt.show()

class filterMode(enum.Enum):
    Class_0 = 0
    Class_1 = 1
    Class_2 = 2
    Occupancy = 3

def get_percentage_in_window(groundTruth, startTime, endTime):
    label_time = np.array([0.0,0.0,0.0])
    gt_x, gt_y = groundTruth
    gt_x, gt_y = np.array(gt_x), np.array(gt_y)
    overlap = np.logical_and([gt_x > startTime],[gt_x<=endTime])
    overlapIndex = overlap.nonzero()[1]

    if len(overlapIndex):
        lastTime = startTime
        for overlap in overlapIndex:
            label_time[gt_y[overlap]] += gt_x[overlap] - lastTime
            lastTime = gt_x[overlap]
        try:
            label_time[gt_y[overlap+1]] += endTime-gt_x[overlap]
        except IndexError:
            print('end')
    else:
        label_time[gt_y[np.argmax(gt_x > endTime)]] = endTime-startTime
    label_time=label_time/(endTime-startTime)
    return label_time

def percentageMode(occupancy_label, mode = filterMode.Occupancy):
    returnIndex = np.where(np.where(occupancy_label!=0)[0] == mode.value)[0]
    return returnIndex if len(returnIndex) else occupancy_label.argmax()

def getGroundTruth(file, windowLength, audioLength):
    # time(x), label(y)
    gt_x, gt_y  = [0], [0]

    windowIndex = 0
    with open(file) as f:
        for lbl in f.read().splitlines():
            lbl = lbl.split('\t')

            startTime = float(lbl[0])
            endTime = float(lbl[1])
            num_speaker = int(lbl[2])

            if startTime != gt_x[-1]:
                gt_x.append(startTime)
                gt_y.append(1)

            gt_x.append(endTime)
            gt_y.append(num_speaker)

    return gt_x, gt_y


if __name__ == "__main__":
    model, device, _ = ml.machineLearning.selectModel(setCPU=True, modelIndex=18)

    train_nodes, eval_nodes = get_graph_node_names(model)
    return_nodes = {
        # node_name: user-specified key for output dict
        'conv1': 'conv1',
        'conv2': 'conv2',
        'conv3': 'conv3',
        'conv4': 'conv4',
    }
    # model = create_feature_extractor(model, return_nodes=return_nodes)
    model.eval()

    # summary(model, (1, 201, 161))
    a = 'E:/Original Audio/Singapore Speech Corpus/[P] Part 3 Same BoundaryMic/3003.wav'
    b = './data/1speaker_0.wav'

    predictFile(b, model, device)

    # predictFolder(model, device, 'E:/Original Audio/')

    # predictLive(model, device)
