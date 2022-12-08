from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import torchaudio
import torchmetrics
from pathlib import Path
import torch
import machineLearning
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

sm = torch.nn.Softmax(dim=1)


def predictFolder(model, device, folderPath):
    audio_paths, _ = utils.getAudioPaths(folderPath, percent=1)
    audio_test_dataset = createDataset(audio_paths, outputAudio=True)

    test_dataloader = torch.utils.data.DataLoader(
        audio_test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        collate_fn=collate_batch
    )

    test_loss, test_acc, confusion_matrix = machineLearning.eval(
        model, test_dataloader, torch.nn.CrossEntropyLoss(), device)

    print(f'Test set  | Loss: {test_loss} Accuracy: {test_acc}% \n')

    sn.heatmap(confusion_matrix.cpu(), annot=True,
               xticklabels=class_map, yticklabels=class_map)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Singapore Speech Corpus test set')
    plt.show()


def predictLabeledFolders(folderPath, model, device):
    folderPath = list(Path(folderPath).glob('**/'))
    cum_folder_acc = 0
    for folder in folderPath:
        print(f'\n---------------{folder}---------------')
        filepaths = list(Path(folder).glob('*.wav'))
        confusionMatrix = torch.zeros([3, 3])
        for path in filepaths:
            confusionMatrix += predictFile(str(
                path), model, device, plotPredicton=True if __name__ == '__main__' else False)
        cum_acc = torch.sum(torch.eye(3)*confusionMatrix) / \
            torch.sum(confusionMatrix) * 100
        cum_folder_acc += cum_acc
        print(f'Cumulative accuracy: {cum_acc}%')
        if __name__ == '__main__':
            sn.heatmap(confusionMatrix, annot=True,
                       xticklabels=class_map, yticklabels=class_map)
            datasetStats = torch.sum(confusionMatrix, 1)
            print(
                f'Dataset | Class 0: {datasetStats[0]} Class 1: {datasetStats[1]}, Class 2: {datasetStats[2]}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title(folder)
            plt.show()

    return cum_folder_acc, confusionMatrix


def predictFile(filePath, model, device, plotPredicton=True):
    model = extractModelFeature(model)
    augmentor = Augmentor()
    labelPath = filePath[0:-3]+'txt'

    pred = []
    pred_graph = []

    # config
    batch_size = 20
    windowLength = 2

    with torch.no_grad():
        wav, sr = augmentor.resample(
            augmentor.rechannel(torchaudio.load(filePath)))
        if plotPredicton:
            specData = specFeatures(model, device, wav, sr, windowLength)
        wav = torchaudio.functional.dcshift(wav, -wav.mean())
        wav = torchaudio.functional.highpass_biquad(wav, 8000, 10)
        sampleLength = windowLength * sr
        wav = wav[0]
        batch_length = batch_size*sampleLength
        for batch in range(0, len(wav)-sampleLength, batch_length):
            data_tensor = torch.zeros([batch_size, 1,  8000 * windowLength])
            for splitIndex, j in enumerate(range(batch, batch + batch_length, sampleLength)):
                trimmed_audio = wav[j:j+sampleLength]
                # print(splitIndex, ' ', len(trimmed_audio))
                if len(trimmed_audio) == sampleLength:
                    data_tensor[splitIndex][0] = trimmed_audio / \
                        trimmed_audio.max()
                else:
                    # end of audio file
                    data_tensor = data_tensor[0:splitIndex]
                    break
            if len(data_tensor) > 1:
                pred = model(data_tensor.to(device))
                if plotPredicton:
                    pred = specData.addSpec(pred)
                else:
                    pred = sm(pred['out']) if type(pred) == dict else sm(pred)

                pred = pred.argmax(dim=1)
                pred_graph += list(pred.cpu().numpy())

    predLength = len(pred_graph)

    if plotPredicton:
        ax1 = plt.subplot(2, 1, 1)
        specData.plotSpec()
        plt.title(filePath)
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        plt.tight_layout()
        plt.bar((torch.arange(predLength+1) * windowLength) -
                1, np.insert(pred_graph, 0, 0))
        plt.xlim(0, windowLength*predLength)
        plt.ylim(0, 2.5)
        plt.ylabel('Number of speakers')
        plt.xlabel('Time (seconds)')
        plt.yticks([0, 1, 2])

    if os.path.exists(labelPath):
        confusion_matrix = torchmetrics.classification.MulticlassConfusionMatrix(
            3)
        ground_truth = getGroundTruth(labelPath)
        gt_vec = torch.zeros(predLength)

        pred_graph = torch.tensor(pred_graph)
        for i, startTime in enumerate(range(0, (predLength)*windowLength, windowLength)):
            gt_vec[i] = percentageMode(get_percentage_in_window(
                ground_truth, startTime, startTime+windowLength), mode=pred_graph[i])

        num_correct_pred = torch.sum(gt_vec == pred_graph)

        gt_x, gt_y = ground_truth

        print(
            f'({os.path.basename(labelPath)}) Accuracy: {num_correct_pred/predLength *100}%')

        if plotPredicton:
            plt.step(gt_x, gt_y, 'c')
            plt.step(torch.arange(predLength+1) *
                     windowLength, np.insert(gt_vec, 0, 0), 'g')
            plt.legend(['Ground Truth', 'Computed Truth', 'Model Prediction'])
            plt.show()
        return confusion_matrix(pred_graph, gt_vec)
    else:
        plt.show()
    return 0


class specFeatures:
    specIndex = 0

    def __init__(self, model, device, wav, sr, windowLength):
        assert wav.shape[0] == 1
        modelOut = model(
            torch.unsqueeze(wav[:, 0:sr*windowLength], dim=0).to(device))
        self.windowOutputShape = modelOut['spec'].shape
        self.audioLength = len(wav[0])/sr
        self.spec = torch.zeros(
            [self.windowOutputShape[2], int(self.windowOutputShape[3]*self.audioLength/windowLength)])

    def addSpec(self, modelOutput):
        if type(modelOutput) == dict:
            for layers in modelOutput:
                if layers != 'out':
                    batchFeatures = modelOutput[layers]
                    for i, feature in enumerate(batchFeatures):
                        self.spec[:, self.specIndex: self.specIndex +
                                  self.windowOutputShape[3]] = feature[0]
                        self.specIndex += self.windowOutputShape[3]
                        # plt.title(f'{layers} ({i})')
                        # plt.imshow(feature[0].cpu())
                        # plt.show()
            return sm(modelOutput['out'])
        else:
            return sm(modelOutput)

    def plotSpec(self):
        plt.imshow(self.spec, origin='lower',
                   aspect='auto', extent=[0, self.audioLength, 0, self.windowOutputShape[2]])


def selectMicrophone():
    import subprocess
    micID = []
    command = ["ffmpeg", "-f", 'dshow', "-list_devices", 'true', "-i", "dummy"]
    out = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    s = stdout.decode("utf-8")
    print(s)

    k = s.split(']')
    for i, device in enumerate(k):
        if 'Alternative name ' in device:
            device_name = k[i-1].split('\n')[0]
            print(f"[{len(micID)}] {device_name}")
            micID.append(device.split('"')[1])

    device = micID[int(input('Select Microphone Device > '))]
    return device


def predictLive(model, device):
    augmentor = Augmentor()

    from torchaudio.io import StreamReader

    streamer = StreamReader(
        src="audio=" + selectMicrophone(),
        format="dshow",
    )

    streamer.add_basic_audio_stream(
        frames_per_chunk=8000*2, sample_rate=8000)

    stream_iterator = streamer.stream()
    wav = torch.Tensor([])

    with torch.no_grad():
        while True:
            (chunk,) = next(stream_iterator)
            # wav = torch.cat((wav, chunk[:, 0]))
            wav, sr = augmentor.audio_preprocessing([chunk.T, 8000])
            wav = torchaudio.functional.dcshift(wav, -wav.mean())
            wav /= wav.max()
            pred = model(torch.unsqueeze(wav, dim=0).to(device))
            pred = sm(pred['out']) if type(pred) == dict else sm(pred)
            print(pred)
            # sn.barplot(y=class_map, x=pred[0].cpu().numpy())
            # plt.show()


class filterEnum(enum.Enum):
    Class_0 = 0
    Class_1 = 1
    Class_2 = 2
    Occupancy = 3


def get_percentage_in_window(groundTruth, startTime, endTime):
    label_time = np.array([0.0, 0.0, 0.0])
    gt_x, gt_y = groundTruth
    gt_x, gt_y = np.array(gt_x), np.array(gt_y)
    overlap = np.logical_and([gt_x > startTime], [gt_x <= endTime])
    overlapIndex = overlap.nonzero()[1]

    if len(overlapIndex):
        lastTime = startTime
        for overlap in overlapIndex:
            label_time[gt_y[overlap]] += gt_x[overlap] - lastTime
            lastTime = gt_x[overlap]
        try:
            label_time[gt_y[overlap+1]] += endTime-gt_x[overlap]
        except IndexError:
            pass
    else:
        label_time[gt_y[np.argmax(gt_x > endTime)]] = endTime-startTime
    label_time = label_time/(endTime-startTime)
    return label_time


def percentageMode(occupancy_label, mode=filterEnum.Occupancy):
    if (occupancy_label == 1).any():
        return int(np.where(occupancy_label == 1)[0])
    elif mode == filterEnum.Occupancy:
        return int(occupancy_label.argmax())
    else:
        target_class = mode.value() if isinstance(mode, filterEnum) else int(mode)
        classOccurance = np.any(
            np.where(occupancy_label != 0)[0] == target_class)
        return target_class if classOccurance else int(occupancy_label.argmax())


def getGroundTruth(file):
    # time(x), label(y)
    gt_x, gt_y = [0], [0]

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


def extractModelFeature(model):
    return_nodes = {
        # node_name: user-specified key for output dict
        'truediv': 'spec',
        # 'elu': 'conv1',
        # 'elu_1': 'conv2',
        # 'conv3': 'conv3',
        # 'conv4': 'conv4',
        'fc3': 'out'
    }

    return create_feature_extractor(model, return_nodes=return_nodes)


if __name__ == "__main__":
    model, device, _ = machineLearning.selectModel(
        setCPU=False)

    print(get_graph_node_names(model)[1])

    model.eval()

    # summary(model, (1, 201, 161))
    a = 'E:/Original Audio/Singapore Speech Corpus/[P] Part 3 Same BoundaryMic/3003.wav'
    print(f'\n---------------------------------------\n')

    filePath = './data/test.wav'
    folderPath = './data/omni mic'

    # predictFile(filePath, model, device)

    # predictLabeledFolders(folderPath, model, device)

    predictFolder(
        model, device, 'E:/Processed Audio/test/')

    # predictLive(model, device)
