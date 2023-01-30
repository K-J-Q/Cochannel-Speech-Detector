import os
import argparse
import enum
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio.functional
import loader

maxPred = 2
windowLength = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sm = torch.nn.Softmax(dim=1)


def predictFolders(folderPath):
    folderPath = list(Path(folderPath).glob('**/'))
    assert len(folderPath) > 0, 'No files found'
    for folder in folderPath:
        filepaths = list(Path(folder).glob('*.wav'))
        if len(filepaths):
            print(f'\n---------------{folder}---------------')
            for path in filepaths:
                predictFile(str(path))


def predictFile(filePath):
    audioProcessor = loader.AudioLoader()
    labelPath = filePath[0:-3] + 'txt'

    pred = []
    pred_graph = []

    # config
    batch_size = 20

    with torch.no_grad():
        wav, sr = audioProcessor.audio_preprocessing(torchaudio.load(filePath))
        specData = specFeatures(model, device, wav, sr, windowLength)
        sampleLength = windowLength * sr
        wav = wav[0]
        batch_length = batch_size * sampleLength

        for batch in range(0, len(wav) - sampleLength, batch_length):
            data_tensor = torch.zeros([batch_size, 1, 8000 * windowLength])
            for splitIndex, j in enumerate(range(batch, batch + batch_length, sampleLength)):
                trimmed_audio = wav[j:j + sampleLength]
                if len(trimmed_audio) == sampleLength:
                    data_tensor[splitIndex][0] = trimmed_audio
                else:
                    # end of audio file
                    data_tensor = data_tensor[0:splitIndex]
                    break
            if len(data_tensor) > 1:
                pred = model(data_tensor.to(device))
                pred = specData.addSpec(pred)
                pred = pred.argmax(dim=1)
                pred_graph += list(pred.cpu().numpy())

    predLength = len(pred_graph)

    filename = filePath.split('\\')[-1]
    ax1 = plt.subplot(2, 1, 1)
    specData.plotSpec(ax1)
    plt.title(filePath)
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.tight_layout()
    plt.bar((torch.arange(predLength + 1) * windowLength) -
            1, np.insert(pred_graph, 0, 0), align='edge')
    plt.xlim(0, windowLength * predLength)
    plt.ylim(0, 2.5)
    plt.ylabel('Number of speakers')
    plt.xlabel('Time (seconds)')
    plt.yticks([0, 1, 2])

    if os.path.exists(labelPath):
        ground_truth = getGroundTruth(labelPath, maxPred=maxPred)
        gt_vec = torch.zeros(predLength)

        pred_graph = torch.tensor(pred_graph)
        for i, startTime in enumerate(range(0, predLength * windowLength, windowLength)):
            gt_vec[i] = percentageMode(get_percentage_in_window(
                ground_truth, startTime, startTime + windowLength))

        num_correct_pred = torch.sum(gt_vec == pred_graph)

        gt_x, gt_y = ground_truth

        print(f'({os.path.basename(labelPath)}) Accuracy: {num_correct_pred / predLength * 100}%')

        plt.step(gt_x, gt_y, 'c')
        plt.step(torch.arange(predLength + 1) *
                 windowLength, np.insert(gt_vec, 0, 0), 'g')
        plt.legend(['Ground Truth', 'Computed Truth', 'Model Prediction'])
        plt.show()
    else:
        plt.show()
    return 0


class specFeatures:
    specIndex = 0

    def __init__(self, model, device, wav, sr, windowLength):
        assert wav.shape[0] == 1
        modelOut = model(
            torch.unsqueeze(wav[:, 0:int(sr * windowLength)], dim=0).to(device))

        self.windowOutputShape = modelOut[1].shape
        self.audioLength = len(wav[0]) / sr

        self.spec = torch.zeros(
            [self.windowOutputShape[2], int(self.windowOutputShape[3] * self.audioLength / windowLength)])

    def addSpec(self, modelOutput):
        batchFeatures = modelOutput[1]
        for i, feature in enumerate(batchFeatures):
            self.spec[:, self.specIndex: self.specIndex + self.windowOutputShape[3]] = feature[0]
            self.specIndex += self.windowOutputShape[3]
        return sm(modelOutput[0])

    def plotSpec(self, axes):
        axes.imshow(self.spec, origin='lower',
                    aspect='auto', extent=[0, self.audioLength, 0, self.windowOutputShape[2]])
        self.specIndex = 0


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
            label_time[gt_y[overlap + 1]] += endTime - gt_x[overlap]
        except IndexError:
            pass
    else:
        label_time[gt_y[np.argmax(gt_x > endTime)]] = endTime - startTime
    label_time = label_time / (endTime - startTime)
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


def getGroundTruth(file, maxPred):
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
            gt_y.append(num_speaker if num_speaker <= maxPred else maxPred)

    return gt_x, gt_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a prediction on either a folder or a file.")
    parser.add_argument("--path", required=True, type=str, help="The path to the file or folder to predict.")
    parser.add_argument("--model", default="speech", type=str, choices=["speech", "radio"],
                        help="The type of model to use for prediction (either 'speech' or 'radio').")
    args = parser.parse_args()

    model = torch.load('resources/speechModel.pt' if args.model == "speech" else 'resources/radio.pt',
                       map_location=device)
    model.eval()

    if os.path.isdir(args.path):
        predictFolders(args.path)
    elif os.path.isfile(args.path):
        predictFile(args.path)
    else:
        print("Path specified is invalid.")
