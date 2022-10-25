import torchaudio
from torchaudio.io import StreamReader
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import machineLearning
from AudioDataset import AudioDataset
import Augmentation
import random
import seaborn as sn
import matplotlib.pyplot as plt

# To ensure reproducibility
random.seed(0)
torch.manual_seed(0)

class_map = ['0', '1']


def predictFolder(folderPath, model, device):
    # load urban sound dataset
    audio_paths = Augmentation.getAudio(folderPath)[500000:510000]
    audio_paths += Augmentation.getAudio(
        'E:/Processed Singapore Speech Corpus/ENV')[0:500]

    # get a sample from the us dataset for inference
    audio_test_dataset = AudioDataset(audio_paths)

    test_dataloader = torch.utils.data.DataLoader(
        audio_test_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True,
    )

    test_loss, test_acc, confusion_matrix = machineLearning.eval(
        model, test_dataloader, torch.nn.CrossEntropyLoss(), device)
    print(f'Validating  | Loss: {test_loss} Accuracy: {test_acc}% \n')
    sn.heatmap(confusion_matrix.cpu(), annot=True,
               xticklabels=class_map, yticklabels=class_map)
    plt.show()


def predictFile(filePath, model, device):
    dataset = AudioDataset([filePath])
    testData = torch.unsqueeze(dataset[0][0], 0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(testData)
        sm = torch.nn.Softmax()
        pred = sm(pred)
        print(pred)
        # print(class_map[pred.argmax()])
        sn.barplot(y=pred[0].cpu().numpy(), x=class_map)
        plt.show()


def predictLive(model, device):
    from Augmentation import Augmentor

    augmentor = Augmentor()

    streamer = StreamReader(
        src="audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{07DEF3C3-A487-4CE6-A6E3-535301DF2D46}",
        format="dshow",
    )

    streamer.add_basic_audio_stream(
        frames_per_chunk=44100*4, sample_rate=44100)

    stream_iterator = streamer.stream()
    wav = torch.Tensor([])
    sm = torch.nn.Softmax()
    model.eval()
    with torch.no_grad():
        while True:
            (chunk,) = next(stream_iterator)
            # wav = torch.cat((wav, chunk[:, 0]))
            spectrogram = torchaudio.transforms.Spectrogram()
            wav, sr = augmentor.audio_preprocessing([chunk.T, 44100])

            spectrogram_tensor = (spectrogram(wav) + 1e-12).log2()
            pred = model(torch.unsqueeze(spectrogram_tensor, 0))
            pred = sm(pred)
            print(class_map[pred.argmax()])
            sn.barplot(y=class_map, x=pred[0].cpu().numpy())
            plt.show()


if __name__ == "__main__":
    # load back the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = [str(p) for p in Path('./saved_model/').glob(f'*.pt')]
    for i, model_path in enumerate(model_paths):
        print(f'[{i}] {model_path}')

    path = model_paths[int(input('Select saved model > '))]
    model = torch.load(path, map_location=device)

    # predictFile(
    #     'E:/Processed Singapore Speech Corpus/WAVE/000010001_0.wav', model, device)
    predictFolder(
        'E:/Processed Singapore Speech Corpus/WAVE/', model, device)
