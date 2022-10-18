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

class_map = ['air conditioner', 'car horn', 'children playing', 'dog bark',
             'drilling', 'engine idling', 'gunshot', 'jackhammer', 'siren', 'street music']


def predictFolder(folderPath, model, device):
    # load urban sound dataset
    audio_paths = Augmentation.getAudio(folderPath)

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

    predictFile('./test/order-99518.wav', model, device)
    # predictFolder('./data/testset', model, device)
