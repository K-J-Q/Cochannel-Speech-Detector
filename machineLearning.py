import importlib
import inspect
import os
from configparser import ConfigParser
from pathlib import Path

import torch
import torchmetrics
from tqdm import tqdm

def selectModel():
    model_class = None
    files = os.listdir('model')
    files = [file for file in files if file.endswith('.py')]
    print('Available models:')
    for i, file in enumerate(files):
        print(f'{i + 1}. {file[:-3]}')

    while model_class is None:
        selection = input('Enter the number of the model you want to use: ')
        try:
            selection = int(selection)
            if selection > 0 and selection <= len(files):
                module = importlib.import_module(
                    'model.' + files[selection - 1][:-3])
                for attr in dir(module):
                    if inspect.isclass(getattr(module, attr)):
                        model_class = getattr(module, attr)
                        break
            else:
                print('Invalid selection')
        except ValueError:
            print('Invalid input')
    return model_class


def selectTrainedModel(setCPU=False, modelIndex=None):
    device = torch.device("cuda" if (torch.cuda.is_available() and not setCPU) else "cpu")
    model_paths = [str(p) for p in Path('./saved_model/').glob(f'*.pt')]
    for i, model_path in enumerate(model_paths):
        print(f'[{i}] {os.path.basename(model_path)}')
    modelIndex = input('Select saved model > ') if modelIndex == None else modelIndex
    if len(modelIndex) == 0:
        model = selectModel()
        return model, device, 0
    else:
        path = model_paths[int(modelIndex)]
        model = torch.load(path, map_location=device)
        epoch = path.split('epoch', 1)[1][:-3]
        try:
            epoch = int(epoch)
        except:
            epoch = int(epoch.split('(')[0])
        return model, device, epoch


def train(model, dataloader, cost, optimizer, device, showProgress=True):
    acc_metric = torchmetrics.Accuracy().to(device)
    total_batch = len(dataloader)
    train_loss, train_accuracy = 0, 0
    train_size = len(dataloader.dataset)
    model.train()
    percentParam = int(total_batch / 10)

    for batch, (X, Y) in tqdm(enumerate(dataloader), unit='batch', total=total_batch, disable=not showProgress):
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(X)

        if type(pred) == tuple:
            pred = pred[0]
            
        batch_loss = cost(pred, Y)
        batch_accuracy = acc_metric(pred, Y)
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()
        train_accuracy += batch_accuracy.item()
        if showProgress and percentParam and batch % percentParam == 0:
            print(f" Loss (per sample): {batch_loss.item() / len(Y)}  Accuracy: {batch_accuracy * 100}%")
        # prof.step()

    train_loss /= train_size
    train_accuracy = acc_metric.compute() * 100
    acc_metric.reset()
    return (train_loss, train_accuracy)


def eval(model, dataloader, cost, device, showProgress = True):
    model.eval()
    acc_metric = torchmetrics.Accuracy().to(device)
    num_classes = model(next(iter(dataloader))[0].to(device))[0].shape[1]
    confusion_matrix = torchmetrics.classification.MulticlassConfusionMatrix(
        num_classes).to(device)
    matrix = torch.zeros([num_classes, num_classes], device=device)

    val_size = len(dataloader.dataset)
    total_batch = len(dataloader)

    val_loss, val_accuracy = 0, 0    

    with torch.no_grad():
        for batch, (X, Y) in tqdm(enumerate(dataloader), unit='batch', total=total_batch, disable=not showProgress):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            if type(pred) == tuple:
                pred = pred[0]
            batch_loss = cost(pred, Y)
            batch_accuracy = acc_metric(pred, Y)
            val_loss += batch_loss.item()
            matrix += confusion_matrix(pred, Y)

    val_loss /= val_size
    val_accuracy = acc_metric.compute() * 100
    acc_metric.reset()
    return (val_loss, val_accuracy, matrix)


def tensorBoardLogging(writer, train_loss, train_accuracy, val_loss,
                       val_accuracy, epoch):
    writer.add_scalar('1 Training/1 Model loss', train_loss, epoch)
    writer.add_scalar('1 Training/2 Model accuracy', train_accuracy, epoch)
    writer.add_scalar('2 Validate/1 Model loss', val_loss, epoch)
    writer.add_scalar('2 Validate/2 Model accuracy', val_accuracy, epoch)
    writer.close()


if __name__ == "__main__":
    print(selectModel())
