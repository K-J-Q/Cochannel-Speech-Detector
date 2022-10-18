import torch
from tqdm import tqdm
import torchmetrics


def train(model, dataloader, cost, optimizer, device):
    acc_metric = torchmetrics.Accuracy().to(device)

    batch_size = len(next(iter(dataloader))[1])
    total_batch = len(dataloader)
    train_loss, train_accuracy = 0, 0
    train_size = len(dataloader.dataset)
    model.train()
    print(f'Total train batch: {total_batch}')
    for batch, (X, Y) in tqdm(enumerate(dataloader), unit='batch'):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        batch_loss = cost(pred, Y)
        batch_accuracy = acc_metric(pred, Y)
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss.item()
        train_accuracy += batch_accuracy.item()
        if batch % 50 == 0:
            print(
                f" Loss (per sample): {batch_loss.item()/batch_size}  Accuracy: {batch_accuracy*100}%")
    train_loss /= train_size
    train_accuracy = acc_metric.compute() * 100
    acc_metric.reset()
    return (train_loss, train_accuracy)


def eval(model, dataloader, cost, device):
    acc_metric = torchmetrics.Accuracy().to(device)
    confusion_matrix = torchmetrics.classification.MulticlassConfusionMatrix(
        10).to(device)
    matrix = torch.zeros([10, 10], device=device)

    val_size = len(dataloader.dataset)
    batch_size = len(next(iter(dataloader))[1])
    total_batch = len(dataloader)

    val_loss, val_accuracy = 0, 0

    model.eval()
    print(f'Total eval batch: {total_batch}')
    with torch.no_grad():
        for batch, (X, Y) in tqdm(enumerate(dataloader), unit='batch'):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            batch_loss = cost(pred, Y)
            batch_accuracy = acc_metric(pred, Y)
            val_loss += batch_loss.item()
            if batch % 100 == 0:
                print(
                    f" Loss (per sample): {batch_loss.item()/batch_size}  Accuracy: {batch_accuracy*100}%"
                )
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

