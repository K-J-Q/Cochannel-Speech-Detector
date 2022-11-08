import torch
from tqdm import tqdm
import torchmetrics
from configparser import ConfigParser
from torch.profiler import profile, record_function, ProfilerActivity

config = ConfigParser()
config.read('config.ini')

batch_size = int((config['data']['batch_size']))
num_classes = 3


def selectModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = [str(p) for p in Path('./saved_model/').glob(f'*.pt')]
    for i, model_path in enumerate(model_paths):
        print(f'[{i}] {model_path}')

    path = model_paths[int(input('Select saved model > '))]
    model = torch.load(path, map_location=device)
    model.eval()
    return model, device

def train(model, dataloader, cost, optimizer, device):
    acc_metric = torchmetrics.Accuracy().to(device)
    total_batch = len(dataloader)
    train_loss, train_accuracy = 0, 0
    train_size = len(dataloader.dataset)
    model.train()
    print(f'Total train batch: {total_batch}')
    with profile(on_trace_ready=torch.profiler.tensorboard_trace_handler(
        './logs/traces'), record_shapes=True,
            profile_memory=True,
            with_stack=True) as prof:
        for batch, (X, Y) in tqdm(enumerate(dataloader), unit='batch', total=total_batch):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            batch_loss = cost(pred, Y)
            batch_accuracy = acc_metric(pred, Y)
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()
            train_accuracy += batch_accuracy.item()
            if batch % int(total_batch/10) == 0:
                print(
                    f" Loss (per sample): {batch_loss.item()/batch_size}  Accuracy: {batch_accuracy*100}%")
            prof.step()

    train_loss /= train_size
    train_accuracy = acc_metric.compute() * 100
    acc_metric.reset()
    return (train_loss, train_accuracy)


def eval(model, dataloader, cost, device):
    acc_metric = torchmetrics.Accuracy().to(device)
    confusion_matrix = torchmetrics.classification.MulticlassConfusionMatrix(
        num_classes).to(device)
    matrix = torch.zeros([num_classes, num_classes], device=device)

    val_size = len(dataloader.dataset)
    total_batch = len(dataloader)

    val_loss, val_accuracy = 0, 0

    model.eval()
    print(f'Total eval batch: {total_batch}')
    with torch.no_grad():
        for batch, (X, Y) in tqdm(enumerate(dataloader), unit='batch', total=total_batch):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            batch_loss = cost(pred, Y)
            batch_accuracy = acc_metric(pred, Y)
            val_loss += batch_loss.item()
            if batch % int(total_batch/10) == 0:
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
