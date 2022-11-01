import loader.Augmentation as Augmentation
from loader.AudioDataset import createDataset
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import ml.machineLearning as machineLearning
from model import ResNet18, M5, CNNNetwork
from configparser import ConfigParser
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import loader.utils as utils
import audiomentations
from torch.profiler import profile, record_function, ProfilerActivity
import torchaudio

if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')

    # Get Audio paths for dataset

    testRun = config['data'].getboolean('is_test_run')

    audio_paths = utils.getAudioPaths(
        './test_data') if testRun else utils.getAudioPaths('E:/Processed Audio/')
    print(len(audio_paths))
    # audio_paths += Augmentation.getAudioPaths(
    #     'E:/Processed Singapore Speech Corpus/ENV', 2)
    # print(len(audio_paths))
    # audio_paths = Augmentation.getAudioPaths('./data')

    audio_train_paths, audio_val_paths = torch.utils.data.random_split(audio_paths, [
                                                                       0.9, 0.1])

    # create dataset with transforms (as required)
    audio_train_dataset = createDataset(
        audio_train_paths, transformParams=utils.getTransforms(config['data'].getboolean('do_augmentations')))
    audio_val_dataset = createDataset(
        audio_val_paths, transformParams=utils.getTransforms(False))

    print(
        f'Train dataset Length: {len(audio_train_dataset)} ({len(audio_train_paths)} before augmentation)'
    )

    print(f'Validation dataset Length: {len(audio_val_dataset)}')

    bsize = int(config['model']['batch_size'])
    workers = int(config['model']['num_workers'])

    # create dataloader for model
    train_dataloader = DataLoader(
        audio_train_dataset,
        batch_size=bsize,
        num_workers=workers,
        persistent_workers=True,
        shuffle=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        audio_val_dataset,
        batch_size=bsize,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    # create model and parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNNetwork().to(device)

    # model = torch.load('./saved_model/NoAugmentations8K.pt')

    lr = float(config['model']['learning_rate'])
    epochs = int(config['model']['num_epochs'])

    lossFn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    title = config['model']['title'] if config['model'][
        'title'] else datetime.now().strftime("%Y-%m-%d,%H-%M-%S")

    # TensorBoard logging (as required)
    if config['logger'].getboolean('master_logger') and not testRun:
        writer = SummaryWriter(utils.uniquify(f'./logs/{title}'))
        if config['logger'].getboolean('log_graph'):
            spec, label = next(iter(val_dataloader))
            writer.add_graph(model, spec.to(device))
        writer.close()
    else:
        for i in config['logger']:
            config['logger'][i] = 'false'

    #  train model
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}\n-------------------------------')
        train_loss, train_accuracy = machineLearning.train(
            model, train_dataloader, lossFn, optimizer, device)
        if config['model'].getboolean('save_model_checkpoint') and epoch % int(config['model']['checkpoint']) == 0:
            torch.save(model, utils.uniquify(
                f'saved_model/{title}_epoch{epoch}.pt'))

        val_loss, val_accuracy, _ = machineLearning.eval(model, val_dataloader,
                                                         lossFn, device)

        if config['logger'].getboolean('log_model_params') and epoch % int(config['model']['checkpoint']) == 0:
            writer.add_hparams(
                {'Learning Rate': lr, 'Batch Size': bsize, 'Epochs': epoch}, {'Accuracy': val_accuracy, 'Loss': val_loss})

        if config['logger'].getboolean('log_iter_params'):
            machineLearning.tensorBoardLogging(writer, train_loss,
                                               train_accuracy, val_loss,
                                               val_accuracy, epoch)

        print(f'Training    | Loss: {train_loss} Accuracy: {train_accuracy}%')
        print(f'Validating  | Loss: {val_loss} Accuracy: {val_accuracy}% \n')
