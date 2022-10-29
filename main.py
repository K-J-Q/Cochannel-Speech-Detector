from pathlib import Path
import Augmentation
from AudioDataset import transformData
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import machineLearning
from model import ResNet18, M5, CNNNetwork
from configparser import ConfigParser
from transforms import getTransforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import utils
import audiomentations
from torch.profiler import profile, record_function, ProfilerActivity
import torchaudio

if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')

    # Get Audio paths for dataset
    audio_paths = Augmentation.getAudioPaths(
        'E:/Processed Singapore Speech Corpus/Singapore Speech Corpus/')[0:10]

    print(len(audio_paths))
    # audio_paths += Augmentation.getAudioPaths(
    #     'E:/Processed Singapore Speech Corpus/ENV', 2)
    # print(len(audio_paths))
    # audio_paths = Augmentation.getAudioPaths('./data')

    test_len = int(
        int(config['data']['train_percent']) * len(audio_paths) / 100)

    audio_train_paths, audio_val_paths = audio_paths[:test_len], audio_paths[
        test_len:]

    if getTransforms(config['data'].getboolean('do_augmentations')):
        transformList = [
            {
                "before_cochannel": [
                    audiomentations.Gain(
                        min_gain_in_db=0, max_gain_in_db=3, p=0.5,),
                    audiomentations.TimeStretch(min_rate=0.8,
                                                max_rate=1.2,
                                                p=0.5,
                                                leave_length_unchanged=False,),
                    audiomentations.Shift(min_fraction=-1,
                                          max_fraction=1,
                                          p=0.5,)
                ],
                "audio": [
                    audiomentations.AddGaussianNoise(min_amplitude=0.001,
                                                     max_amplitude=0.025,
                                                     p=0.5),
                    audiomentations.PitchShift(min_semitones=-4,
                                               max_semitones=4,
                                               p=0.5),
                ],
            },
            {
                "before_cochannel": [
                    audiomentations.Gain(
                        min_gain_in_db=0, max_gain_in_db=3, p=0.5),
                    audiomentations.TimeStretch(min_rate=0.8,
                                                max_rate=1.2,
                                                p=0.5,
                                                leave_length_unchanged=False,),
                    audiomentations.Shift(min_fraction=-1,
                                          max_fraction=1,
                                          p=0.5,)
                ],
                "spectrogram": [
                    torchaudio.transforms.TimeMasking(80),
                    torchaudio.transforms.FrequencyMasking(80)
                ],
            },
        ]
    else:
        transformList = []

    # create dataset with transforms (as required)
    audio_train_dataset = transformData(
        audio_train_paths, transformList, transformParams=transformList)
    audio_val_dataset = transformData(audio_val_paths)

    print(
        f'Train dataset Length: {len(audio_train_dataset)} ({len(audio_train_paths)} before augmentation)'
    )
    print(f'Validation dataset Length: {len(audio_val_dataset)}')

    bsize = int(config['model']['batch_size'])
    workers = int(config['model']['num_workers'])

    # create dataloader for model
    train_dataloader = torch.utils.data.DataLoader(
        audio_train_dataset,
        batch_size=bsize,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        audio_val_dataset,
        batch_size=bsize,
        num_workers=1,
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
    if config['logger'].getboolean('master_logger'):
        writer = SummaryWriter(utils.uniquify(f'./logs/{title}'))
        if config['logger'].getboolean('log_graph'):
            spec, label = next(iter(val_dataloader))
            writer.add_graph(model, spec.to(device))
        writer.close()
    else:
        for i in config['logger']:
            config['logger'][i] = 'false'
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        confusion_matrix_list = []

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

        else:
            train_acc_list.append(train_accuracy)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_accuracy)
            val_loss_list.append(val_loss)

        print(f'Training    | Loss: {train_loss} Accuracy: {train_accuracy}%')
        print(f'Validating  | Loss: {val_loss} Accuracy: {val_accuracy}% \n')

    # Print out values for logging
    if not config['logger'].getboolean('master_logger'):
        print("trainLoss = ", train_loss_list)
        print("trainAcc = ", train_acc_list)
        print("valLoss = ", val_loss_list)
        print("valAcc = ", val_acc_list)
