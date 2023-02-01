import glob
import os
from configparser import ConfigParser
from datetime import datetime

import torch
import torch_audiomentations as aug
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import loader.utils as utils
import machineLearning
import testModel
from loader.AudioDataset import AudioDataset, collate_batch

testPath = 'E:/Processed Audio/test'
trainPath = 'E:/Processed Audio/train' if os.name == 'nt' else '/media/jianquan/Data/Processed Audio/train/'

startEpoch = 0

augmentations = aug.Compose(
    transforms=[
        # aug.TimeInversion(),
        # torchaudio.transforms.PitchShift(8000, 2)
        # aug.AddColoredNoise(p=1, min_snr_in_db=0, max_snr_in_db=5),
        # aug.ApplyImpulseResponse(ir_paths='E:/Processed Audio/IR'),
        # aug.AddBackgroundNoise(p=1, background_paths='E:/Processed Audio/Musical', min_snr_in_db=-3,max_snr_in_db=0)
    ]
)

augmentations = None


def create_data(audio_path, train_test_split, num_merge, batch_size, workers, addNoise, gainDiv):
    audio_train_paths, audio_val_paths = utils.getAudioPaths(audio_path, train_test_split)
    audio_train_dataset = AudioDataset(
        audio_train_paths, outputAudio=True, isTraining=True, num_merge=num_merge, add_noise=addNoise, gain_div=gainDiv)
    audio_val_dataset = AudioDataset(
        audio_val_paths, outputAudio=True, isTraining=False, num_merge=num_merge, add_noise=addNoise, gain_div=0.2)

    train_dataloader = DataLoader(
        audio_train_dataset,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_batch,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        audio_val_dataset,
        batch_size=batch_size,
        num_workers=int(workers / 2),
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_batch,
        persistent_workers=True
    )

    return train_dataloader, val_dataloader


def initiateModel(load_pretrained, nfft=None, augmentations=None, num_merge=None, normParam=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_pretrained:
        model, _, modelEpoch = machineLearning.selectTrainedModel()
        startEpoch = modelEpoch
    else:
        model = machineLearning.selectModel()
        model = model(nfft, num_merge + 1, normParam, augmentations).to(device)
        startEpoch = 0

    model.eval()

    return model, device, startEpoch

if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')

    lr = float(config['model']['learning_rate'])
    epochs = int(config['model']['num_epochs'])
    decay = float(config['model']['weight_decay'])
    bsize = int(config['data']['batch_size'])
    workers = int(config['model']['num_workers'])
    num_merge = int(config['augmentations']['num_merge'])
    percent = float(config['data']['train_percent'])
    nfft = int(config['data']['n_fft'])
    load_pretrained = config['model'].getboolean('load_pretrained')
    title = config['model']['title'] if config['model'][
        'title'] else datetime.now().strftime("%Y-%m-%d,%H-%M-%S")
    log_graph = config['logger'].getboolean('log_graph')
    checkpoint_interval = int(config['model']['checkpoint'])
    class_size = int(config['data']['class_size'])
    augment_noise = float(config['augmentations']['augment_noise'])
    gain_div = float(config['augmentations']['gain_div'])

    utils.clearUselesslogs(minFiles=3)

    train_dataloader, val_dataloader = create_data(trainPath, percent, num_merge, bsize, workers, augment_noise, gain_div, (1, 1))
    model, device, startEpoch = initiateModel(load_pretrained, nfft, augmentations, num_merge, 12)

    logTitle, modelIndex = utils.uniquify(f'./logs/{title}', True)

    writer = SummaryWriter(logTitle)

    model.to(device)

    if log_graph:
        data = next(iter(val_dataloader))[0].to(device)
        model(data)
        writer.add_graph(model, data)

    writer.close()

    # testModel.predictFolder(model, device, testPath,
    #                         f'records/{title}({modelIndex})_epoch0')

    lossFn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=decay)
    
    for epoch in range(startEpoch + 1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}\n-------------------------------')
        if epoch == 1:
            start = datetime.now()
        train_loss, train_accuracy = machineLearning.train(
            model, train_dataloader, lossFn, optimizer, device)
        if epoch % checkpoint_interval == 0:
            torch.save(model, utils.uniquify(
                f'saved_model/{title} ({modelIndex})_epoch{epoch}.pt'))
            test_acc, _ = testModel.predictFolder(
                model, device, testPath,f'records/{title}({modelIndex})_epoch{epoch}')

        val_loss, val_accuracy, _ = machineLearning.eval(
            model, val_dataloader, lossFn, device)

        if epoch % checkpoint_interval == 0:
            writer.add_hparams(
                {'Learning Rate': lr, 'Batch Size': bsize, 'class_size': class_size,
                 'Epochs': epoch, 'Weight Decay': decay},
                {'Accuracy': val_accuracy, 'Loss': val_loss, 'Test Accuracy': test_acc})

        machineLearning.tensorBoardLogging(writer, train_loss,
                                           train_accuracy, val_loss,
                                           val_accuracy, epoch)

        print(f'\nTraining    | Loss: {train_loss} Accuracy: {train_accuracy}%')
        print(f'Validating  | Loss: {val_loss} Accuracy: {val_accuracy}% \n')

        if epoch == 1:
            end = datetime.now()
            print(f'Time to complete epoch: {end - start}')
            print(
                f'Estimated time to complete training: {(end - start) * epochs}({end + (end - start) * epochs})')
        # scheduler.step(val_loss)

    test_acc, _ = testModel.predictFolder(
        model, device, testPath, f'records/{title}({modelIndex})_epoch{epoch}_acc({val_accuracy})')

    if epoch % checkpoint_interval != 0:
        writer.add_hparams({'Learning Rate': lr, 'Batch Size': bsize, 'class_size': class_size,
                            'Epochs': int(epoch), 'Weight Decay': decay},
                           {'Accuracy': val_accuracy, 'Loss': val_loss, 'Test Accuracy': test_acc})

    # delete models starting with title variable using glob
    for file in glob.glob(f'saved_model/{title} ({modelIndex})*'):
        print(f'Deleting {file}')
        os.remove(file)

    torch.save(model, utils.uniquify(
        f'saved_model/{title} ({modelIndex})_epoch{epoch}.pt'))

    print('Done!')
