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
import torchaudio
import optuna

testPath = './data/omni mic/real'
trainPath = 'E:/Processed Audio/train' if os.name == 'nt' else '/media/jianquan/Data/Processed Audio/train/'

startEpoch = 0

augmentations = aug.Compose(
    transforms=[
        # aug.TimeInversion(),
        # torchaudio.transforms.PitchShift(8000, 2)
        # aug.AddColoredNoise(p=1, min_snr_in_db=0, max_snr_in_db=5),
        # aug.ApplyImpulseResponse(ir_paths='E:/Processed Audio/IR'),
        # aug.AddBackgroundNoise(p=1, background_paths='E:/Processed Audio/backgroundNoise', min_snr_in_db=-3,max_snr_in_db=0)
    ]
)

augmentations = None



if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')

    utils.clearUselesslogs(minFiles=3)

    num_merge = int(config['augmentations']['num_merge'])

    audio_train_paths, audio_val_paths = utils.getAudioPaths(
        trainPath, percent=float(config['data']['train_percent']))

    # create dataset with transforms (as required)
    audio_train_dataset = AudioDataset(
        audio_train_paths, outputAudio=True, isTraining=True, num_merge=num_merge)
    audio_val_dataset = AudioDataset(
        audio_val_paths, outputAudio=True, isTraining=False, num_merge=num_merge)

    print(
        f'Train dataset Length: {len(audio_train_dataset)} ({len(audio_train_paths[0])} before augmentation)'
    )

    print(f'Validation dataset Length: {len(audio_val_dataset)}')

    bsize = int(config['data']['batch_size'])
    workers = int(config['model']['num_workers'])

    # create dataloader for model
    train_dataloader = DataLoader(
        audio_train_dataset,
        batch_size=bsize,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_batch,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        audio_val_dataset,
        batch_size=bsize,
        num_workers=int(workers / 2),
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_batch,
        persistent_workers=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if config['model'].getboolean('load_pretrained'):
        model, _, modelEpoch = machineLearning.selectTrainedModel()
        startEpoch = modelEpoch if startEpoch == 0 else startEpoch
    else:
        model = machineLearning.selectModel()
        model = model(int(config['data']['n_fft']),
                      augmentations, outputClasses=num_merge+1).to(device)

    model.eval()
    lr = float(config['model']['learning_rate'])
    epochs = int(config['model']['num_epochs'])
    decay = float(config['model']['weight_decay'])

    lossFn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr, 1, weight_decay=decay)

    title = config['model']['title'] if config['model'][
        'title'] else datetime.now().strftime("%Y-%m-%d,%H-%M-%S")
    logTitle, modelIndex = utils.uniquify(f'./logs/{title}', True)
    # TensorBoard logging (as required)
    if config['logger'].getboolean('master_logger'):
        writer = SummaryWriter(logTitle)
        if config['logger'].getboolean('log_graph'):
            data = next(iter(val_dataloader))[0].to(device)
            model(data)  # to initialise lazy params
            writer.add_graph(model, data)
        writer.close()
    else:
        for i in config['logger']:
            config['logger'][i] = 'false'

    testModel.predictFolder(model, device, 'E:/Processed Audio/test',
                            f'records/{title}({modelIndex})_epoch0')

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=2, verbose=True)

    #  train model
    for epoch in range(startEpoch + 1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}\n-------------------------------')
        if epoch == 1:
            start = datetime.now()
        train_loss, train_accuracy = machineLearning.train(
            model, train_dataloader, lossFn, optimizer, device)
        if config['model'].getboolean('save_model_checkpoint') and epoch % int(config['model']['checkpoint']) == 0:
            torch.save(model, utils.uniquify(
                f'saved_model/{title} ({modelIndex})_epoch{epoch}.pt'))
            test_acc, _ = testModel.predictFolder(
                model, device, 'E:/Processed Audio/test', f'records/{title}({modelIndex})_epoch{epoch}')

        val_loss, val_accuracy, _ = machineLearning.eval(
            model, val_dataloader, lossFn, device)

        if config['logger'].getboolean('log_model_params'):
            if epoch % int(config['model']['checkpoint']) == 0:
                writer.add_hparams(
                    {'Learning Rate': lr, 'Batch Size': bsize, 'class_size': int(config['data']['class_size']),
                     'Epochs': epoch, 'Weight Decay': decay, 'Dropout': float(
                        config['model']['dropout'])},
                    {'Accuracy': val_accuracy, 'Loss': val_loss, 'Test Accuracy': test_acc})
            else:
                machineLearning.tensorBoardLogging(writer, train_loss,
                                                   train_accuracy, val_loss,
                                                   val_accuracy, epoch)

        print(
            f'\nTraining    | Loss: {train_loss} Accuracy: {train_accuracy}%')
        print(f'Validating  | Loss: {val_loss} Accuracy: {val_accuracy}% \n')

        if epoch == 1:
            end = datetime.now()
            print(f'Time to complete epoch: {end - start}')
            print(
                f'Estimated time to complete training: {(end - start) * epochs}({end + (end - start) * epochs})')
        # scheduler.step(val_loss)

    test_acc, _ = testModel.predictFolder(
        model, device, 'E:/Processed Audio/test', f'records/{title}({modelIndex})_epoch{epoch}_acc({val_accuracy})')

    if config['logger'].getboolean('log_model_params') and epoch % int(config['model']['checkpoint']) != 0:
        writer.add_hparams({'Learning Rate': lr, 'Batch Size': bsize, 'class_size': int(config['data']['class_size']),
                            'Epochs': int(epoch), 'Weight Decay': decay, 'Dropout': float(Jconfig['model']['dropout'])}, {'Accuracy': val_accuracy, 'Loss': val_loss, 'Test Accuracy': test_acc})

    # delete models starting with title variable using glob
    for file in glob.glob(f'saved_model/{title} ({modelIndex})*'):
        print(f'Deleting {file}')
        os.remove(file)

    torch.save(model, utils.uniquify(
        f'saved_model/{title} ({modelIndex})_epoch{epoch}.pt'))

    print('Done!')
