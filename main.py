from loader.AudioDataset import createDataset, collate_batch
from torch.utils.data import Dataset, DataLoader
import torch
import machineLearning
from model import *
from configparser import ConfigParser
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import loader.utils as utils
import testModel

if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')
    torch.backends.cudnn.benchmark = True

    utils.clearUselesslogs(minFiles=3)

    # Get Audio paths for dataset
    testRun = config['data'].getboolean('is_test_run')
    audio_train_paths, audio_val_paths = utils.getAudioPaths(
        '/media/jianquan/Data/Processed Audio/train/', percent=float(config['data']['train_percent']))

    # create dataset with transforms (as required)
    audio_train_dataset = createDataset(audio_train_paths, transformParams=utils.getTransforms(
        config['data'].getboolean('do_augmentations')), outputAudio=True)
    audio_val_dataset = createDataset(
        audio_val_paths, transformParams=utils.getTransforms(False), outputAudio=True)

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
        persistent_workers=True,
        # prefetch_factor=12,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    val_dataloader = DataLoader(
        audio_val_dataset,
        batch_size=bsize,
        num_workers=workers,
        persistent_workers=True,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_batch
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    startEpoch = 0

    if config['model'].getboolean('load_pretrained'):
        model, _, startEpoch = machineLearning.selectModel()
    else:
        model = CNNNetwork(nfft=int(config['data']['n_fft'])).to(device)

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
    if config['logger'].getboolean('master_logger') and not testRun:
        writer = SummaryWriter(logTitle)
        if config['logger'].getboolean('log_graph'):
            data = next(iter(val_dataloader))[0].to(device)
            model(data)  # to initialise lazy params
            writer.add_graph(model, data)
        writer.close()
    else:
        for i in config['logger']:
            config['logger'][i] = 'false'

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, verbose=True)

    #  train model
    for epoch in range(startEpoch+1, epochs+1):
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}/{epochs}\n-------------------------------')
        print(f'LR: {lr}')
        train_loss, train_accuracy = machineLearning.train(
            model, train_dataloader, lossFn, optimizer, device)
        if config['model'].getboolean('save_model_checkpoint') and epoch % int(config['model']['checkpoint']) == 0:
            torch.save(model, utils.uniquify(
                f'saved_model/{title}({modelIndex})_epoch{epoch}.pt'))

        val_loss, val_accuracy, _ = machineLearning.eval(
            model, val_dataloader, lossFn, device)

        if config['logger'].getboolean('log_model_params') and epoch % int(config['model']['checkpoint']) == 0:
            test_acc = testModel.predictLabeledFolders('./data', model, device)
            writer.add_hparams({'Learning Rate': lr, 'Batch Size': bsize, 'class_size': int(config['data']['class_size']), 'Epochs': epoch, 'Weight Decay': decay, 'Dropout': float(
                config['model']['dropout'])}, {'Accuracy': val_accuracy, 'Loss': val_loss, 'Test Accuracy': test_acc})

        if config['logger'].getboolean('log_iter_params'):
            machineLearning.tensorBoardLogging(writer, train_loss,
                                               train_accuracy, val_loss,
                                               val_accuracy, epoch)

        print(f'Training    | Loss: {train_loss} Accuracy: {train_accuracy}%')
        print(f'Validating  | Loss: {val_loss} Accuracy: {val_accuracy}% \n')
        scheduler.step(val_loss)

    test_acc = testModel.predictLabeledFolders('./data', model, device)

    if epoch != None:
        torch.save(model, utils.uniquify(
            f'saved_model/{title}({modelIndex})_epoch{epoch}.pt'))
    else:
        val_loss, val_accuracy, _ = machineLearning.eval(
            model, val_dataloader, lossFn, device)
        epoch = epochs + 0.1

    if config['logger'].getboolean('log_model_params') and epoch % int(config['model']['checkpoint']) != 0:
        writer.add_hparams({'Name': title, 'Learning Rate': lr, 'Batch Size': bsize, 'class_size': int(config['data']['class_size']), 'Epochs': int(epoch), 'Weight Decay': decay, 'Dropout': float(
            config['model']['dropout'])}, {'Accuracy': val_accuracy, 'Loss': val_loss, 'Test Accuracy': test_acc})
