from loader.AudioDataset import createDataset, collate_batch
from torch.utils.data import Dataset, DataLoader
import torch
import ml.machineLearning as machineLearning
from model import *
from configparser import ConfigParser
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import loader.utils as utils

if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')
    torch.backends.cudnn.benchmark = True

    utils.clearUselesslogs(minFiles= 3)

    # Get Audio paths for dataset
    testRun = config['data'].getboolean('is_test_run')

    audio_train_paths, audio_val_paths = utils.getAudioPaths(
        '/media/jianquan/Data/Processed Audio', percent=float(config['data']['train_percent']))
        
    # create dataset with transforms (as required)+
    audio_train_dataset = createDataset(audio_train_paths,transformParams=utils.getTransforms(config['data'].getboolean('do_augmentations')))
    audio_val_dataset = createDataset(audio_val_paths, transformParams=utils.getTransforms(False))

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
        # persistent_workers=True,
        # prefetch_factor=12,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    val_dataloader = DataLoader(
        audio_val_dataset,
        batch_size=bsize,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_batch
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    startEpoch = 0

    if config['model'].getboolean('load_pretrained'):
        model, _, startEpoch = machineLearning.selectModel()
    else:
        model = ResNet18.to(device)
        
    lr = float(config['model']['learning_rate'])
    epochs = int(config['model']['num_epochs'])
    decay = float(config['model']['weight_decay'])

    lossFn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr,weight_decay=decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr, 1, weight_decay=decay)

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

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=1)

    for ep in range(startEpoch):
        scheduler.step()

    #  train model
    for epoch in range(startEpoch, epochs):
        lr = scheduler.get_lr()[0]
        print(f'Epoch {epoch+1}/{epochs}\n-------------------------------')
        print(f'LR: {lr}')
        train_loss, train_accuracy = machineLearning.train(
            model, train_dataloader, lossFn, optimizer, device)
        if config['model'].getboolean('save_model_checkpoint') and epoch % int(config['model']['checkpoint']) == 0:
            torch.save(model, utils.uniquify(
                f'saved_model/{title}_epoch{epoch}.pt'))
        scheduler.step()

        val_loss, val_accuracy, _= machineLearning.eval(
            model, val_dataloader, lossFn, device)

        if config['logger'].getboolean('log_model_params') and epoch % int(config['model']['checkpoint']) == 0:
            writer.add_hparams(
                {'Learning Rate': lr, 'Batch Size': bsize, 'Epochs': epoch, 'Weight Decay': decay}, {'Accuracy': val_accuracy, 'Loss': val_loss})

        if config['logger'].getboolean('log_iter_params'):
            machineLearning.tensorBoardLogging(writer, train_loss,
                                               train_accuracy, val_loss,
                                               val_accuracy, epoch)

        print(f'Training    | Loss: {train_loss} Accuracy: {train_accuracy}%')
        print(f'Validating  | Loss: {val_loss} Accuracy: {val_accuracy}% \n')

    if config['logger'].getboolean('log_model_params') and epoch % int(config['model']['checkpoint']) != 0:
        writer.add_hparams({'Learning Rate': lr, 'Batch Size': bsize, 'Epochs': epoch, 'Weight Decay': decay}, {'Accuracy': val_accuracy, 'Loss': val_loss})

    torch.save(model, utils.uniquify(f'saved_model/{title}_epoch{epoch}.pt'))
