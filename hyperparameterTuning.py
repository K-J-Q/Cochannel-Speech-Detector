import joblib
import torch
import loader.utils as utils
import machineLearning
import optuna
from optuna.trial import TrialState
from model.CNN_mel_median import CNNNetwork_mel_median
from model.CNN_mel_whisper import CNNNetwork_mel_whisper
import main
import os
import argparse
from torch_audiomentations import Compose, Identity, PolarityInversion, BandPassFilter, TimeInversion


def objective(trial):
    wd = trial.suggest_float('decay', 0.0001, 0.1, log=True)
    nfft = trial.suggest_int('nfft', 128, 1024, 128)
    normMethod = trial.suggest_categorical('normMethod', ['whisper', 'median'])
    normParam = trial.suggest_int('whisperParam', 0, 20) if normMethod == 'whisper' else trial.suggest_float('medianParam', 0, 20)
    addNoise = trial.suggest_float('addNoise', 0.0001, 0.5, log=True)
    gainDiv = trial.suggest_float('gainDiv', 0.0001, 0.1, log=True)
    augmentParams = trial.suggest_categorical('augmentation', ['noAugmentation', 'timeInv'])
    minOverlapPercentage = trial.suggest_float('overlapPercentage', 0.5 , 1)

    if augmentParams == 'timeInv':
        transform = [TimeInversion()]

    if augmentParams == "noAugmentation":
        augment = None
    else:
        augment = Compose(
            transforms=transform
        )

    train_dataloader, _ = main.create_data(
        main.trainPath, percent, num_merge, bsize, workers, addNoise, gainDiv, (minOverlapPercentage, 1))
    val_dataloader, _ = main.create_data(
        main.testPath, percent, num_merge, bsize, int(workers/2), addNoise, gainDiv, (0.7,1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNNetwork_mel_whisper(nfft, augment, outputClasses=num_merge + 1,  normParam=normParam) if normMethod == 'whisper' else CNNNetwork_mel_median(
        nfft, augment, outputClasses=num_merge + 1, normParam=normParam)

    model.to(device)

    lossFn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=wd)

    epoch = 0
    val_accuracy = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = machineLearning.train(
            model, train_dataloader, lossFn, optimizer, device, showProgress=False)
        val_loss, val_accuracy, _ = machineLearning.eval(
            model, val_dataloader, lossFn, device, showProgress=False)
        trial.report(val_accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return val_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--duration', type=int)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--num_merge', type=int, default=2)
    parser.add_argument('--bsize', type=int, default=10)
    parser.add_argument('--class_size', type=int, default=8)
    args = parser.parse_args()

    percent = 1
    NAME = args.name
    lr = args.lr
    epochs = args.epochs
    workers = args.workers
    num_merge = args.num_merge
    bsize = args.bsize
    class_size = args.class_size
    duration = args.duration

    study = optuna.create_study(study_name=NAME, direction="maximize")
    study.optimize(objective, timeout=duration, show_progress_bar=True)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # move dir to records/optuna
    os.chdir('records/optuna')

    # create folder
    if not os.path.exists(study.study_name):
        os.mkdir(study.study_name)
    os.chdir(study.study_name)

    # save data
    joblib.dump(study, utils.uniquify("study.pkl"))
    study.trials_dataframe().to_csv(utils.uniquify('optunaData.csv'))

    # visualise
    with open(utils.uniquify(f'summary.html'), 'a') as f:
        f.write(optuna.visualization.plot_optimization_history(
            study).to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(optuna.visualization.plot_slice(study).to_html(
            full_html=False, include_plotlyjs='cdn'))
        f.write(optuna.visualization.plot_parallel_coordinate(
            study).to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(optuna.visualization.plot_contour(study).to_html(
            full_html=False, include_plotlyjs='cdn'))
        f.write(optuna.visualization.plot_intermediate_values(
            study).to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(optuna.visualization.plot_edf(study).to_html(
            full_html=False, include_plotlyjs='cdn'))
        f.write(optuna.visualization.plot_param_importances(
            study).to_html(full_html=False, include_plotlyjs='cdn'))
