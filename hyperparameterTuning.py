import joblib
import torch
import loader.utils as utils
import machineLearning
import optuna
from optuna.trial import TrialState
from model.CNN_mel import CNNNetwork_mel
import main
import os


def objective(trial):
    wd = trial.suggest_float('decay', 0.0001, 0.1, log=True)
    nfft = trial.suggest_int('nfft', 128, 1024, 128)
    dropout = trial.suggest_float('dropout', 0.01, 0.5, log=True)
    normParam = trial.suggest_int('normParam', 0, 12)
    addNoise = trial.suggest_float('addNoise', 0.0001, 0.5, log=True)
    gainDiv = trial.suggest_float('gainDiv', 0.0001, 0.1, log=True)

    train_dataloader, val_dataloader = main.create_data(main.trainPath, percent, num_merge, bsize, workers, addNoise, gainDiv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNNetwork_mel(nfft, None, outputClasses=num_merge + 1, dropout=dropout, normParam=normParam)
    model.to(device)

    lossFn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=wd)

    epoch = 2
    val_accuracy = 0
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}\n-------------------------------')
        train_loss, train_accuracy = machineLearning.train(model, train_dataloader, lossFn, optimizer, device)
        val_loss, val_accuracy, _ = machineLearning.eval(model, val_dataloader, lossFn, device)

        print(f'\nTraining    | Loss: {train_loss} Accuracy: {train_accuracy}%')
        print(f'Validating  | Loss: {val_loss} Accuracy: {val_accuracy}% \n')

        trial.report(val_accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return val_accuracy


if __name__ == '__main__':
    NAME = '2 classes'

    lr = 0.001
    epochs = 1
    workers = 6
    percent = 0.8
    num_merge = 2
    bsize = 4
    class_size = 1

    study = optuna.create_study(study_name=NAME, direction="maximize")
    study.optimize(objective, n_trials=10, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

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
    if not os.path.exists(NAME):
        os.mkdir(NAME)
    os.chdir(NAME)

    # save data
    joblib.dump(study, utils.uniquify("study.pkl"))
    study.trials_dataframe().to_csv(utils.uniquify('optunaData.csv'))

    # visualise
    with open(utils.uniquify(f'summary.html'), 'a') as f:
        f.write(optuna.visualization.plot_optimization_history(study).to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(optuna.visualization.plot_slice(study).to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(optuna.visualization.plot_parallel_coordinate(study).to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(optuna.visualization.plot_contour(study).to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(optuna.visualization.plot_intermediate_values(study).to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(optuna.visualization.plot_edf(study).to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(optuna.visualization.plot_param_importances(study).to_html(full_html=False, include_plotlyjs='cdn'))
