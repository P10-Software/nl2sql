import optuna
from src.core.abstention_classifier import AbstentionClassifier
from src.common.logger import get_logger
import json

logger = get_logger(__name__)

DATA_LOCALE = 'local/AbstentionClassifier/bird_train' # Fix when in UCloud


def objective(trial: optuna.Trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
    epochs = trial.suggest_int('epochs', 2, 5)

    model = AbstentionClassifier(frozen=True)

    with open('', 'r') as fp:
        data = json.load(fp)

    f2_score = model.fine_tune(
        data=data,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        save_path=f'.local/AbstentionClassifier/binary_head/optuna_trial_n{trial.number}'
    )

    return f2_score


study = optuna.create_study(
    direction='maximize',
    study_name='Decoder Abstention Classifier',
    storage="sqlite///.local/AbstentionClassifier/BinaryHead/optuna_trials.sqlite",
    load_if_exists=True
    )
study.optimize(objective, n_trials=50)

trial = study.best_trial

logger.info(f'Best trial: {trial.number}')
logger.info(f'F2: {trial.value:.4f}')
for key, value in trial.params.items():
    logger.info(f'{key}: {value}')
