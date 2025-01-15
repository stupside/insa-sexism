import optuna

import logging

from cli.src.model import Model, ModelOptions
from cli.src.trainer import TrainOptions, fit
from cli.src.vectorizer import TextVectorizer

from cli.src.sets.train import TrainDataSet

study = optuna.create_study(
    study_name="tuning",
    direction="maximize",  # Single direction instead of dict
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
)


def objective(trial: optuna.Trial, vectorizer: TextVectorizer, trainset: TrainDataSet):
    vocab_size = len(vectorizer.vocabulary)

    # More appropriate parameter spaces
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_dims = []
    for i in range(n_layers):
        # Use log scale for network dimensions
        layer_dims.append(trial.suggest_int(f"layer_{i}", 64, 512, step=64))

    # Log scale for continuous parameters
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    embedding_dim = trial.suggest_int("embedding_dim", 128, 512, step=64)

    opts = ModelOptions(
        input_dim=vocab_size,
        output_dim=1,
        layer_dims=layer_dims,
        dropout_rate=dropout_rate,
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
    )

    try:
        model = Model.get(options=opts)
    except Exception as e:
        logging.error(f"Failed to create model: {e}")
        raise optuna.TrialPruned()

    # Log scale for learning parameters
    num_epochs = trial.suggest_int("num_epochs", 10, 30, step=2)
    batch_size = trial.suggest_int("batch_size", 8, 32, step=8)
    learn_rate = trial.suggest_float("learn_rate", 1e-5, 1e-3, step=1e-5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, step=1e-6)

    opts = TrainOptions(
        seed=150,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learn_rate=learn_rate,
        num_workers=0,
        weight_decay=weight_decay,
        train_val_split=0.85,
    )

    train_metrics, val_metrics = fit(model, options=opts, dataset=trainset)

    return val_metrics.f1.float()  # Return only validation F1


def tune(trials: int):
    study.optimize(
        objective,
        n_trials=trials,
        catch=(Exception,),
        callbacks=[
            lambda _, trial: logging.info(
                f"Trial {trial.number} finished with value: {trial.value}"
            )
        ],
    )

    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best value: {study.best_trial.value}")
    logging.info("Best hyperparameters:", study.best_trial.params)
