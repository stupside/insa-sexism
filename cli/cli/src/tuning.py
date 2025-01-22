import optuna
import logging

from typing import Dict, Any

from cli.src.model import Model, ModelOptions
from cli.src.trainer import TrainOptions, fit, validate, split
from cli.src.vectorizer import TextVectorizer
from cli.src.sets.train import TrainDataSet


def create_study() -> optuna.Study:
    return optuna.create_study(
        study_name="sexism_detection",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=2,  # Reduced from 5
            n_warmup_steps=2,  # Reduced from 5
            interval_steps=1,  # Reduced from 3
        ),
    )


def objective(
    trial: optuna.Trial, vectorizer: TextVectorizer, trainset: TrainDataSet
) -> float:
    # Simplified parameter spaces
    embedding_dim = trial.suggest_int("embedding_dim", 128, 512, step=128)
    n_layers = trial.suggest_int("n_layers", 1, 3)

    layer_dims = []
    prev_dim = embedding_dim
    for i in range(n_layers):
        dim = trial.suggest_int(f"layer_{i}", 64, prev_dim, step=64)
        layer_dims.append(dim)
        prev_dim = dim

    # Simplified training parameters
    model_params = {
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "batch_size": trial.suggest_int("batch_size", 16, 64, step=16),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
    }

    vocab_size = len(vectorizer.vocabulary)

    try:
        model_opts = ModelOptions(
            input_dim=vocab_size,
            output_dim=1,
            layer_dims=layer_dims,
            dropout_rate=model_params["dropout_rate"],
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
        )

        model = Model.get(
            vocab_size=vocab_size,
            options=model_opts,
        )

        train_opts = TrainOptions(
            seed=42,
            num_epochs=16,
            batch_size=model_params["batch_size"],
            learn_rate=model_params["learning_rate"],
            weight_decay=model_params["weight_decay"],
            train_val_split=0.85,
            num_workers=0,
        )

        print(f"Trial {trial.number} - Model: {model_opts}, Train: {train_opts}")

        # Split dataset into train and validation
        trainsubset, valsubset = split(options=train_opts, dataset=trainset)

        # Training phase
        print("Training model")
        best_train_f1 = 0.0
        for epoch, train_metrics in fit(model, subset=trainsubset, options=train_opts):
            current_f1 = train_metrics.f1.item()
            best_train_f1 = max(best_train_f1, current_f1)

            trial.report(current_f1, step=epoch)

            if trial.should_prune() and epoch > 5:  # Don't prune too early
                raise optuna.TrialPruned()

        # Validation phase
        print("Validating model")
        best_val_f1 = 0.0
        for val_metrics in validate(model, subset=valsubset, options=train_opts):
            best_val_f1 = max(best_val_f1, val_metrics.f1.item())

        logging.info(
            f"Trial finished - Train F1: {best_train_f1:.4f}, Val F1: {best_val_f1:.4f}"
        )

        return best_val_f1

    except Exception as e:
        logging.error(f"Trial failed with error: {str(e)}")
        return float("-inf")  # Return worst possible score instead of pruning


def tune_hyperparams(
    trials: int, vectorizer: TextVectorizer, trainset: TrainDataSet
) -> Dict[str, Any]:
    study = create_study()
    logging.info(f"Starting hyperparameter optimization with {trials} trials")

    study.optimize(
        lambda trial: objective(trial, vectorizer, trainset),
        n_trials=trials,
        timeout=60000,
        catch=(Exception,),
        callbacks=[
            lambda _, trial: logging.info(
                f"Trial {trial.number} finished with value: {trial.value:.4f}"
            )
        ],
    )

    if len(study.trials) == 0:
        logging.error("No trials completed successfully")
        return {"error": "No successful trials"}

    return {
        "n_trials": len(study.trials),
        "best_score": study.best_value if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else None,
    }
