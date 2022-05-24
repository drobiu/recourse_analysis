import datetime
import os
import sys
from typing import Dict

from carla import MLModelCatalog, Data, MLModel, RecourseMethod
from carla.data.catalog import CsvCatalog
from carla.recourse_methods import Clue, Wachter


def disable():
    sys.stdout = open(os.devnull, 'w')


def enable():
    sys.stdout = sys.__stdout__


def get_timestamp():
    """
    Generates a timestamp for use in experiment identification.
    """
    time = datetime.datetime.now()
    return f"{time.day:02d}{time.hour:02d}{time.minute:02d}"


def train_model(dataset: Data, model: MLModelCatalog = None, training_params: Dict = None) -> MLModelCatalog:
    """
    Trains a new model on a given dataset.
    :param dataset: The dataset to train the model on.
    :param model: If model exists, retrain it.
    :param training_params: The hyperparameters used during training.
    :return: Newly trained MLModel object.
    """
    hyperparameters = training_params['hyperparameters']
    if not hyperparameters:
        if training_params['model_type'] == 'ann':
            hyperparameters = {"lr": 0.005, "epochs": 4, "batch_size": 1, "hidden_size": [10, 10]}

    if not model:
        model = MLModelCatalog(
            dataset,
            model_type=training_params['model_type'],
            load_online=(not isinstance(dataset, CsvCatalog)),
            backend="pytorch"
        )

    model.train(
        learning_rate=hyperparameters["lr"],
        epochs=hyperparameters["epochs"],
        batch_size=hyperparameters["batch_size"],
        hidden_size=hyperparameters["hidden_size"],
        force_train=True
    )

    return model


def train_recourse_method(
        method: str, model: MLModel, dataset=None, data_name=None, hyperparams=None
) -> RecourseMethod:
    """
    Train a new recourse generator object.
    :param method: Lowercase name of the recourse generator method.
    :param model: MLModel to train the method on.
    :param dataset: Optional Data to train the method on.
    :param data_name: Name of the dataset.
    :param hyperparams: Hyperparameters used by the recourse generator.
    :return: Newly trained recourse generator.
    """
    if Clue.__name__ == method:
        if not hyperparams:
            hyperparams = {
                "data_name": data_name,
                "train_vae": True,
                "width": 10,
                "depth": 3,
                "latent_dim": 12,
                "batch_size": 20,
                "epochs": 3,
                "lr": 0.001,
                "early_stop": 20,
            }

        # load a recourse model and pass black box model
        rm = Clue(dataset, model, hyperparams)

    else:
        if not hyperparams:
            hyperparams = {
                "loss_type": "BCE",
                "t_max_min": 3 / 60
            }

        # load a recourse model and pass black box model
        rm = Wachter(model, hyperparams)

    return rm


def train_new_model(dataset):
    model = MLModelCatalog(dataset, "ann", backend="pytorch")
    model.train(
        learning_rate=0.001,
        epochs=10,
        max_depth=50,
        n_estimators=50,
        batch_size=20,
        force_train=True
    )
    return model


def update_dataset(dataset, factuals, counterfactuals):
    for df in [dataset._df, dataset._df_test, dataset._df_train]:
        for ((i_f, r_f), (i_c, r_c)) in zip(factuals.iterrows(), counterfactuals.iterrows()):
            if len(counterfactuals.loc[i_c].dropna()) > 0 and i_f in df.index:
                df.loc[i_f] = counterfactuals.loc[i_c]
