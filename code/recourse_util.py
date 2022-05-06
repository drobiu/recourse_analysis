from carla.data.catalog import OnlineCatalog
from carla import MLModelCatalog
from carla.recourse_methods import Clue
from carla.models.negative_instances import predict_negative_instances
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


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
    #     for index, row in factuals.iterrows():
    #         fac_ind.append(index)
    #     for index, row in counterfactuals.iterrows():
    #         dataset.loc[index] = counterfactuals.loc[index]

    for ((i_f, r_f), (i_c, r_c)) in zip(factuals.iterrows(), counterfactuals.iterrows()):
        if len(counterfactuals.loc[i_c].dropna()) > 0:
            dataset.loc[i_f] = counterfactuals.loc[i_c]


def train_recourse_method(dataset, model, data_name, method, hyperparams=None):
    rm = None
    if method == "clue":
        if hyperparams is None:
            hyperparams = {
                "data_name": data_name,
                "train_vae": True,
                "width": 10,
                "depth": 3,
                "latent_dim": 12,
                "batch_size": 64,
                "epochs": 1,
                "lr": 0.001,
                "early_stop": 20,
            }

        # load a recourse model and pass black box model
        rm = Clue(dataset, model, hyperparams)

    return rm


def predict(model, data):
    pred = model.predict(data._df)
    return np.where(pred > 0.5, 1, 0)


def print_f1_score(model, data):
    score = f1_score(np.array(data._df[data.target]), predict(model, data))
    print(f"F1 score: {score}")


def print_accuracy(model, data):
    score = accuracy_score(np.array(data._df[data.target]), predict(model, data))
    print(f"Accuracy score: {score}")


def print_scores(model_pre, model_post, data):
    print("Before recourse:")
    print_f1_score(model_pre, data)
    print_accuracy(model_pre, data)
    print("\nAfter recourse:")
    print_f1_score(model_post, data)
    print_accuracy(model_post, data)