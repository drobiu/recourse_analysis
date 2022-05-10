import warnings
from copy import deepcopy

import imageio
import timeit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from carla.data.catalog import CsvCatalog, OnlineCatalog
from carla import MLModelCatalog
from carla.recourse_methods import Clue, Wachter
from carla.models.negative_instances import predict_negative_instances
from carla.evaluation.benchmark import Benchmark
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn import metrics
from recourse_util import update_dataset, predict

warnings.filterwarnings("ignore")


def train_model(dataset):
    training_params = {"lr": 0.005, "epochs": 4, "batch_size": 1, "hidden_size": [5]}

    model = MLModelCatalog(
        dataset,
        model_type="linear",
        load_online=False,
        backend="pytorch"
    )

    model.train(
        learning_rate=training_params["lr"],
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        hidden_size=training_params["hidden_size"],
        force_train=True
    )

    return model


def train_recourse_method(method, model, dataset=None, data_name=None, hyperparams=None):
    rm = None
    if method == "clue":
        hyperparams = {
            "data_name": data_name,
            "train_vae": True,
            "width": 10,
            "depth": 3,
            "latent_dim": 12,
            "batch_size": 20,
            "epochs": 5,
            "lr": 0.0001,
            "early_stop": 20,
        }

        # load a recourse model and pass black box model
        rm = Clue(dataset, model, hyperparams)

    else:
        hyperparams = {
            "loss_type": "BCE",
            "t_max_min": 0.5 / 60
        }

        # load a recourse model and pass black box model
        rm = Wachter(model, hyperparams)

    return rm


def draw(data, features, target):
    plt.scatter(data[features[0]], data[features[1]], c=data[target])
    plt.show()


def get_factuals(dataset, sample_num=5, max_m_iter=3):
    m_iter = 0
    model = train_model(dataset)
    factuals = predict_negative_instances(model, dataset._df)
    n_factuals = len(factuals)
    while m_iter < max_m_iter and n_factuals < sample_num:
        model = train_model(dataset)
        factuals = predict_negative_instances(model, dataset._df)
        n_factuals = len(factuals)
        m_iter += 1

    return model, factuals


def get_empty_results():
    return {
        'datasets': [],
        'means': [],
        'covariances': [],
        'clustering': [],
        'accuracies': [],
        'f1_scores': [],
        'benchmark': []
    }


def add_data_statistics(model, dataset, results):
    results['datasets'].append(dataset._df.copy())
    results['means'].append(dataset._df[dataset.continuous].mean().to_numpy())
    results['covariances'].append(dataset._df[dataset.continuous].cov().to_numpy())
    results['clustering'].append(find_elbow(dataset))
    results['accuracies'].append(accuracy_score(np.array(dataset._df[dataset.target]), predict(model, dataset)))
    results['f1_scores'].append(f1_score(np.array(dataset._df[dataset.target]), predict(model, dataset)))


def find_elbow(dataset, n=10):
    ch_metrics = []
    x = dataset.df[dataset.continuous]

    for i in range(2, n):
        model = KMeans(n_clusters=i, random_state=1).fit(x)
        ch_metrics.append(metrics.calinski_harabasz_score(x, model.labels_))

    return ch_metrics.index(np.max(ch_metrics)) + 2


class CustomBenchmark(Benchmark):
    def __init__(
            self,
            mlmodel,
            recourse_method,
            factuals: pd.DataFrame,
            counterfactuals: pd.DataFrame,
            timer
    ) -> None:
        self._mlmodel = mlmodel
        self._recourse_method = recourse_method
        self._factuals = factuals.copy()
        self._counterfactuals = counterfactuals.copy()
        self._counterfactuals.index = self._factuals.index.copy()
        self._timer = timer

        # Avoid using scaling and normalizing more than once
        if isinstance(mlmodel, MLModelCatalog):
            self._mlmodel.use_pipeline = False  # type: ignore


class Experiment:
    def __init__(self):
        self.num = 10
        self._iter_id = 0
        self._data_path = 'datasets/bimodal_dataset_1.csv'

        self._features = []
        self._dataset_name = None
        self._dataset = None

        self.results = {}

    def run_experiment(self):
        if not self._dataset:
            print("Load a dataset before running experiments")

        self._iter_id += 1

        clue_dataset = deepcopy(self._dataset)
        clue_result = get_empty_results()
        self.results['clue'] = clue_result

        wachter_dataset = deepcopy(self._dataset)
        wachter_result = get_empty_results()
        self.results['wachter'] = wachter_result

        iterations = 5
        samples = 5

        for i in range(iterations):
            clue_model, clue_factuals = get_factuals(clue_dataset, sample_num=samples)
            wachter_model, wachter_factuals = get_factuals(wachter_dataset, sample_num=samples)

            factuals = pd.merge(clue_factuals, wachter_factuals, how='inner', on=list(self._dataset._df.columns))
            factuals = pd.merge(factuals, self._dataset._df, how='inner', on=list(self._dataset._df.columns))

            if len(factuals) > samples:
                factuals = factuals.sample(samples)

            self.execute_experiment_iteration('clue', clue_dataset, clue_model, factuals, clue_result)
            self.execute_experiment_iteration('wachter', wachter_dataset, wachter_model, factuals, wachter_result)

    def execute_experiment_iteration(self, method, dataset, model, factuals, results, draw_state=False):
        print("Number of factuals", len(factuals))

        if method == 'clue':
            rm = train_recourse_method('clue', model, dataset, data_name=self._dataset_name)
        else:
            rm = train_recourse_method('wachter', model)

        start = timeit.default_timer()
        counterfactuals = rm.get_counterfactuals(factuals)
        stop = timeit.default_timer()
        print("Number of counterfactuals:", len(counterfactuals.dropna()))

        update_dataset(dataset, factuals, counterfactuals)

        benchmark = CustomBenchmark(model, rm, factuals, counterfactuals, stop - start)
        results['benchmark'].append(benchmark.run_benchmark())

        add_data_statistics(model, dataset, results)

        if draw_state:
            draw(dataset._df, self._features[:2], self._dataset.target)

        return dataset

    def load_dataset(self, name, **kwargs):
        if name == 'custom':
            dataset = CsvCatalog(
                file_path=kwargs.pop('path'),
                categorical=kwargs.pop('categorical', []),
                continuous=kwargs.pop('continuous', []),
                immutables=kwargs.pop('immutables', []),
                target=kwargs.pop('target')
            )
        else:
            dataset = OnlineCatalog(name)

        self._dataset_name = name
        self._features = [*dataset.continuous, *dataset.categorical]
        self._dataset = dataset

    def generate_animation(self, results, method='clue', features=None):
        data = results[method]['datasets']

        if not features:
            features = self._features[:2]

        names = [f"images/{method}{str(n)}.png" for n in range(len(data))]

        for i, name in enumerate(names):
            plt.scatter(data[i][features[0]], data[i][features[1]], c=data[i][self._dataset.target])
            plt.savefig(name)
            plt.close()

        gif_path = f"gifs/{method}_gif_{self._iter_id}.gif"

        with imageio.get_writer(f'{gif_path}', mode='I') as writer:
            for filename in names:
                image = imageio.v3.imread(filename)
                writer.append_data(image)

        print(f"Saved gif to {gif_path}")

        for filename in set(names):
            os.remove(filename)

    def save_gifs(self):
        self.generate_animation(self.results, 'clue')
        self.generate_animation(self.results, 'wachter')


if __name__ == "__main__":
    experiment = Experiment()
    experiment.load_dataset("custom", path='datasets/bimodal_dataset_1.csv', continuous=['feature1', 'feature2'], target='target')
    experiment.run_experiment()
    experiment.save_gifs()
    print(experiment.results)
