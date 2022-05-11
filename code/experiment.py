import datetime
import json
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple

import carla
import imageio
import timeit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from carla.data.catalog import CsvCatalog, OnlineCatalog
from carla import MLModelCatalog, Data, RecourseMethod, MLModel
from carla.recourse_methods import Clue, Wachter
from carla.models.negative_instances import predict_negative_instances
from carla.evaluation.benchmark import Benchmark
from pandas import DataFrame
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn import metrics
from recourse_util import update_dataset, predict

warnings.filterwarnings("ignore")


def train_model(dataset: Data, training_params: Dict = None) -> MLModelCatalog:
    """
    Trains a new model on a given dataset.
    :param dataset: The dataset to train the model on.
    :param training_params: The hyperparameters used during training.
    :return: Newly trained MLModel object.
    """
    if not training_params:
        training_params = {"lr": 0.005, "epochs": 4, "batch_size": 1, "hidden_size": [4]}

    model = MLModelCatalog(
        dataset,
        model_type="ann",
        load_online=(not isinstance(dataset, CsvCatalog)),
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
    if method == 'CLUE':
        if not hyperparams: hyperparams = {
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
        if not hyperparams: hyperparams = {
            "loss_type": "BCE",
            "t_max_min": 0.5 / 60
        }

        # load a recourse model and pass black box model
        rm = Wachter(model, hyperparams)

    return rm


def draw(data: DataFrame, features: List, target: str):
    """
    Draw the data using pyplot scatterplot.
    :param data: Data to be plotted.
    :param features: List of data feature names.
    :param target: Target class.
    """
    plt.scatter(data[features[0]], data[features[1]], c=data[target])
    plt.show()


def get_empty_results() -> Dict:
    """
    Generate a Dict for storing experiment results in.
    """
    return {
        'datasets': [],
        'means': [],
        'covariances': [],
        'clustering': [],
        'accuracies': [],
        'f1_scores': [],
        'benchmark': [],
        'probabilities': [],
    }


def add_data_statistics(model: MLModel, dataset: Data, results: Dict):
    """
    Append the newest experiment statistics to the results Dict.
    :param model: Current MLModel.
    :param dataset: Current dataset object.
    :param results: Results Dict.
    """
    results['datasets'].append(dataset.df)
    results['means'].append(dataset._df[dataset.continuous].mean().to_numpy())
    results['covariances'].append(dataset._df[dataset.continuous].cov().to_numpy())
    results['clustering'].append(find_elbow(dataset))
    results['accuracies'].append(accuracy_score(np.array(dataset._df[dataset.target]), predict(model, dataset)))
    results['f1_scores'].append(f1_score(np.array(dataset._df[dataset.target]), predict(model, dataset)))


def find_elbow(dataset: Data, n: int = 10):
    """
    Find the amount of clusters existing in the dataset using the Caliński-Harabasz
    elbow finding metric in KMeans clustering.
    :param dataset: Current dataset.
    :param n: Number of clusters to consider.
    :return: Calculated number of clusters.
    """
    ch_metrics = []
    x = dataset.df[dataset.continuous]

    for i in range(2, n):
        model = KMeans(n_clusters=i, random_state=1).fit(x)
        ch_metrics.append(metrics.calinski_harabasz_score(x, model.labels_))

    return ch_metrics.index(np.max(ch_metrics)) + 2


def get_timestamp():
    """
    Generates a timestamp for use in experiment identification.
    """
    time = datetime.datetime.now()
    return f"{time.day}{time.hour}{time.minute}"


class CustomBenchmark(Benchmark):
    """
    Custom benchmark class extending the carla.evaluation.benchmark.Benchmark class
    allowing for setting the counterfactuals to be benchmarked and the timings
    in the class constructor.

    Parameters
    ----------
    mlmodel: MLModel
        ML model used by the benchmarking methods.
    recourse_method: RecourseMethod
        Recourse method evaluated in the benchmark.
    factuals: DataFrame
        Factual instances used in the recourse process.
    counterfactuals: DataFrame
        Counterfactual instances generated by recourse_method
    timer: int
        Amount of time used by recourse_method to generate the counterfactuals in seconds.
    """

    def __init__(
            self,
            mlmodel: MLModel,
            recourse_method: RecourseMethod,
            factuals: DataFrame,
            counterfactuals: DataFrame,
            timer: int
    ):
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
    """
    The experiment class used for setting up, running and evaluating experiments.

    Parameters
    ----------
    TODO: Parameterize the class

    Methods
    -------
    load_dataset:
        Loads the dataset used in the experiments.
        Has to be called before running any experiments.
    run_experiment:
        Runs the experiment sequence with a set amount of iterations and counterfactuals, on a set dataset.
    generate_animation:
        Generates animations using results of a given method.
    save_gifs:
        Uses generate_animation to automatically generate gifs of the recourse for both generators.
    """

    def __init__(self):

        self._iter_id = get_timestamp()
        self._data_path = 'datasets/bimodal_dataset_1.csv'
        self._logger = carla.get_logger(Experiment.__name__)

        self._features = []
        self._dataset_name = None
        self._dataset = None
        self._used_factuals_indices = set()

        self.results = {}

    def load_dataset(self, name, **kwargs):
        """
        Loads a dataset of a specified name. Use 'custom' if loading a custom dataset.
        :param name: Str containing the name of the dataset.
            'custom' when loading a custom dataset.
        :param kwargs: If using a custom dataset: A Dict containing the parameters of the dataset.
            The dict should be of the following form:
            {
                'path': a string containing the dataset path
                'categorical': a list of strings corresponding to the names of
                    the categorical features of the dataset.
                'continuous': a list of strings corresponding to the names of
                    the continuous features of the dataset.
                'immutable': a list of strings corresponding to the names of
                    the immutable features of the dataset.
                'target': a string containing the name of the target feature of the dataset.
            }
        """
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

    def run_experiment(self, save_output=False, iterations=35, samples=2):
        """
        Runs the experiment using the CLUE and Wachter recourse generators on a set amount of
        epochs and a set amount of counterfactuals per epoch. The experiment uses the dataset
        set by the load_dataset method.
        :param:
        TODO: Parameterize the method
        """
        if not self._dataset:
            self._logger.error("Load a dataset before running experiments.")
            return

        clue_dataset = deepcopy(self._dataset)
        clue_result = get_empty_results()
        self.results['CLUE'] = clue_result

        wachter_dataset = deepcopy(self._dataset)
        wachter_result = get_empty_results()
        self.results['Wachter'] = wachter_result

        self.results['metadata'] = {'iterations': iterations, 'samples': samples}

        for method in ['CLUE', 'Wachter']:
            self.results[method]['datasets'].append(self._dataset.df)

        self._logger.info(f'Starting experiment sequence with {iterations} iterations and {samples} samples.')

        for i in range(iterations):
            self._logger.info(f'Experiment iteration [{i+1}/{iterations}]:')
            clue_model, clue_factuals = self.get_factuals(clue_dataset, sample_num=samples)
            wachter_model, wachter_factuals = self.get_factuals(wachter_dataset, sample_num=samples)

            factuals = pd.merge(clue_factuals, wachter_factuals, how='inner', on=list(self._dataset._df.columns))
            factuals = pd.merge(factuals, self._dataset._df, how='inner', on=list(self._dataset._df.columns))
            factuals = factuals.drop(index=self._used_factuals_indices, errors='ignore')

            print(len(factuals))
            if len(factuals) > samples:
                factuals = factuals.sample(samples)
            if len(factuals) == 0:
                continue

            self._used_factuals_indices.update(list(factuals.index))

            self._logger.info(f'Number of factuals: {len(factuals)}')

            self._execute_experiment_iteration('CLUE', clue_dataset, clue_model, factuals, clue_result)
            self._execute_experiment_iteration('Wachter', wachter_dataset, wachter_model, factuals, wachter_result)

        if save_output:
            self.save_results()

    def _execute_experiment_iteration(
            self, method: str, dataset: Data, model: MLModel, factuals: DataFrame, results: Dict, draw_state=False
    ):
        """
        Executes a single iteration of the experiment. Trains a recourse method on the MLModel returned
        by get_factuals and the Data set by load_dataset. Uses the recourse method to generate counterfactuals,
        times the recourse generator, saves the recourse data to the results Dict using add_data_statistics,
        generates a benchmark and updates the dataset using update_dataset.

        :param method: Name of the recourse method.
        :param dataset: Data object used in the experiment iteration.
        :param model: MLModel used in the experiment iteration.
        :param factuals: DataFrame containing factuals.
        :param results: Results Dict to save the experiment results to.
        :param draw_state: Flag defining whether to show a plot of the current state of the dataset.

        :return: The updated Data object.
        """

        if method == 'CLUE':
            rm = train_recourse_method('CLUE', model, dataset, data_name=self._dataset_name)
        else:
            rm = train_recourse_method('Wachter', model)

        self._logger.info(f'Generating counterfactuals with {method}.')

        start = timeit.default_timer()
        counterfactuals = rm.get_counterfactuals(factuals)
        stop = timeit.default_timer()
        self._logger.info(f'Number of counterfactuals: {len(counterfactuals.dropna())}')

        update_dataset(dataset, factuals, counterfactuals)

        benchmark = CustomBenchmark(model, rm, factuals, counterfactuals, stop - start)
        results['benchmark'].append(benchmark.run_benchmark())

        results['probabilities'].append(self.get_probability_range(model))

        add_data_statistics(model, dataset, results)

        if draw_state:
            draw(dataset._df, self._features[:2], self._dataset.target)

        return dataset

    def get_factuals(self, dataset: Data, sample_num: int = 5, max_m_iter: int = 3) -> Tuple[MLModel, DataFrame]:
        """
        Computes the factuals as negative target class instances predicted by the model.
        Retrains the model until the prediction yields at least a set amount of
        data points or the max iterations number is reached.
        :param dataset: The dataset to predict the factuals.
        :param sample_num: Minimal amount of factuals.
        :param max_m_iter: Max iteration number.
        :return: Tuple[MLModel, DataFrame] containing the newly trained model and the factuals.
        """
        m_iter = 0

        # Train a new MLModel
        self._logger.info('Training model.')
        model = train_model(dataset)
        # Predict factuals
        factuals = predict_negative_instances(model, dataset._df)
        n_factuals = len(factuals)
        # If not enough factuals generated and the max amount of
        # iterations not reached, retrain model and try again
        while m_iter < max_m_iter and n_factuals < sample_num:
            self._logger.info(f'Not enough factuals found, retraining [{m_iter+1}/{max_m_iter}]')
            model = train_model(dataset)
            factuals = predict_negative_instances(model, dataset._df)
            n_factuals = len(factuals)
            m_iter += 1

        return model, factuals

    def get_probability_range(self, model: MLModel) -> List:
        """
        Return mean and variance of the predicted probabilities of the original positive class.
        :param model: MLModel to predict the probabilities with.
        :return: List[mean, variance]
        """
        positive = self._dataset.df.where(self._dataset.df[self._dataset.target] == 1)
        positive = positive.drop(index=self._used_factuals_indices, errors='ignore')
        prob = model.predict_proba(positive.dropna())
        return [np.mean(prob[:, 1]), np.var(prob[:, 1])]

    def generate_animation(self, results: Dict, method='CLUE', features=None):
        """
        Generates an animation using data in results for a set recourse method.
        :param results: Results Dict to be used in the animation.
        :param method: Recourse method name.
        :param features: List of two strings containing the names of features to be plotted.
        """
        data = results[method]['datasets']

        if not features:
            features = self._features[:2]

        names = [f"images/{method}{str(n)}.png" for n in range(len(data))]

        for i, name in enumerate(names):
            plt.scatter(data[i][features[0]], data[i][features[1]], c=data[i][self._dataset.target])
            plt.text(0, 0, self._dataset_name, ha='left', va='center')
            plt.text(1, 1, f"Epoch {i}/{results['metadata']['iterations']}", ha='right', va='center')
            plt.text(0, 1, method, ha='left', va='center')
            plt.text(1, 0, f"{results['metadata']['samples']} samples/epoch", ha='right', va='center')
            plt.savefig(name)
            plt.close()

        gif_path = f"gifs/{self._iter_id}_{method}_gif.gif"

        with imageio.get_writer(f'{gif_path}', mode='I') as writer:
            for filename in names:
                image = imageio.v3.imread(filename)
                writer.append_data(image)

        self._logger.info(f"Saved gif to {gif_path}")

        for filename in set(names):
            os.remove(filename)

    def save_gifs(self):
        """
        Uses generate_animation to save gifs depicting the recourse process.
        :return:
        """
        self.generate_animation(self.results, 'CLUE')
        self.generate_animation(self.results, 'Wachter')

    def save_results(self, path=None):
        """
        Save the results Dict to a file.
        """
        if not path:
            path = f'results/{self._iter_id}.json'

        out = {}
        for i in self.results:
            out[i] = {}
            for j in self.results[i]:
                if j != 'datasets':
                    out[i][j] = self.results[i][j]

        with open(path, 'w+') as file:
            file.write(json.dumps(out))

        self._logger.info(f'Saved results to {path}')


if __name__ == "__main__":
    experiment = Experiment()
    experiment.load_dataset(
        "custom",
        path='datasets/unimodal_dataset_1.csv', continuous=['feature1', 'feature2'], target='target'
    )
    experiment.run_experiment(iterations=7, samples=10)
    experiment.save_gifs()
    print(experiment.results)
