import datetime
import json
import sys
import warnings
from copy import deepcopy
from functools import reduce
from typing import Dict, List, Tuple

import carla
import imageio
import timeit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

from carla.data.catalog import CsvCatalog, OnlineCatalog
from carla import MLModelCatalog, Data, RecourseMethod, MLModel
from carla.recourse_methods import Clue, Wachter
from carla.models.negative_instances import predict_negative_instances
from carla.evaluation.benchmark import Benchmark
from pandas import DataFrame
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn import metrics
from recourse_util import update_dataset, predict
from kneed import KneeLocator

warnings.filterwarnings("ignore")


def disable():
    sys.stdout = open(os.devnull, 'w')


def enable():
    sys.stdout = sys.__stdout__


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
        'pred_data': [],
        'mmd': [],
        'mmd_p_value': [],
        'disagreement': [],
        'model_mmd': [],
        'prob_mmd': [],
    }


def add_data_statistics(dataset: Data, results: Dict, model: MLModelCatalog = None):
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
    results['probabilities'].append(model.predict(dataset.df).flatten())


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
    clusters = []
    entropy = []

    for i in range(2, n):
        model = KMeans(n_clusters=i, random_state=1).fit(x)
        clusters.append(i)
        entropy.append(model.inertia_)
        ch_metrics.append(metrics.calinski_harabasz_score(x, model.labels_))

    return KneeLocator(clusters, entropy, S=1.0, curve="convex", direction="decreasing").elbow


def mmd(df_a: DataFrame, df_b: DataFrame, target: str) -> float:
    """
    Computes the Maximum Mean Discrepancy metric using formula from
    Gretton et al. (2012) https://dl.acm.org/doi/10.5555/2188385.2188410
    :param df_a: DataFrame of the first distribution
    :param df_b: DataFrame of the second distribution
    :param target: str
    :return: float MMD metric for the two DataFrames
    """
    df_a = df_a.loc[df_a[target] == 1].sample(100, replace=True).drop(target, axis=1)
    df_b = df_b.loc[df_b[target] == 1].sample(100, replace=True).drop(target, axis=1)

    df_c = df_a.append(df_b)

    distances = pdist(df_c, 'sqeuclidean')

    sigma = np.sqrt(np.median(distances))

    total = 0
    len_a = len(df_a)
    len_b = len(df_b)
    len_c = len(df_c)

    def get_dist_index(i, j, m):
        return int(m * i + j - ((i + 2) * (i + 1)) / 2)

    def k_fun(dist, sigma):
        return np.exp(-(1 / sigma) * dist)

    for i in range(len_a):
        for j in range(len_b):
            if i != j:
                total += k_fun(distances[get_dist_index(i, j, len_c)], sigma) / (len_a ** 2 - len_a)
                total += k_fun(distances[get_dist_index(i + len_a, j + len_a, len_c)], sigma) / (len_b ** 2 - len_b)
            total -= 2 * k_fun(distances[get_dist_index(i, j + len_a, len_c)], sigma) / (len_a * len_b)

    return total


def mmd_sklearn(df_a: DataFrame, df_b: DataFrame, target: str = None, samples=0.1) -> float:
    """
    Computes the Maximum Mean Discrepancy metric using formula from
    Gretton et al. (2012) https://dl.acm.org/doi/10.5555/2188385.2188410
    Uses sklearn.metrics.pairwise.rbf_kernel, it is more stable than the
    manual kernel calculation method.
    :param df_a: DataFrame of the first distribution
    :param df_b: DataFrame of the second distribution
    :param target: str
    :return: float MMD metric for the two DataFrames
    """

    if target:
        df_a = df_a.loc[df_a[target] == 1].drop(target, axis=1)
        df_b = df_b.loc[df_b[target] == 1].drop(target, axis=1)

    len_a = len(df_a)
    len_b = len(df_b)

    df_a = df_a.sample(min(len_a, max(1000, int(len_a * samples))))
    df_b = df_b.sample(min(len_b, max(1000, int(len_b * samples))))

    len_a = len(df_a)
    len_b = len(df_b)

    df_c = df_a.append(df_b)

    distances = pdist(df_c, 'sqeuclidean')

    sigma = np.sqrt(np.median(distances))

    total = 0

    total += np.sum(rbf_kernel(df_a, gamma=1 / sigma), axis=None) / (len_a ** 2 - len_a)
    total += np.sum(rbf_kernel(df_b, gamma=1 / sigma), axis=None) / (len_b ** 2 - len_b)
    total -= 2 * np.sum(rbf_kernel(df_a, df_b, gamma=1 / sigma), axis=None) / (len_a * len_b)

    return total


def mmd_p_value(df_a: DataFrame, df_b: DataFrame, target_mmd, target, iterations=1000):
    merged = df_a.append(df_b, ignore_index=True)
    merged = merged.loc[merged[target] == 1]
    ge = 0
    for i in range(iterations):
        shuffled = merged.sample(frac=1)
        len_shuffled = len(shuffled)
        half_a = shuffled.iloc[:int(len_shuffled/2)]
        half_b = shuffled.iloc[int(len_shuffled/2):]
        if mmd_sklearn(half_a, half_b) >= target_mmd:
            ge += 1

    return ge/iterations


def disagreement(model_a: MLModelCatalog, model_b: MLModelCatalog, data: Data) -> float:
    """
    Calculates the model disagreement pseudo-metric
    :param model_a: First model to be compared
    :param model_b: Second model to be compared
    :param data: The data on which to calculate the metric
    :return: The model disagreement
    """
    pred_a = predict(model_a, data)
    pred_b = predict(model_b, data)
    return sum([1 if a != b else 0 for (a, b) in zip(pred_a, pred_b)]) / len(data.df)


def get_timestamp():
    """
    Generates a timestamp for use in experiment identification.
    """
    time = datetime.datetime.now()
    return f"{time.day:02d}{time.hour:02d}{time.minute:02d}"


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


class RecourseMethod:
    def __init__(self, name, class_type, hyperparameters):
        self.name = name
        self.type = class_type
        self.hyperparameters = hyperparameters
        self.dataset = None
        self.factuals = None
        self.model = None


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

    def __init__(self, **kwargs):

        self._iter_id = get_timestamp()
        self._data_path = 'datasets/bimodal_dataset_1.csv'
        self._logger = carla.get_logger(Experiment.__name__)
        self._out_count = 0
        self._options = {
            'generate_meshes': kwargs.get('generate_meshes', True)
        }

        self._features = []
        self._dataset_name = None
        self._dataset = None
        self._first_model = None
        self._used_factuals_indices = set()

        self.results = {}

        self._model_options = kwargs.pop('model', False)
        self._generator_options = kwargs.pop('generators', False)

        if self._generator_options:
            self._methods = {name: RecourseMethod(name, obj['class'], obj['hyperparameters'])
                             for name, obj in self._generator_options.items()}

        self._meshes = {k: [] for k in self._methods}
        self._low_res_points = None
        self._low_res_meshes = {k: [] for k in self._methods}

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
        self._dataset = dataset
        self._features = dataset.df.drop(self._dataset.target, axis=1).columns.values.tolist()

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

        for method in self._methods:
            self._methods[method].dataset = deepcopy(self._dataset)
            self.results[method] = get_empty_results()

        self._first_model = train_model(self._dataset, training_params=self._model_options)

        self.results['metadata'] = {
            'iterations': iterations,
            'samples': samples,
            'generators': self._generator_options,
            'model': self._model_options,
        }

        self._logger.info(f'Starting experiment sequence with {iterations} iterations and {samples} samples.')

        for i in range(iterations):
            self._logger.info(f'Experiment iteration [{i + 1}/{iterations}]:')

            for method, obj in self._methods.items():
                model, factuals = self.get_factuals(obj.dataset, obj.model, sample_num=samples)
                obj.model = model
                obj.factuals = factuals

            if self._options['generate_meshes']:
                self.update_meshes({name: obj.model for name, obj in self._methods.items()})

            self.update_meshes_low_res({name: obj.model for name, obj in self._methods.items()})

            factuals = reduce(lambda left, right: pd.merge(left, right, on=list(self._dataset.df.columns),
                                                           how='inner'),
                              [obj.factuals for obj in self._methods.values()])

            factuals = pd.merge(factuals, self._dataset.df, how='inner', on=list(self._dataset.df.columns))
            factuals = factuals.drop(index=self._used_factuals_indices, errors='ignore')

            print(len(factuals))
            if len(factuals) > samples:
                factuals = factuals.sample(samples)
            if len(factuals) == 0:
                continue

            self._used_factuals_indices.update(list(factuals.index))

            self._logger.info(f'Number of factuals: {len(factuals)}')

            for name, obj in self._methods.items():
                self._execute_experiment_iteration(name, obj.dataset, obj.model, factuals, self.results[name])

        if self._options['generate_meshes']:
            self.update_meshes({name: obj.model for name, obj in self._methods.items()})

        self.update_meshes_low_res({name: obj.model for name, obj in self._methods.items()})

        for name, obj in self._methods.items():
            self._update_last_epoch(obj.dataset, obj.model, obj.name, samples)

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

        disable()
        rm = train_recourse_method(self._methods[method].type, model, dataset,
                                   data_name=self._dataset_name, hyperparams=self._methods[method].hyperparameters
                                   )
        enable()

        self._logger.info(f'Generating counterfactuals with {method}.')

        disable()
        start = timeit.default_timer()
        counterfactuals = rm.get_counterfactuals(factuals)
        stop = timeit.default_timer()
        enable()
        self._logger.info(f'Number of counterfactuals: {len(counterfactuals.dropna())}')

        benchmark = CustomBenchmark(model, rm, factuals, counterfactuals, stop - start)
        results['benchmark'].append(benchmark.run_benchmark().to_json())

        results['pred_data'].append(self.get_probability_range(model))

        a = timeit.default_timer()
        results['mmd'].append(mmd_sklearn(self._dataset.df, dataset.df, self._dataset.target))
        b = timeit.default_timer()
        print(b - a)

        a = timeit.default_timer()
        results['mmd_p_value'].append(
            mmd_p_value(self._dataset.df, dataset.df, results['mmd'][-1], self._dataset.target))
        b = timeit.default_timer()
        print(results['mmd_p_value'][-1])
        print(b - a)

        results['disagreement'].append(disagreement(self._first_model, model, self._dataset))

        a = timeit.default_timer()
        results['model_mmd'].append(self.compute_prob_model_shift(self._low_res_meshes[method]))
        b = timeit.default_timer()
        print(b - a)

        add_data_statistics(dataset, results, model)

        results['prob_mmd'].append(mmd_sklearn(DataFrame(results['probabilities'][0]),
                                               DataFrame(results['probabilities'][-1]), ''))

        update_dataset(dataset, factuals, counterfactuals)

        if draw_state:
            draw(dataset.df, self._features[:2], self._dataset.target)

        return dataset

    def get_factuals(self, dataset: Data, model: MLModelCatalog = None, sample_num: int = 5, max_m_iter: int = 3
                     ) -> Tuple[MLModel, DataFrame]:
        """
        Computes the factuals as negative target class instances predicted by the model.
        Retrains the model until the prediction yields at least a set amount of
        data points or the max iterations number is reached.
        :param dataset: The dataset to predict the factuals.
        :param model: If model exists, retrain and use it.
        :param sample_num: Minimal amount of factuals.
        :param max_m_iter: Max iteration number.
        :return: Tuple[MLModel, DataFrame] containing the newly trained model and the factuals.
        """
        m_iter = 0

        # Train a new MLModel
        self._logger.info('Training model.')
        disable()
        model = train_model(dataset, model, self._model_options)
        # Predict factuals
        factuals = predict_negative_instances(model, dataset.df)
        n_factuals = len(factuals)
        # If not enough factuals generated and the max amount of
        # iterations not reached, retrain model and try again
        enable()
        while m_iter < max_m_iter and n_factuals < sample_num:
            self._logger.info(f'Not enough factuals found, retraining [{m_iter + 1}/{max_m_iter}]')
            disable()
            model = train_model(dataset, model, self._model_options)
            factuals = predict_negative_instances(model, dataset.df)
            n_factuals = len(factuals)
            enable()
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

    def update_meshes(self, models):
        resolution = 100
        for m in self._methods:
            a = np.linspace(-0.1, 1.1, resolution)
            xx, yy = np.meshgrid(a, a)

            data = np.column_stack((xx.flatten(), yy.flatten()))

            pred = models[m].predict(data)

            def smoothstep(x):
                return (1 + 1000000 ** (-x + 0.5)) ** (-1)

            self._meshes[m].append((xx, yy, smoothstep(pred)))

    def update_meshes_low_res(self, models):
        if not self._low_res_points:
            resolution = int(np.ceil(10000 ** (1 / len(self._dataset.df.columns))))

            df = self._dataset.df.drop(self._dataset.target, axis=1)
            ranges = [np.linspace(df[col].min(), df[col].max(), resolution) for col in df.columns.values]
            self._low_res_points = np.meshgrid(*ranges)

        for m in self._methods:
            pred_df = DataFrame()

            for col, data in zip(self._features, self._low_res_points):
                pred_df[col] = data.flatten()

            # data = np.column_stack(tuple(*(coord.flatten() for coord in coords)))

            pred = models[m].predict(pred_df)
            pred_df['pred'] = pred

            # size = max(100, int(len(pred_df) * 0.1))

            self._low_res_meshes[m].append(pred_df)

    def _update_last_epoch(self, dataset, model, method, samples):
        model, factuals = self.get_factuals(dataset, model, sample_num=samples)

        self.results[method]['mmd'].append(mmd_sklearn(self._dataset.df, dataset.df, self._dataset.target))

        self.results[method]['disagreement'].append(disagreement(self._first_model, model, self._dataset))

        self.results[method]['model_mmd'].append(self.compute_prob_model_shift(self._low_res_meshes[method]))

        add_data_statistics(dataset, self.results[method], model)

        self.results[method]['prob_mmd'].append(mmd_sklearn(DataFrame(self.results[method]['probabilities'][0]),
                                                            DataFrame(self.results[method]['probabilities'][-1]), ''))

    def compute_prob_model_shift(self, meshes):
        # data_a = zip(meshes[0][0].flatten(), meshes[0][1].flatten(), meshes[0][2].flatten())
        # df_a = DataFrame(np.array([[a, b, c, 1] for (a, b, c) in data_a]),
        #                  columns=self._first_model.feature_input_order)
        #
        # data_b = zip(meshes[-1][0].flatten(), meshes[-1][1].flatten(), meshes[-1][2].flatten())
        # df_b = DataFrame(np.array([[a, b, c, 1] for (a, b, c) in data_b]),
        #                  columns=self._first_model.feature_input_order)

        # return mmd_sklearn(df_a, df_b, 'target')
        return mmd_sklearn(meshes[0], meshes[-1])

    def generate_animation(self, results: Dict, method='CLUE', options=None, features=None):
        """
        Generates an animation using data in results for a set recourse method.
        :param results: Results Dict to be used in the animation.
        :param method: Recourse method name.
        :param features: List of two strings containing the names of features to be plotted.
        """
        data = results[method]['datasets']

        if not features:
            features = self._features[:2]

        names = [f"images/{method}{str(n)}_{self._iter_id}.png" for n in range(len(data))]
        out_names = []

        coloring_type = options.get('type', 'default')

        if coloring_type == 'default':
            colors = [df[self._dataset.target] for df in data]
        elif coloring_type == 'pred_class':
            colors = [np.where(prob > 0.5, 1, 0) for prob in results[method]['probabilities']]
        elif coloring_type == 'prob':
            colors = results[method]['probabilities']

        fpi = options.get('slow', 3)
        mesh = options.get('mesh', True) and self._options['generate_meshes']
        large = options.get('large', True)

        for i, name in enumerate(names):
            for n in range(fpi):
                if large:
                    fig = plt.figure(constrained_layout=True, figsize=(13, 6))
                    axs = fig.subplot_mosaic(
                        [['left', 'top_middle', 'top_right'], ['left', 'bottom_middle', 'bottom_right']],
                        gridspec_kw={'width_ratios': [2, 1, 1]})

                    if mesh:
                        xx, yy, pred = self._meshes[method][i]
                        axs['left'].contourf(xx, yy, pred.T[0].reshape(xx.shape[0], xx.shape[0]), levels=10)
                    axs['left'].scatter(
                        data[i][features[0]],
                        data[i][features[1]],
                        c=np.where(colors[i] > 0.5, '#c78f1e', '#0096f0'),
                        edgecolors='black'
                    )
                    axs['left'].axis(xmin=-0.1, xmax=1.1, ymin=-0.1, ymax=1.1)

                    axs['top_middle'].plot(results[method]['mmd'][:i + 1])
                    axs['top_middle'].set_title('MMD')
                    axs['top_right'].plot(results[method]['disagreement'][:i + 1])
                    axs['top_right'].set_title('Disagreement')
                    axs['bottom_right'].plot(results[method]['model_mmd'][:i + 1])
                    axs['bottom_right'].set_title('Model MMD')
                    axs['bottom_middle'].plot(results[method]['prob_mmd'][:i + 1])
                    axs['bottom_middle'].set_title('Probability MMD')

                    f_name = f'{name[:-4]}_{n}.png'
                    fig.savefig(f_name)
                    plt.close()

                else:
                    if mesh:
                        xx, yy, pred = self._meshes[method][i]
                        plt.contourf(xx, yy, pred.T[0].reshape(xx.shape[0], xx.shape[0]), levels=10)
                    plt.scatter(
                        data[i][features[0]],
                        data[i][features[1]],
                        c=np.where(colors[i] > 0.5, '#c78f1e', '#0096f0'),
                        edgecolors='black'
                    )
                    plt.text(0.2, -0.2, self._dataset_name, ha='left', va='center')
                    plt.text(1, 1.15, f"Epoch {i}/{results['metadata']['iterations']}", ha='right', va='center')
                    plt.text(0, 1.15, method, ha='left', va='center')
                    plt.text(.8, -0.2, f"{results['metadata']['samples']} samples/epoch", ha='right', va='center')
                    plt.ylim(-0.1, 1.1)
                    plt.xlim(-0.1, 1.1)
                    f_name = f'{name[:-4]}_{n}.png'
                    plt.savefig(f_name)
                    plt.close()

                if i in [0, 9, 19, 29, 39, 49, 59] and False:
                    xx, yy, pred = self._meshes[method][i]
                    plt.contourf(xx, yy, pred.T[0].reshape(xx.shape[0], xx.shape[0]), levels=10)
                    plt.scatter(
                        data[i][features[0]],
                        data[i][features[1]],
                        c=np.where(colors[i] > 0.5, '#c78f1e', '#0096f0'),
                        edgecolors='black'
                    )
                    plt.ylim(-0.1, 1.1)
                    plt.xlim(-0.1, 1.1)
                    plt.xlabel('faeture 1')
                    plt.ylabel('feature 2')
                    plt.savefig(f'images/out_{self._iter_id}_{method}_{i}.png')
                    plt.close()

                out_names.append(f_name)

        gif_path = f"gifs/{self._iter_id}{f'_{self._out_count * (self._out_count > 0)}'}_{method}_gif.gif"

        with imageio.get_writer(f'{gif_path}', mode='I') as writer:
            for filename in out_names:
                image = imageio.v3.imread(filename)
                writer.append_data(image)

        self._logger.info(f"Saved gif to {gif_path}")

        self._out_count += 1

        for filename in list(out_names)[1:-1]:
            os.remove(filename)

    def save_gifs(self, **kwargs):
        """
        Uses generate_animation to save gifs depicting the recourse process.
        :return:
        """
        for name in self._methods:
            self.generate_animation(self.results, name, kwargs, None)

    def save_results(self, path=None):
        """
        Save the results Dict to a file.
        """
        if not path:
            path = f'results/{self._iter_id}.json'

        out = {}
        for i in self._methods:
            out[i] = {
                'means': np.array(self.results[i]['means'], dtype=float).tolist(),
                'covariances': np.array(self.results[i]['covariances'], dtype=float).tolist(),
                'clustering': np.array(self.results[i]['clustering'], dtype=int).tolist(),
                'accuracies': np.array(self.results[i]['accuracies'], dtype=float).tolist(),
                'f1_scores': np.array(self.results[i]['f1_scores'], dtype=float).tolist(),
                'pred_data': np.array(self.results[i]['pred_data'], dtype=float).tolist(),
                'mmd': np.array(self.results[i]['mmd'], dtype=float).tolist(),
                'disagreement': np.array(self.results[i]['disagreement'], dtype=float).tolist(),
                'model_mmd': np.array(self.results[i]['model_mmd'], dtype=float).tolist(),
                'prob_mmd': np.array(self.results[i]['prob_mmd'], dtype=float).tolist(),
            }
        out['metadata'] = self.results['metadata']

        with open(path, 'w+') as file:
            json.dump(out, file, indent=2)

        self._logger.info(f'Saved results to {path}')


if __name__ == "__main__":
    experiment = Experiment(
        generate_meshes=True,
        generators={
            'CLUE0': {
                'class': Clue.__name__,
                'hyperparameters': {
                    "data_name": "custom",
                    "train_vae": True,
                    "width": 10,
                    "depth": 3,
                    "latent_dim": 12,
                    "batch_size": 5,
                    "epochs": 3,
                    "lr": 0.001,
                    "early_stop": 20,
                }
            },
            'Wachter': {
                'class': Wachter.__name__,
                'hyperparameters': None,
            },
        },
        model={
            'model_type': 'ann',
            'hyperparameters': {"lr": 0.005, "epochs": 4, "batch_size": 20, "hidden_size": [10, 20, 10]}
        },
    )
    experiment.load_dataset(
        "custom",
        path='datasets/unimodal_dataset_1.csv', continuous=['feature1', 'feature2'], target='target'
    )
    experiment.run_experiment(iterations=20, samples=2)
    # experiment.save_gifs()
    experiment.save_results()
    experiment.save_gifs(type='pred_class', slow=5)
    # experiment.save_gifs(type='prob')
    # print(experiment.results)
