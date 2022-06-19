import json
import random
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

import torch
from carla.data.catalog import CsvCatalog, OnlineCatalog
from carla import MLModelCatalog, Data, MLModel
from carla.recourse_methods import Clue, Wachter
from carla.models.negative_instances import predict_negative_instances
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from recourse_analysis.metrics import mmd_sklearn, mmd_p_value, disagreement, compute_prob_model_shift
from recourse_analysis.util import CustomBenchmark, RecourseMethodData, update_dataset, get_timestamp, \
    train_model, disable, enable, train_recourse_method
from recourse_analysis.statistics import get_empty_results, add_data_statistics

warnings.filterwarnings("ignore")


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
        self._logger = carla.get_logger(Experiment.__name__)
        self._title = kwargs.pop('title', None)
        self._seed = kwargs.pop('seed', np.random.get_state()[1][0])
        np.random.seed(self._seed)
        random.seed(self._seed)
        torch.manual_seed(self._seed)
        torch.set_deterministic(True)
        self._out_count = 0
        self._options = {
            'generate_meshes': kwargs.get('generate_meshes', True)
        }

        self._features = []
        self._dataset_name = None
        self._dataset_path = None
        self._dataset = None
        self._first_model = None
        self._used_factuals_indices = set()

        self.results = {}

        self._model_options = kwargs.pop('model', False)
        self._generator_options = kwargs.pop('generators', False)

        print(self._generator_options.items())

        if self._generator_options:
            self._methods = {name: RecourseMethodData(name, obj['class'], obj['hyperparameters'])
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
        path = kwargs.pop('path')
        if name == 'custom':
            dataset = CsvCatalog(
                file_path=path,
                categorical=kwargs.pop('categorical', []),
                continuous=kwargs.pop('continuous', []),
                immutables=kwargs.pop('immutables', []),
                target=kwargs.pop('target')
            )
        else:
            dataset = OnlineCatalog(name)
            dataset._df = dataset.df.sample(40000)
            # stratify the test set
            x_train, x_test, y_train, y_test = train_test_split(
                dataset.df.drop(dataset.target, axis=1),
                dataset.df[dataset.target],
                test_size=0.16,
                stratify=dataset.df[dataset.target]
            )
            x_train[dataset.target] = y_train
            x_test[dataset.target] = y_test
            dataset._df = pd.concat([x_train, x_test])
            dataset._df_train = x_train
            dataset._df_test = x_test

        self._dataset_name = name
        self._dataset_path = path
        self._dataset = dataset
        self._features = dataset.df.drop(self._dataset.target, axis=1).columns.values.tolist()

    def run_experiment(self, save_output=False, iterations=35, samples=2):
        """
        Runs the experiment using the CLUE and Wachter recourse generators on a set amount of
        epochs and a set amount of counterfactuals per epoch. The experiment uses the dataset
        set by the load_dataset method.
        """
        if not self._dataset:
            self._logger.error("Load a dataset before running experiments.")
            return

        self._first_model = train_model(self._dataset, training_params=self._model_options)
        self._first_model = train_model(self._dataset, training_params=self._model_options, model=self._first_model)

        for method in self._methods:
            self._methods[method].dataset = deepcopy(self._dataset)
            # self._methods[method].model = deepcopy(self._first_model)
            self.results[method] = get_empty_results()

        self.results['metadata'] = {
            'iterations': iterations,
            'samples': samples,
            'generators': self._generator_options,
            'model': self._model_options,
            'dataset': self._dataset_path,
            'np_seed': int(self._seed),
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

        results['cf_pred_data'].extend(model.predict(counterfactuals.dropna()).flatten())

        results['mmd'].append(mmd_sklearn(self._dataset.df.sample(frac=1), dataset.df, self._dataset.target))

        results['disagreement'].append(disagreement(self._first_model, model, self._dataset))

        a = timeit.default_timer()
        results['model_mmd'].append(compute_prob_model_shift(self._low_res_meshes[method]))
        b = timeit.default_timer()
        print(b - a)

        add_data_statistics(dataset, results, model)

        results['prob_mmd'].append(mmd_sklearn(DataFrame(results['probabilities'][0]),
                                               DataFrame(results['probabilities'][-1]), ''))

        update_dataset(dataset, factuals, counterfactuals)

        return dataset

    def _update_last_epoch(self, dataset, model, method, samples):
        model, factuals = self.get_factuals(dataset, model, sample_num=samples)

        self.results[method]['mmd'].append(mmd_sklearn(self._dataset.df, dataset.df, self._dataset.target))

        self.results[method]['disagreement'].append(disagreement(self._first_model, model, self._dataset))

        self.results[method]['model_mmd'].append(compute_prob_model_shift(self._low_res_meshes[method]))

        self.results[method]['mmd_p_value'].append(
            mmd_p_value(self._dataset.df, dataset.df, self.results[method]['mmd'][-1], self._dataset.target))

        add_data_statistics(dataset, self.results[method], model)

        self.results[method]['prob_mmd'].append(mmd_sklearn(DataFrame(self.results[method]['probabilities'][0]),
                                                            DataFrame(self.results[method]['probabilities'][-1]), ''))

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
        # disable()
        model = train_model(dataset, model, self._model_options)
        # Predict factuals
        factuals = predict_negative_instances(model, dataset.df)
        n_factuals = len(factuals)
        # If not enough factuals generated and the max amount of
        # iterations not reached, retrain model and try again
        # enable()
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

            self._meshes[m].append((xx, yy, pred))

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

            pred = models[m].predict(pred_df)
            pred_df['pred'] = pred

            self._low_res_meshes[m].append(pred_df)

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

        names = [f"../images/{method}{str(n)}_{self._iter_id}.png" for n in range(len(data))]
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
                    # plt.text(.8, -0.2, f"{results['metadata']['samples']} samples/epoch", ha='right', va='center')
                    plt.ylim(-0.1, 1.1)
                    plt.xlim(-0.1, 1.1)
                    f_name = f'{name[:-4]}_{n}.png'
                    plt.savefig(f_name)
                    plt.close()

                # if i in [0, 19, 39, 59, 79, 99] and False:
                #     xx, yy, pred = self._meshes[method][i]
                #     plt.contourf(xx, yy, pred.T[0].reshape(xx.shape[0], xx.shape[0]), levels=10)
                #     plt.scatter(
                #         data[i]["NumberOfTimes90DaysLate"],
                #         data[i]["MonthlyIncome"],
                #         c=np.where(colors[i] > 0.5, '#c78f1e', '#0096f0'),
                #         edgecolors='black'
                #     )
                #     plt.ylim(-0.1, 1.1)
                #     plt.xlim(-0.1, 1.1)
                #     plt.xlabel('feature 1')
                #     plt.ylabel('feature 2')
                #     plt.savefig(f'images/out_{self._iter_id}_{method}_{i}.png')
                #     plt.close()

                out_names.append(f_name)

        gif_path = f"../gifs/{self._iter_id}{f'_{self._out_count * (self._out_count > 0)}'}_{method}_gif.gif"

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
            path = f'../results/{self._iter_id}{"_" + self._title if self._title else ""}.json'

        out = {}
        for i in self._methods:
            out[i] = {
                'means': np.array(self.results[i]['means'], dtype=float).tolist(),
                'covariances': np.array(self.results[i]['covariances'], dtype=float).tolist(),
                'clustering': np.array(self.results[i]['clustering'], dtype=int).tolist(),
                'accuracies': np.array(self.results[i]['accuracies'], dtype=float).tolist(),
                'f1_scores': np.array(self.results[i]['f1_scores'], dtype=float).tolist(),
                'pred_data': np.array(self.results[i]['pred_data'], dtype=float).tolist(),
                'cf_pred_data': np.array(self.results[i]['cf_pred_data'], dtype=float).tolist(),
                'mmd': np.array(self.results[i]['mmd'], dtype=float).tolist(),
                'disagreement': np.array(self.results[i]['disagreement'], dtype=float).tolist(),
                'model_mmd': np.array(self.results[i]['model_mmd'], dtype=float).tolist(),
                'prob_mmd': np.array(self.results[i]['prob_mmd'], dtype=float).tolist(),
                'boundary_positive': np.array(self.results[i]['boundary_positive'], dtype=float).tolist(),
                'boundary_negative': np.array(self.results[i]['boundary_negative'], dtype=float).tolist(),
                'mmd_p_value': np.array(self.results[i]['mmd_p_value'], dtype=float).tolist(),
                'benchmark': self.results[i]['benchmark'],
            }
        out['metadata'] = self.results['metadata']

        with open(path, 'w+') as file:
            json.dump(out, file, indent=2)

        self._logger.info(f'Saved results to {path}')


if __name__ == "__main__":
    title = 'separable_clue_hyperparameters'
    generators = {
                'CLUE4': {
                    'class': Clue.__name__,
                    'hyperparameters': {
                        "data_name": "custom",
                        "train_vae": True,
                        "width": 10,
                        "depth": 3,
                        "latent_dim": 1,
                        "batch_size": 5,
                        "epochs": 2,
                        "lr": 0.001,
                        "early_stop": 10,
                    }
                },
                'CLUE5': {
                    'class': Clue.__name__,
                    'hyperparameters': {
                        "data_name": "custom",
                        "train_vae": True,
                        "width": 10,
                        "depth": 3,
                        "latent_dim": 3,
                        "batch_size": 5,
                        "epochs": 2,
                        "lr": 0.001,
                        "early_stop": 10,
                    }
                }
            }

    model = {
        'model_type': 'ann',
        'hyperparameters': {"lr": 0.005, "epochs": 4, "batch_size": 2, "hidden_size": [10, 10]}
    }

    metadata = json.load(open('../datasets/give_me_some_credit_balanced/metadata.json'))

    for i in range(5):
        experiment = Experiment(
            title=f'{title}_{i}',
            generate_meshes=False,
            generators=generators,
            model=model,
            seed=1
        )
        experiment.load_dataset(
            "custom",
            # path=f'../datasets/give_me_some_credit_balanced/{metadata["filename"]}',
            # continuous=metadata['continuous'], categorical=metadata['categorical'],
            # immutables=metadata['immutables'], target=metadata['target']
            path=f'../datasets/plus_shaped.csv',
            continuous=['feature1', 'feature2'], categorical=[],
            immutables=[], target='target'
        )

        experiment.run_experiment(iterations=10, samples=10)
        # experiment.save_gifs()
        experiment.save_results()
        # experiment.save_gifs(type='pred_class')