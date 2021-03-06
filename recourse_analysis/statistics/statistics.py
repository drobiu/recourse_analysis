from typing import Dict

from carla import MLModelCatalog, Data
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from recourse_analysis.metrics.metrics import find_elbow, boundary
from recourse_analysis.util.predictions import predict


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
        'cf_pred_data': [],
        'mmd': [],
        'mmd_positive': [],
        'mmd_negative': [],
        'mmd_p_value': [],
        'disagreement': [],
        'model_mmd': [],
        'prob_mmd': [],
        'boundary_positive': [],
        'boundary_negative': [],
    }


def add_data_statistics(dataset: Data, results: Dict, model: MLModelCatalog = None):
    """
    Append the newest experiment statistics to the results Dict.
    :param model: Current MLModel.
    :param dataset: Current dataset object.
    :param results: Results Dict.
    """
    results['datasets'].append(dataset.df)
    results['means'].append(dataset.df[dataset.continuous].mean().to_numpy())
    results['covariances'].append(dataset.df[dataset.continuous].cov().to_numpy())
    results['clustering'].append(find_elbow(dataset))
    results['accuracies'].append(accuracy_score(np.array(dataset.df_test[dataset.target]), predict(model, dataset, True)))
    results['f1_scores'].append(f1_score(np.array(dataset.df_test[dataset.target]), predict(model, dataset, True)))
    results['probabilities'].append(model.predict(dataset.df).flatten())
    results['boundary_positive'].append(boundary(dataset, model, target_label=1))
    results['boundary_negative'].append(boundary(dataset, model, target_label=0))
