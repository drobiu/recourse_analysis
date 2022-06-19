import numpy as np
from carla import Data, MLModelCatalog, MLModel
from kneed import KneeLocator

from pandas import DataFrame
from scipy.spatial.distance import pdist
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel

from recourse_analysis.util.predictions import predict


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

    # df_c = df_a.append(df_b)

    # distances = pdist(df_c, 'sqeuclidean')
    #
    # sigma = np.sqrt(np.median(distances))

    sigma = 0.15

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
        # shuffled = merged.sample(frac=1)
        shuffled = merged.iloc[np.random.permutation(len(merged))]
        len_shuffled = len(shuffled)
        half_a = shuffled.iloc[:int(len_shuffled / 2)]
        half_b = shuffled.iloc[int(len_shuffled / 2):]
        mmd_val = mmd_sklearn(half_a, half_b, target=target)
        if mmd_val >= target_mmd:
            ge += 1

    return ge / iterations


def find_elbow(dataset: Data, n: int = 10, chm=False):
    """
    Find the amount of clusters existing in the dataset using the Caliński-Harabasz
    elbow finding metric in KMeans clustering.
    :param dataset: Current dataset.
    :param n: Number of clusters to consider.
    :param chm: Use the Caliński-Harabasz metric.
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
        if chm:
            ch_metrics.append(metrics.calinski_harabasz_score(x, model.labels_))

    return KneeLocator(clusters, entropy, S=1.0, curve="convex", direction="decreasing").elbow


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


def boundary(data: Data, model: MLModel, sample: int = 1000, target_label: int = None):
    df = data.df
    if target_label is not None:
        df = df.loc[df[data.target] == target_label]
    samples = df.sample(min(len(df), sample))
    return np.mean([(p - 0.5) ** 2 for p in model.predict(samples)])


def compute_prob_model_shift(meshes):
    return mmd_sklearn(meshes[0], meshes[-1])
