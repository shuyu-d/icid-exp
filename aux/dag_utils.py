import numpy as np
# import numpy.random as rnd
import random as rnd
import csv
import networkx as nx

def get_bunch_edges(W, is_directed=True):
    # Get the bunch of edges in the form of a list of 2-tuples
    if is_directed:
        Gb = nx.DiGraph()
        ignore_selfloop=False
    else:
        Gb = nx.Graph()
        ignore_selfloop=True
    if ignore_selfloop:
        W = W - np.diag(np.diag(W))
    edg1d = np.nonzero(W)
    eb = [(edg1d[0][i], edg1d[1][i]) for i in range(len(edg1d[0]))]
    Gb.add_edges_from(eb)
    return eb, Gb

def get_data_sachs(center=True, normalize=True, ndata=100):
    dim = 11
    if ndata < 0:
        ndata = None
    # rng_key_1, rng_key_2 = rnd.split(rnd.PRNGKey(random_seed))
    Xs = process_sachs(
           center=center, normalize=normalize,
           n_data=ndata)
    test_Xs = process_sachs(center=True, normalize=True) # #rng_key=rng_key_2)
    ground_truth_W = get_sachs_ground_truth()
    n_data = len(Xs)
    ground_truth_sigmas = np.ones(dim) * np.nan
    return Xs, ground_truth_W

def process_sachs(
    center: bool = True,
    print_labels: bool = False,
    normalize=False,
    n_data=None,
    rng_key=None
):
    data = []
    with open("./data/sachs_observational.csv") as csvfile:
        filereader = csv.reader(csvfile, delimiter=",")
        for i, row in enumerate(filereader):
            if i == 0:
                if print_labels:
                    print(row)
                continue
            data.append(np.array([float(x) for x in row]).reshape((1, -1)))
    if n_data is None:
        data_out = np.concatenate(data, axis=0)
    else:
        if rng_key is None:
            data_out = np.concatenate(data, axis=0)
            N = data_out.shape[0]
            sel = rnd.sample(range(N), n_data)
            data_out = data_out[sel]
        else:
            data_out = np.concatenate(data, axis=0)
            idxs = rnd.choice(rng_key, len(data_out), shape=(n_data,), replace=False)
            data_out = data_out[idxs]

    if center:
        if normalize:
            data_out = (data_out - np.mean(data_out, axis=0)) / np.std(data_out, axis=0)
        else:
            data_out = data_out - np.mean(data_out, axis=0)

    return data_out


def get_sachs_ground_truth():
    """Labels are ['praf', 'pmek', 'plcg', 'PIP2', 'PIP3', 'p44/42', 'pakts473',
    'PKA', 'PKC', 'P38', 'pjnk']."""
    W = np.load("./data/sachs_ground_truth.npy")
    return W
