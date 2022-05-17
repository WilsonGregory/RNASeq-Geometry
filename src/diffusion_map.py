import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import time

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

import torch
from torch.utils.data import DataLoader

from pydiffmap import diffusion_map

def parse_adata(adata):
    """
    Split a scanpy adata into X and y, where y is adata.obs['annotation'] converted to numbers using LabelEncoder
    Args:
        adata (scanpy adata): The result of something like sc.read_h5ad(), must have .obs['annotation']
    returns: tuple (X, y, encoder), X data, y as the encoded labels, and the encoder
    """
    X = adata.X.copy()
    labels = adata.obs['annotation'].values
    encoder = LabelEncoder()
    encoder.fit(labels)
    y = encoder.transform(labels)
    return X, y, encoder

def split_data_into_dataloaders(X, y, train_size, val_size, batch_size = 64, num_workers = 0, seed = None):
    """
    Split X and Y into training set (fraction train_size), validation set (fraction val_size)
    and the rest into a test set. train_size + val_size must be less than 1.
    Args:
        X (array): Input data
        y (vector): Output labels
        train_size (float): 0 to 1, fraction of data for train set
        val_size (float): 0 to 1, fraction of data for validation set
        batch_size (int): defaults to 64
        num_workers (int): number of cores for multi-threading, defaults to 0 for no multi-threading
        seed (int): defaults to none, set to reproduce experiments with same train/val split
    """
    assert train_size + val_size < 1
    assert len(X) == len(y)
    assert batch_size > 1

    if seed is not None:
        np.random.seed(seed)

    test_size = 1 - train_size - val_size

    slices = np.random.permutation(np.arange(X.shape[0]))
    train_end = int(train_size* len(X))
    val_end = int((train_size + val_size)*len(X))

    train_indices = slices[:train_end]
    val_indices = slices[train_end:val_end]
    test_indices = slices[val_end:]

    train_x = X[train_indices, :]
    val_x = X[val_indices, :]
    test_x = X[test_indices, :]

    train_y = y[train_indices]
    val_y = y[val_indices]
    test_y = y[test_indices]

    train_x = torch.Tensor(train_x)
    val_x = torch.Tensor(val_x)
    test_x = torch.Tensor(test_x)

    train_y = torch.LongTensor(train_y)
    val_y = torch.LongTensor(val_y)
    test_y = torch.LongTensor(test_y)

    train_dataloader = DataLoader(torch.utils.data.TensorDataset(train_x, train_y), batch_size=batch_size, shuffle = True, num_workers = num_workers)
    val_dataloader = DataLoader(torch.utils.data.TensorDataset(val_x, val_y), batch_size=batch_size, shuffle = False, num_workers = num_workers)
    test_dataloader = DataLoader(torch.utils.data.TensorDataset(test_x, test_y), batch_size=batch_size, shuffle = False, num_workers = num_workers)

    return train_dataloader, val_dataloader, test_dataloader, train_indices, val_indices, test_indices

def new_model_metrics(train_x, train_y, test_x, test_y, markers = None, model = None):
    """
    Trains and tests a specified model (or RandomForest, if none specified) with a subset of the dimensions
    specified by the indices in the markers array. Returns the error rate, a testing report, and a confusion
    matrix of the results.
    Args:
        train_x: (numpy array) the training data input
        train_y: (numpy array) the training data labels
        test_x: (numpy array) testing data input
        test_y: (numpy array) testing data labels
        markers: (numpy array) marker indices, a subset of the column indices of train_x/test_x, defaults to all
        model: model to train and test on, defaults to RandomForest
    """
    if markers is not None:
        train_x = train_x[:, markers]
        test_x = test_x[:, markers]

    if model is None:
        model = RandomForestClassifier()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    train_rep = classification_report(train_y, model.predict(train_x), output_dict=True)
    test_rep = classification_report(test_y, pred_y, output_dict=True)
    cm = confusion_matrix(test_y, pred_y)
    if cm is None:
        raise Exception("Some error in generating confusion matrix")
    misclass_rate = 1 - accuracy_score(test_y, pred_y)
    return misclass_rate, test_rep, cm

def getSwissRoll():
    # set parameters
    length_phi = 15   #length of swiss roll in angular direction
    length_Z = 15     #length of swiss roll in z direction
    sigma = 0.1       #noise strength
    m = 10000         #number of samples

    # create dataset
    phi = length_phi*np.random.rand(m)
    xi = np.random.rand(m)
    Z = length_Z*np.random.rand(m)
    X = 1./6*(phi + sigma*xi)*np.sin(phi)
    Y = 1./6*(phi + sigma*xi)*np.cos(phi)

    swiss_roll = np.array([X, Y, Z]).transpose()

    # check that we have the right shape
    print(swiss_roll.shape)

    # initialize Diffusion map object.
    neighbor_params = {'n_jobs': -1, 'algorithm': 'ball_tree'}
    n_evecs = 4

    mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs=n_evecs, k=100, epsilon=0.0156, alpha=1.0, neighbor_params=neighbor_params)
    # fit to data and return the diffusion map.
    mydmap.fit(swiss_roll)

    return swiss_roll, mydmap.dmap[:,0], 0.0156, 100


def plot2d(X, n_cols, skip_eigenvecs = {}, colors = None, show=True):
    assert n_cols <= X.shape[1]

    for i in range(n_cols):
        if i in skip_eigenvecs:
            continue

        for j in range(i+1, n_cols):
            if j in skip_eigenvecs:
                continue

            fig1, ax1 = plt.subplots()
            if colors is not None:
                ax1.scatter(X[:,i], X[:,j], c=colors, cmap='Spectral')
            else:
                ax1.scatter(X[:,i], X[:,j])

            # used for finding the eigenvalue outliers, prob a better way of doing this?
            # for k in range(X.shape[0]):
            #     ax1.annotate(k, (X[k,i], X[k,j]))

            ax1.set_xlabel(f'Eigenvec {i}')
            ax1.set_ylabel(f'Eigenvec {j}')

            # fig1.savefig(f'/Users/wilson/Documents/jhu/spring_2022/highD_approx/final_project/citeseq_evec_pairs/citeseq_evecs_{i}_{j}')

    plt.tight_layout()

    if show:
        plt.show()


def plot3d(X, n_cols, skip_eigenvecs = {}, colors = None, show=True):
    assert n_cols <= X.shape[1]

    for i in range(n_cols):
        if i in skip_eigenvecs:
            continue

        for j in range(i+1, n_cols):
            if j in skip_eigenvecs:
                continue

            for k in range(j+1, n_cols):
                if k in skip_eigenvecs:
                    continue

                fig1, ax1 = plt.subplots(subplot_kw={'projection':'3d'})
                if colors is not None:
                    ax1.scatter(X[:,i], X[:,j], X[:,k], c=colors, cmap='Spectral')
                else:
                    ax1.scatter(X[:,i], X[:,j], X[:,k])

                ax1.set_xlabel(f'Eigenvec {i}')
                ax1.set_ylabel(f'Eigenvec {j}')
                ax1.set_zlabel(f'Eigenvec {k}')

    plt.tight_layout()

    if show:
        plt.show()


def getZeiselData():
    adata = sc.read_h5ad('data/zeisel/Zeisel.h5ad')
    adata.obs['annotation'] = adata.obs['names0']
    X, y, encoder = parse_adata(adata)
    print(X.shape)

    # epsilon = 256 # this is what BGH finds
    epsilon = X.shape[1]/5
    k = 10

    return X, y, epsilon, k

def getCiteSeqData():
    adata = sc.read_h5ad('data/cite_seq/CITEseq.h5ad')
    adata.obs['annotation'] = adata.obs['names']
    X, y, encoder = parse_adata(adata)
    print(X.shape)

    # remove some outliers
    outliers = [1066, 779, 8458]
    X = np.delete(X, outliers, axis=0)
    y = np.delete(y, outliers)

    # epsilon = 64 # given by BGH
    epsilon = X.shape[1]/5
    k = 100

    return X, y, epsilon, k


def getSkreePlots(X, y, epsilon, k, n_evecs):
    mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=1, k=k)
    mydmap.fit(X)

    pca = PCA(n_components=np.min([n_evecs, X.shape[1]]))
    X_pca = pca.fit_transform(X)

    diff_map_evals = np.sqrt(-1. / mydmap.evals)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(n_evecs), diff_map_evals[:n_evecs], marker='o')
    ax1.set_title('Diffusion Map Eigenvalues')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')

    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(pca.explained_variance_)), pca.explained_variance_, marker='o')
    ax2.set_title('PCA Eigenvalue')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue')
    plt.show()

def evaluateDiffMapClassification(X, y, epsilon, k, eval_models, num_times=1):
    n_evecs = len(np.unique(y))

    print(X.shape)
    print(y.shape)
    print(epsilon)
    print(k)
    print(n_evecs)

    mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=1, k=k)
    mydmap.fit(X)

    print('here')

    pca = PCA(n_components=n_evecs)
    X_pca = pca.fit_transform(X)

    results = { key: np.zeros((3,num_times)) for key in eval_models.keys() }
    for i in range(num_times):
        _, _, _, train_indices, _, test_indices = split_data_into_dataloaders(X,y, 0.8, 0) # train 80%, val 0%, test 20%
        print(i)

        for model_label, eval_model in eval_models.items():

            # Diffusion map eval
            misclass_rate, _, _ = new_model_metrics(
                mydmap.dmap[train_indices, :],
                y[train_indices],
                mydmap.dmap[test_indices, :],
                y[test_indices],
                model = eval_model,
            )
            results[model_label][0,i] = misclass_rate

            # All original features baseline
            baseline_misclass, _, _ = new_model_metrics(
                X[train_indices, :],
                y[train_indices],
                X[test_indices, :],
                y[test_indices],
                model = eval_model,
            )
            results[model_label][1,i] = baseline_misclass

            # PCA baseline
            pca_misclass, _, _ = new_model_metrics(
                X_pca[train_indices, :],
                y[train_indices],
                X_pca[test_indices, :],
                y[test_indices],
                model = eval_model,
            )
            results[model_label][2,i] = pca_misclass

    for k,v in results.items():
        print(k, np.mean(results[k], axis=1))


def runDiffMapAndPlotPairs(X, y, epsilon, k, max_plots=7):
    n_evecs = len(np.unique(y))

    mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=1, k=k)
    mydmap.fit(X)

    # plot the diffusion embeddings
    plot2d(mydmap.dmap, n_cols=np.min([n_evecs, max_plots]), colors=y, show=True)


def findDiffMapParam(X, y, epsilon, k, param_range, param, num_times=1, display_time=False):
    # test for different values of k
    n_evecs = len(np.unique(y))

    runtime = []
    misclass_rates = []

    #First, train all dmaps over the parameter range
    dmaps = {}
    for param_value in param_range:
        print(f'{param}: {param_value}')

        if param == 'epsilon':
            mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=param_value, alpha=1, k=k)
        elif param == 'k':
            mydmap = diffusion_map.DiffusionMap.from_sklearn(n_evecs = n_evecs, epsilon=epsilon, alpha=1, k=param_value)

        start_time = time.time()
        mydmap.fit(X)
        end_time = time.time() - start_time
        runtime.append(end_time)
        dmaps[param_value] = mydmap

    # PCA baseline
    pca = PCA(n_components=n_evecs)
    X_pca = pca.fit_transform(X)

    baseline_rates = []
    pca_rates = []
    diff_map_rates = np.zeros((num_times, len(param_range)))
    for i in range(num_times):
        print(f'run: {i}')
        _, _, _, train_indices, _, test_indices = split_data_into_dataloaders(X,y, 0.8, 0) # train 80%, val 0%, test 20%

        for idx, param_value in enumerate(param_range):
            misclass_rate, _, _ = new_model_metrics(
                dmaps[param_value].dmap[train_indices, :],
                y[train_indices],
                dmaps[param_value].dmap[test_indices, :],
                y[test_indices],
            )
            diff_map_rates[i,idx] = misclass_rate

        baseline_misclass, _, _ = new_model_metrics(X[train_indices, :], y[train_indices], X[test_indices, :], y[test_indices])
        baseline_rates.append(baseline_misclass)

        pca_misclass, _, _ = new_model_metrics(
            X_pca[train_indices, :],
            y[train_indices],
            X_pca[test_indices, :],
            y[test_indices],
        )
        pca_rates.append(pca_misclass)

    print(diff_map_rates.shape)
    print(diff_map_rates)

    fig1, ax1 = plt.subplots()
    ax1.plot(param_range, np.mean(diff_map_rates, axis=0), label='diff_map', marker='o')
    ax1.hlines(np.mean(baseline_rates), param_range[0], param_range[-1], label='baseline', colors=['red'])
    ax1.hlines(np.mean(pca_rates), param_range[0], param_range[-1], label='pca', colors=['orange'])
    ax1.set_xlabel(param)
    ax1.set_ylabel('Misclass Rate')
    ax1.legend()

    if display_time:
        fig2, ax2 = plt.subplots()
        ax2.plot(param_range, runtime, label='runtime', marker='o')
        ax2.set_xlabel('Epsilon')
        ax2.set_ylabel('Time')
        ax2.legend()

    plt.tight_layout()
    plt.show()


# MAIN
num_times = 10

# Zeisel
# k_range = [5,10,20,30,50,100,200,500,1000,2000,3000]
# epsilon_range = [128,200,400,800,1000,1500,2000,4000]
# X, y, epsilon, k = getZeiselData()

# # Cite-Seq
k_range = [5,10,20,30,50,100,200,500,1000,3000,5000,8000]
epsilon_range = [32,64,80,100,150,200,300,500,750,1000]
X, y, epsilon, k = getCiteSeqData()

# Swiss Roll
# X, y, epsilon, k = getSwissRoll()

eval_models = {
    'knn': KNeighborsClassifier(),
    'rf': RandomForestClassifier(),
    'neural network': MLPClassifier(max_iter=1000),
}

# runDiffMapAndPlotPairs(X, y, epsilon, k, max_plots=7)

evaluateDiffMapClassification(X, y, epsilon, k, num_times=1, eval_models=eval_models)

# findDiffMapParam(X, y, epsilon, k, param='k', param_range=k_range, num_times=num_times, display_time=False)
# findDiffMapParam(X, y, epsilon, k, param='epsilon', param_range=epsilon_range, num_times=num_times, display_time=False)

# getSkreePlots(X, y, epsilon, k, n_evecs=100)


