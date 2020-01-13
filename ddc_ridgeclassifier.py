"""
Script to recover double descent curve (ddc) through random features plus ridge classifier. Is it possible to choose
between real opu (to use it contact lighton at https://www.lighton.ai/contact-us/) and its synthetic version. The
available datasets within the script are mnist and cifar10; the input of the OPU must be binary; but don't worry we take
care of it.
"""


import logging
import pickle
import random

logging.basicConfig(level=logging.INFO)

import numpy as np
from sklearn.linear_model import RidgeClassifier

from ddc_utils import mnist, cifar10

try:
    from lightonopu.opu import OPU
except ImportError:
    pass


def synthetic(x, r1, r2):
    """
    perform the operation
    """
    o = np.dot(x, r1) ** 2 + np.dot(x, r2) ** 2  # real and complex part
    return o


def get_data(binary, dataset, encoding_method):
    if dataset == 'mnist':
        train, valid = mnist(batch_size=n_subset, n_train_samples=n_subset, binary=binary, encoder=encoding_method)
    else:
        train, valid = cifar10(batch_size=n_subset, n_train_samples=n_subset, binary=binary, encoder=encoding_method)

    x_train, y_train = iter(train).next()
    x_test, y_test = iter(valid).next()

    x_train, x_test = x_train.reshape(n_subset, -1).numpy(), x_test.reshape(x_test.shape[0], -1).numpy()
    y_train, y_test = y_train.numpy(), y_test.numpy()
    print('Check dimension: x_train.shape={}, x_test.shape={}'.format(x_train.shape, x_test.shape))
    return x_train, y_train, x_test, y_test


def fit_ridge(l2_reg, train_random_features, y_train, test_random_features, y_test):
    clf = RidgeClassifier(alpha=l2_reg)
    clf.fit(train_random_features, y_train.ravel())
    train_accuracy = clf.score(train_random_features, y_train)
    test_accuracy = clf.score(test_random_features, y_test)
    print("Train accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)
    return train_accuracy, test_accuracy


def synthetic_opu(n_subset, binary, rps, dataset, n_trials):
    x_train, y_train, x_test, y_test = get_data(binary, dataset)

    res_dic = {}
    for n_components in rps:

        logging.info("Running experiment for {} components".format(n_components))
        big_n_components = n_trials * n_components  # to run n_trials experiments at the same time
        dim = 28 * 28 if dataset == 'mnist' else 32 * 32 * 3
        R1 = np.random.randn(dim, big_n_components) / np.sqrt(n_components)
        R2 = np.random.randn(dim, big_n_components) / np.sqrt(n_components)
        big_train_random_features = synthetic(x_train, R1, R2)
        big_test_random_features = synthetic(x_test, R1, R2)

        l2_dic = {}
        for l2_reg in [0.]:
            train_accs = []
            test_accs = []
            for k in range(n_trials):
                train_random_features = np.copy(big_train_random_features[:, k * n_components:(k + 1) * n_components])
                test_random_features = np.copy(big_test_random_features[:, k * n_components:(k + 1) * n_components])

                train_accuracy, test_accuracy = fit_ridge(l2_reg, train_random_features, y_train, test_random_features,
                                                          y_test)
                train_accs.append(train_accuracy)
                test_accs.append(test_accuracy)
            l2_dic[l2_reg] = {'train': train_accs, 'test': test_accs}
        res_dic[n_components] = l2_dic

    with open('synthopu_{}ksamples_5trials.pkl'.format(n_subset / 1e3), 'wb') as f:
        pickle.dump(res_dic, f)


def real_opu(n_subset, binary, rps, dataset, encoding_method, n_trials):
    x_train, y_train, x_test, y_test = get_data(binary, dataset, encoding_method)

    # OPU -----------------------------------------------------------
    max_rps = max(rps)
    r_opu = OPU(n_components=max_rps * n_trials, verbose_level=1)
    r_opu.open()
    big_train_random_features = r_opu.transform1d(x_train)
    big_test_random_features = r_opu.transform1d(x_test)
    r_opu.close()
    # ----------------------------------------------------------------

    res_dic = {}
    for n_components in rps:

        logging.info("Running experiment for {} components".format(n_components))

        l2_dic = {}
        for l2_reg in [0.]:
            train_accs = []
            test_accs = []
            for k in range(n_trials):
                a = random.randint(0, max_rps - n_components)
                train_random_features = np.copy(big_train_random_features[:, a:a + n_components])
                test_random_features = np.copy(big_test_random_features[:, a:a + n_components])

                train_accuracy, test_accuracy = fit_ridge(l2_reg, train_random_features, y_train, test_random_features,
                                                          y_test)

                train_accs.append(train_accuracy)
                test_accs.append(test_accuracy)
            l2_dic[l2_reg] = {'train': train_accs, 'test': test_accs}
        res_dic[n_components] = l2_dic

    with open('opu_{}ksamples_5trials.pkl'.format(n_subset / 1e3), 'wb') as f:
        pickle.dump(res_dic, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('parameters')

    parser.add_argument("-is_real_opu", default=False, help="to choose if using real or synthetic opu", type=bool)
    parser.add_argument("-n_subset", default=10000, help="training subset of the chosen dataset", type=int)
    parser.add_argument("-dataset", default='mnist', help="possibe choices: mnist, cifar10", type=str)
    parser.add_argument("-encoding_method", default='threshold', help="threshold or autoencoder", type=str)
    parser.add_argument("-n_trials", default=5, help="# of trials for each random features value", type=int)
    parser.add_argument("-rps", default=sorted([500 * k for k in range(1, 41)] + [9250, 9750, 10250, 10750]), type=list)

    args = parser.parse_args()
    n_subset = args.n_subset
    is_real_opu = args.is_real_opu
    dataset = args.dataset
    encoding_method = args.encoding_method
    rps = args.rps
    n_trials = args.n_trials

    if dataset not in {'mnist', 'cifar10'}:
        raise ValueError("Available datasets are 'mnist', 'cifar10'")

    if is_real_opu:
        real_opu(n_subset, binary=True, rps=rps, dataset=dataset, encoding_method=encoding_method, n_trials=n_trials)
    else:
        synthetic_opu(n_subset, binary=False, rps=rps, dataset=dataset, n_trials=n_trials)
