from functools import partial

import numpy as np
import polars as pl

from pypuf.simulation import ArbiterPUF

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import umap

import optuna

from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from scipy.optimize import curve_fit

challenges = np.load("./data/challenges/ergodicity_challenges.npy")
challenges_polar = (-1)**challenges.astype(np.int8)

challenges_phi = ArbiterPUF.transform_atf(challenges_polar, k=1).reshape(len(challenges), -1)


def objective_logreg(trial: optuna.Trial, X, y):
    C = trial.suggest_float("C", 1e-4, 1e3, log=True)
    clf = LogisticRegression(
        C=C,
        l1_ratio=1.0,  # Pure L1
        solver="liblinear",
        max_iter=5000,
        tol=1e-3,
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True)
    scores = cross_val_score(clf, X, y, cv=skf)
    return np.mean(scores)

df_dev = df_summary_ergo.filter(pl.col('device_id') == '210376BCA71E')
mean_response = df_dev['mean_response'].to_numpy()

ids_train, ids_test = train_test_split(np.arange(len(df_dev)), train_size=0.6)

fn = partial(objective_logreg, X=challenges_phi_ergo[ids_train], y=np.where(mean_response[ids_train] >= 0.5, 1, 0))
study = optuna.create_study(direction="maximize")
study.optimize(fn, n_trials=10, n_jobs=1, show_progress_bar=False)

clf = LogisticRegression(C=study.best_params['C'],l1_ratio=1.0,solver="liblinear",max_iter=5000,tol=1e-3)
clf.fit(challenges_phi_ergo[ids_test], np.where(mean_response[ids_test] >= 0.5, 1, 0))

# w is the weight vector of shape (n_stages,)
w = clf.coef_[0]
# The product of w and \Phi(challenge) produces the delay accumulated at each stage. If we sum per row, we obtain the accumulated delay
delta_t = w * challenges_phi_ergo[ids_test]

fig, ax = plt.subplots(figsize=(8, 5))

def sigmoid(x, a, b, c):
    return c / (1.0 + np.exp(-a*(x-b)))

sum_delay = np.sum(delta_t, axis=1)
mean_response_test = mean_response[ids_test]

popt, pcov = curve_fit(sigmoid, sum_delay, mean_response_test, method='dogbox', bounds=([0, -30, 0],[1, 30, 50]))
print(f'Parameteres of the sigmoid: a={popt[0]} b={popt[1]} c={popt[2]}')

ax.scatter(np.sum(delta_t, axis=1), mean_response[ids_test], s=2, label='Empirical Data')

x_sigmoid = np.linspace(-30, 30, 50)
ax.plot(x_sigmoid, sigmoid(x_sigmoid, *popt), color='orange', alpha=0.5, linewidth=2, label="Fitted Sigmoid")

ax.set_ylabel(r"Mean Response")
ax.set_xlabel(r"$\sum w\cdot\Phi_{test}$")

ax.legend()
fig.tight_layout()
plt.show()
fig.savefig("figures/apuf_128_delay_projection.png")

## Analysis by slices

def compute_w(df_slice, challenges_phi, n_trials=5):
    mean_response = df_slice['mean_response'].to_numpy()

    ids_train, ids_test = train_test_split(np.arange(len(df_slice)), train_size=0.8)
    # ids_test_slices[slice_loc] = ids_test
    
    fn = partial(objective_logreg, X=challenges_phi[ids_train], y=np.where(mean_response[ids_train] >= 0.5, 1, 0))
    study = optuna.create_study(direction="maximize")
    study.optimize(fn, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    clf = LogisticRegression(C=study.best_params['C'],l1_ratio=1.0,solver="liblinear",max_iter=5000,tol=1e-3)
    clf.fit(challenges_phi[ids_train], np.where(mean_response[ids_train] >= 0.5, 1, 0))

    # w is the weight vector of shape (n_stages,) 
    # w_slices[slice_loc] = clf.coef_[0]
    # The product of w and \Phi(challenge) produces the delay accumulated at each stage. If we sum per row, we obtain the accumulated delay
    # delta_t_slices[slice_loc] = w * challenges_phi[ids_test]
    return clf.coef_[0], ids_test


w_slices = {}
ids_test_slices = {}
for slice_id, slice_loc in enumerate(slices):
    print(f"Performing LogReg Analysis on {slice_loc}")
    df_slice = df_summary.filter(pl.col('slice') == slice_loc)
    w_slices[slice_loc], ids_test_slices[slice_loc] = compute_w(df_slice, challenges_phi_ergo)


fig, ax = plt.subplots(figsize=(12, 6))
colors = dict(zip(slices, ['orange', 'blue', 'darkgreen', 'purple']))

x_sigmoid = np.linspace(-50, 50, 1000)
# def sigmoid(x, L, k, x0):
#     return L / (1.0 + np.exp(-k*(x-x0)))

def sigmoid(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k*(x-x0)))

for slice_id, slice_loc in enumerate(slices):
    df_slice = df_summary.filter(pl.col('slice') == slice_loc)
    mean_response = df_slice['mean_response'].to_numpy()
    
    ids_test = ids_test_slices[slice_loc]
    mean_response_test = mean_response[ids_test]
    w = w_slices[slice_loc]
    
    delta_t = w * challenges_phi_ergo[ids_test]
    sum_delay = np.sum(delta_t, axis=1)

    # popt, pcov = curve_fit(sigmoid, sum_delay, mean_response_test, method='dogbox', bounds=([1.0, -20, -20],[1.001, 20, 20]))
    popt, pcov = curve_fit(sigmoid, sum_delay, mean_response_test, bounds=([-10, -10],[10, 10]))
    # popt, pcov = curve_fit(sigmoid, sum_delay, mean_response_test)
    print(f'[{slice_loc:<12}] k={popt[0]} x0={popt[1]}')
    print(pcov)
    ax.plot(x_sigmoid, sigmoid(x_sigmoid, *popt), color=colors[slice_loc], alpha=1.0, linestyle="--", linewidth=1.2, label=f"Fitted {slice_loc}")

    ax.scatter(sum_delay, mean_response_test, color=colors[slice_loc], alpha=0.3, s=2.5, label=f'Empirical {slice_loc}')


ax.set_xlabel(r"$\sum w\cdot\Phi_{test}$")
ax.set_ylabel(r"Mean Response")
ax.legend(loc="lower right", bbox_to_anchor=(0.95, 0.15), fontsize=7)
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.4)
ax.axvline(color="black", linestyle="--", alpha=0.4)
ax.set(xlim=(-10, 10))

fig.tight_layout()
plt.show()
fig.savefig("figures/apuf_64_delay_projection_all_slices.png")


## Delay projection on multiple devices

df_summary_all = pl.read_csv("./data/ergodicity/crps_summary_merged.csv").filter(pl.col('stages') == 64)
devices = df_summary_all['device_id'].unique().to_list()
slices = df_summary_all['slice'].unique().to_list()

w_slices = {}
ids_test_slices = {}

for dev in devices:
    df_device = df_summary_all.filter(pl.col('device_id') == dev)
    w_slices[dev] = {}
    ids_test_slices[dev] = {}
    
    for slice_id, slice_loc in enumerate(slices):
        print(f"[{dev}] {slice_loc}")
        df_slice = df_device.filter(pl.col('slice') == slice_loc)
        w, ids_test = compute_w(df_slice)
        w_slices[dev][slice_loc] = w
        ids_test_slices[dev][slice_loc] = ids_test

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
colors = dict(zip(slices, ['orange', 'blue', 'darkgreen', 'purple']))

x_sigmoid = np.linspace(-50, 50, 1000)

def sigmoid(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k*(x-x0)))

mapping = {
    'LEFT-UPPER': (0, 0),
    'LEFT-LOWER': (1, 0), 
    'RIGHT-LOWER': (1, 1),
    'RIGHT-UPPER': (0, 1),
}

parameters = []
for dev in devices:
    for slice_id, slice_loc in enumerate(slices):
        ax = axs[mapping[slice_loc]]
        df_slice = df_summary_all.filter(pl.col('device_id') == dev, pl.col('slice') == slice_loc)
        mean_response = df_slice['mean_response'].to_numpy()
    
        ids_test = ids_test_slices[dev][slice_loc]
        mean_response_test = mean_response[ids_test]
        w = w_slices[dev][slice_loc]
    
        delta_t = w * challenges_phi_ergo[ids_test]
        sum_delay = np.sum(delta_t, axis=1)

        # popt, pcov = curve_fit(sigmoid, sum_delay, mean_response_test, method='dogbox', bounds=([1.0, -20, -20],[1.001, 20, 20]))
        popt, pcov = curve_fit(sigmoid, sum_delay, mean_response_test, bounds=([0, -20],[20, 20]))
        # popt, pcov = curve_fit(sigmoid, sum_delay, mean_response_test)
        # print(f'[{slice_loc:<12}] k={popt[0]} x0={popt[1]}')
        # print(pcov)
        parameters.append({'device': dev, 'slice': slice_loc,'k': popt[0], 'x0': popt[1]})
        
        ax.plot(x_sigmoid, sigmoid(x_sigmoid, *popt), color=colors[slice_loc], alpha=1.0, linestyle="--", linewidth=1.2, label=f"[{dev}] {slice_loc}")
    
        ax.set_xlabel(r"$\sum w\cdot\Phi_{test}$")
        ax.set_ylabel(r"Mean Response")
        ax.set_title(slice_loc)
        ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.02), fontsize=7)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.4)
        ax.axvline(color="black", linestyle="--", alpha=0.4)
        ax.set(xlim=(-10, 10))

params_df = pl.DataFrame(parameters)
fig.tight_layout()
plt.show()
fig.savefig("figures/apuf_delay_projection_all_devices.png")

## Cluster analysis

challenges = np.load("./data/challenges/ergodicity_clusters_challenges.npy")
challenges_polar = (-1)**challenges.astype(np.int8)

challenges_phi = ArbiterPUF.transform_atf(challenges_polar, k=1).reshape(len(challenges), -1)

df_summary_all = pl.read_csv("./data/ergodicity_cluster/crps_summary_merged.csv").filter(pl.col('stages') == 64).with_columns(baseline_id = np.tile(np.repeat(np.arange(100), 2**11), 4*23))
devices = df_summary_all['device_id'].unique().to_list()
slices = df_summary_all['slice'].unique().to_list()
baselines = df_summary_all['baseline_id'].unique().to_list()

ws = {}
ids_test_slices = {}

for name, df_group in df_summary_all.group_by(['device_id', 'slice', 'baseline_id'], maintain_order=True):
    print(name)
    w, ids_test = compute_w(df_group, challenges_phi)
    ws[name] = w
    ids_test_slices[name] = ids_test

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
# colors = dict(zip(slices, ['orange', 'blue', 'darkgreen', 'purple']))
cmap = plt.colormaps["tab20"]
device_colors = {
    dev: cmap(i % cmap.N)
    for i, dev in enumerate(devices)
}

x_sigmoid = np.linspace(-50, 50, 1000)

def sigmoid(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k*(x-x0)))

mapping = {
    'LEFT-UPPER': (0, 0),
    'LEFT-LOWER': (1, 0), 
    'RIGHT-LOWER': (1, 1),
    'RIGHT-UPPER': (0, 1),
}

seen_devices = {slice_loc: set() for slice_loc in mapping.keys()}
parameters = []
for name, df_group in df_summary_all.group_by(['device_id', 'slice', 'baseline_id'], maintain_order=True):
    device_id, slice_loc, baseline_id = name
    # print(name)
    if baseline_id >= 10:
        # print(f"{device_id} {slice_loc} Maximum baseline reached")
        continue
    ax = axs[mapping[slice_loc]]
    mean_response = df_group['mean_response'].to_numpy()
    
    ids_test = ids_test_slices[name]
    mean_response_test = mean_response[ids_test]
    w = ws[name]
    
    delta_t = w * challenges_phi[ids_test]
    sum_delay = np.sum(delta_t, axis=1)

    try:
        popt, pcov = curve_fit(sigmoid, sum_delay, mean_response_test, bounds=([0, -50],[50, 50]))
    except RuntimeError:
        continue
    # popt, pcov = curve_fit(sigmoid, sum_delay, mean_response_test)
    parameters.append({'device': device_id, 'slice': slice_loc, 'baseline_id': baseline_id, 'k': popt[0], 'x0': popt[1]})

    label = None
    if device_id not in seen_devices[slice_loc]:
        label = str(device_id)
        seen_devices[slice_loc].add(device_id)

    ax.plot(x_sigmoid, sigmoid(x_sigmoid, *popt), color=device_colors[device_id], alpha=0.35, linestyle="--", linewidth=1.0, label=label)
    
    ax.set_xlabel(r"$\sum w\cdot\Phi_{test}$")
    ax.set_ylabel(r"Mean Response")
    ax.set_title(slice_loc)
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.02), fontsize=5)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.3)
    ax.axvline(color="black", linestyle="--", alpha=0.3)
    ax.set(xlim=(-10, 10))

params_df = pl.DataFrame(parameters)
fig.tight_layout()
plt.show()
fig.savefig("figures/apuf_delay_projection_all_baselines.png")
