import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, f1_score, r2_score


# Map task type -> metric functions
METRICS_CONFIG = {
    "binary_classification": {
        "accuracy": "_binary_accuracy",
        "roc_auc": "_binary_roc_auc",
        "f1": "_binary_f1_score",
    },
    "multiclass_classification": {
        "accuracy": "_accuracy",
        "f1_macro": "_multiclass_f1_macro",
    },
    "regression": {
        "mse": "_mse",
        "mae": "_mae",
        "r2": "_r2_score",
        "pearson": "_pearson_correlation_coefficient",
        "spearman": "_spearman_correlation_coefficient",
    },
}

def get_task_metrics(task_type):
    task_type = task_type.lower()
    assert task_type in METRICS_CONFIG, f"[metrics.get_task_metrics] Unsupported task type: {task_type}"
    task_metrics = METRICS_CONFIG[task_type]
    return {name: globals()[func_name] for name, func_name in task_metrics.items()}


def compute_metrics(metrics_dict, outputs, labels):
    return {metric_name: metric_fn(outputs, labels) for metric_name, metric_fn in metrics_dict.items()}


# ------------------------------
# Multiclass classification
# ------------------------------
def _accuracy(out, labels):
    outputs = np.argmax(np.asarray(out), axis=1)
    true_labels = np.argmax(np.asarray(labels), axis=1)
    return float(np.sum(outputs == true_labels)) / float(np.asarray(labels).shape[0])


def _multiclass_f1_macro(y_pred, y_true):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    return f1_score(y_true_labels, y_pred_labels, average="macro")


# ------------------------------
# Binary classification
# ------------------------------
def _binary_accuracy(y_pred, y_true, threshold=0.5):
    y_pred = np.asarray(y_pred).squeeze()
    y_true = np.asarray(y_true).squeeze().astype(int)
    preds = (y_pred >= threshold).astype(int)
    return float(np.mean(preds == y_true))


def _binary_f1_score(y_pred, y_true, threshold=0.5):
    y_pred = np.asarray(y_pred).squeeze()
    y_true = np.asarray(y_true).squeeze().astype(int)
    preds = (y_pred >= threshold).astype(int)
    return f1_score(y_true, preds)


def _binary_roc_auc(y_pred, y_true):
    y_pred = np.asarray(y_pred).squeeze()
    y_true = np.asarray(y_true).squeeze().astype(int)
    if np.unique(y_true).size == 2:
        return roc_auc_score(y_true, y_pred)
    return 0.5


# ------------------------------
# Regression
# ------------------------------
def _mse(y_pred, y_true):
    y_pred = np.asarray(y_pred).squeeze()
    y_true = np.asarray(y_true).squeeze()
    return float(np.mean((y_true - y_pred) ** 2))


def _mae(y_pred, y_true):
    y_pred = np.asarray(y_pred).squeeze()
    y_true = np.asarray(y_true).squeeze()
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2_score(y_pred, y_true):
    y_pred = np.asarray(y_pred).squeeze()
    y_true = np.asarray(y_true).squeeze()
    return float(r2_score(y_true, y_pred))


def _pearson_correlation_coefficient(y_pred, y_true):
    y_pred = np.asarray(y_pred).squeeze()
    y_true = np.asarray(y_true).squeeze()
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def _spearman_correlation_coefficient(y_pred, y_true):
    y_pred = np.asarray(y_pred).squeeze()
    y_true = np.asarray(y_true).squeeze()
    r, _ = spearmanr(y_true, y_pred)
    return float(r)


