import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.metrics import roc_curve, auc


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    accuracy, precision, recall, F1; AUC if probs given.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    if y_prob is not None:
        y_prob = y_prob.flatten()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        metrics.update({'auc': float(roc_auc)})
    return metrics


def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def plot_roc_curve(y_true, y_prob):
    y_true = y_true.flatten();
    y_prob = y_prob.flatten()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', lw=1)
    ax.set_xlabel('FPR');
    ax.set_ylabel('TPR');
    ax.legend()
    buf = io.BytesIO();
    fig.savefig(buf, format='png');
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8'), float(roc_auc)


def plot_cost_history(costs, val_costs=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(costs, label='Train')
    if val_costs is not None:
        ax.plot(val_costs, label='Val')
    ax.set_xlabel('Iteration');
    ax.set_ylabel('Cost');
    ax.legend()
    buf = io.BytesIO();
    fig.savefig(buf, format='png');
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def plot_kfold_results(fold_metrics):
    folds = [m['fold'] for m in fold_metrics]
    accs = [m['accuracy'] for m in fold_metrics]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(folds, accs);
    ax.axhline(np.mean(accs), color='red', linestyle='--')
    ax.set_xlabel('Fold');
    ax.set_ylabel('Accuracy')
    buf = io.BytesIO();
    fig.savefig(buf, format='png');
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
