import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, average_precision_score
from scipy.spatial import cKDTree

EPS = 1e-7


def dice_score(pred_bin, gt_bin):
    p = pred_bin.reshape(-1).astype(np.float32)
    g = gt_bin.reshape(-1).astype(np.float32)

    inter = np.dot(p, g)
    return float((2.0 * inter + 1.0) / (p.sum() + g.sum() + 1.0))


def hausdorff95(pred_bin, gt_bin):
    p_pts = np.argwhere(pred_bin > 0)
    g_pts = np.argwhere(gt_bin > 0)

    if len(p_pts) == 0 or len(g_pts) == 0:
        return None

    tree_g = cKDTree(g_pts)
    tree_p = cKDTree(p_pts)

    d_p2g, _ = tree_g.query(p_pts, k=1)
    d_g2p, _ = tree_p.query(g_pts, k=1)

    return float(np.percentile(np.concatenate([d_p2g, d_g2p]), 95))


def compute_seg_metrics(pred_prob, gt_mask, threshold=0.5):
    pred_bin = (pred_prob >= threshold).astype(np.uint8)
    gt_bin   = (gt_mask >= 0.5).astype(np.uint8)

    inter = (pred_bin * gt_bin).sum()
    p_sum = pred_bin.sum()
    g_sum = gt_bin.sum()

    dice = (2 * inter + 1.0) / (p_sum + g_sum + 1.0)
    iou  = inter / (p_sum + g_sum - inter + EPS)
    precision = inter / (p_sum + EPS)
    recall    = inter / (g_sum + EPS)

    hd = hausdorff95(pred_bin, gt_bin)

    return {
        "DSC": float(dice),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "HD95": float(hd) if hd is not None else None,
    }


def compute_cls_metrics(y_prob, y_true, threshold=0.5):
    y_prob = np.asarray(y_prob)
    y_true = np.asarray(y_true)

    y_pred = (y_prob >= threshold).astype(np.int32)

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)
    else:
        auc, ap = 0.0, 0.0

    acc = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    sensitivity = tp / (tp + fn + EPS)
    specificity = tn / (tn + fp + EPS)
    precision   = tp / (tp + fp + EPS)
    f1 = 2 * tp / (2 * tp + fp + fn + EPS)

    return {
        "AUC": float(auc),
        "AP": float(ap),
        "Accuracy": float(acc),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "F1": float(f1),
    }


class MetricStore:
    def __init__(self):
        self._data = {}

    def update(self, d):
        for k, v in d.items():
            if v is None or np.isnan(v):
                continue
            self._data.setdefault(k, []).append(float(v))

    def mean(self):
        return {
            k: round(float(np.mean(v)), 4)
            for k, v in self._data.items()
            if len(v) > 0
        }

    def std(self):
        return {
            k: round(float(np.std(v)), 4)
            for k, v in self._data.items()
            if len(v) > 0
        }

    def reset(self):
        self._data = {}