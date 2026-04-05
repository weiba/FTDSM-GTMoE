import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import  roc_auc_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

np.random.seed(123)
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def load_patch(data_path, w=10, s=5,
                    percentile_same_roi=80,
                    percentile_same_win=80,
                    percentile_diff=80):
    data = np.load(data_path)
    fmri_data = data['fmri_data']  
    B, N, T = fmri_data.shape
    num_windows = (T - w) // s + 1

    all_window_seqs = []
    all_window_pccs = []
    all_window_adjs = []
    global_adjs = []
    patch_adjs = []

    for i in tqdm(range(B)):
        x = fmri_data[i]  
        window_seqs = []
        window_pccs = []
        window_adjs = []

        for t in range(0, T - w + 1, s):
            x_window = x[:, t:t + w]
            window_seqs.append(x_window)

            pcc = np.corrcoef(x_window)
            abs_pcc = np.abs(pcc)
            threshold = np.percentile(abs_pcc, 80)
            adj = (abs_pcc >= threshold).astype(np.float32)
            np.fill_diagonal(adj, 1)
            adj = normalize_adj(adj)

            src, dst = np.where(adj != 0)
            edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
            edge_weight = torch.tensor(adj[src, dst], dtype=torch.float32)
            adj_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, size=(N, N))

            window_pccs.append(pcc)
            window_adjs.append(adj_sparse)

        all_window_seqs.append(torch.tensor(np.stack(window_seqs), dtype=torch.float32))
        all_window_pccs.append(torch.tensor(np.stack(window_pccs), dtype=torch.float32))
        all_window_adjs.append(window_adjs)


        pcc_full = np.corrcoef(x)
        abs_pcc_full = np.abs(pcc_full)
        threshold = np.percentile(abs_pcc_full, 80)
        adj_full = (abs_pcc_full >= threshold).astype(np.float32)
        np.fill_diagonal(adj_full, 1)
        adj_full = normalize_adj(adj_full)

        src, dst = np.where(adj_full != 0)
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        edge_weight = torch.tensor(adj_full[src, dst], dtype=torch.float32)
        global_adj_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, size=(N, N))
        global_adjs.append(global_adj_sparse)


        patches = np.stack(window_seqs)  
        num_patches = num_windows * N
        patches = patches.reshape(num_patches, w)


        patch_pcc = np.corrcoef(patches)
        abs_patch_pcc = np.abs(patch_pcc)
        np.fill_diagonal(abs_patch_pcc, 0)


        roi_idx = np.tile(np.arange(N), num_windows)
        win_idx = np.repeat(np.arange(num_windows), N)

        same_roi_diff_win = (roi_idx[:, None] == roi_idx[None, :]) & (win_idx[:, None] != win_idx[None, :])
        same_win_diff_roi = (win_idx[:, None] == win_idx[None, :]) & (roi_idx[:, None] != roi_idx[None, :])
        diff_roi_diff_win = (roi_idx[:, None] != roi_idx[None, :]) & (win_idx[:, None] != win_idx[None, :])

        patch_adj = np.zeros_like(patch_pcc, dtype=np.float32)


        th1 = np.percentile(abs_patch_pcc[same_roi_diff_win], percentile_same_roi)
        patch_adj[same_roi_diff_win] = (abs_patch_pcc[same_roi_diff_win] >= th1).astype(np.float32)

        th2 = np.percentile(abs_patch_pcc[same_win_diff_roi], percentile_same_win)
        patch_adj[same_win_diff_roi] = (abs_patch_pcc[same_win_diff_roi] >= th2).astype(np.float32)

        th3 = np.percentile(abs_patch_pcc[diff_roi_diff_win], percentile_diff)
        patch_adj[diff_roi_diff_win] = (abs_patch_pcc[diff_roi_diff_win] >= th3).astype(np.float32)

        np.fill_diagonal(patch_adj, 1)
        patch_adj = normalize_adj(patch_adj)

        src, dst = np.where(patch_adj != 0)
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
        edge_weight = torch.tensor(patch_adj[src, dst], dtype=torch.float32)
        patch_adj_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, size=(num_patches, num_patches))
        patch_adjs.append(patch_adj_sparse)


    if "AD_NC_new.npz" in data_path:
        labels = torch.LongTensor([1] * 222 + [0] * 213)
        dataset_name = "AD_NC"
    elif "LMCI_NC_new.npz" in data_path:
        labels = torch.LongTensor([1] * 192 + [0] * 213)
        dataset_name = "LMCI_NC"
    elif "LMCI_AD_new.npz" in data_path:
        labels = torch.LongTensor([1] * 192 + [0] * 222)
        dataset_name = "LMCI_AD"
    elif "EMCI_LMCI_new.npz" in data_path:
        labels = torch.LongTensor([1] * 190 + [0] * 192)
        dataset_name = "EMCI_LMCI"
    elif "EMCI_LMCI_ADNI3.npz" in data_path:
        labels = torch.LongTensor([1] * 65 + [0] * 78)
        dataset_name = "EMCI_LMCI_ADNI3"
    elif "LMCI_AD_ADNI3.npz" in data_path:
        labels = torch.LongTensor([1] * 78 + [0] * 134)
        dataset_name = "LMCI_AD_ADNI3"
    elif "EMCI_LMCI_test.npz" in data_path:
        labels = torch.LongTensor([1] * 125 + [0] * 114)
        dataset_name = "EMCI_LMCI_test"
    elif "LMCI_AD_test.npz" in data_path:
        labels = torch.LongTensor([1] * 114 + [0] * 87)
        dataset_name = "LMCI_AD_test"    
    else:
        raise ValueError("Unknown dataset!")

    return all_window_seqs, all_window_pccs, all_window_adjs, global_adjs, patch_adjs, labels, dataset_name
  
def stastic_indicators(output, labels):
    epsilon = 1e-7  
    predictions = output.max(1)[1]
    
    TP = ((predictions == 1) & (labels == 1)).sum()
    TN = ((predictions == 0) & (labels == 0)).sum()
    FN = ((predictions == 0) & (labels == 1)).sum()
    FP = ((predictions == 1) & (labels == 0)).sum()

    ACC = (TP + TN) / (TP + TN + FP + FN + epsilon)
    SEN = TP / (TP + FN + epsilon)
    P = TP / (TP + FP + epsilon)
    SPE = TN / (FP + TN + epsilon)
    BAC = (SEN + SPE) / 2
    F1 = (2 * P * SEN) / (P + SEN + epsilon)
    MCC = ((TP * TN) - (FP * FN)) / (torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + epsilon)

    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    try:
        AUC = roc_auc_score(labels, output[:,1])
    except ValueError:
        AUC = 0.0

    try:
      kappa = cohen_kappa_score(labels,predictions)
    except ValueError:
      kappa = 0.0

    return ACC, SEN, SPE, BAC, F1, MCC, AUC,kappa




