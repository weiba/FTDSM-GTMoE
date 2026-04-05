import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as Data
from sklearn.model_selection import StratifiedKFold
from model import * 
import utils
from utils import load_patch

def save_metrics_txt(fold_test_metrics, dataset_name, win_size, stride, num_expert, top_k):
    save_dir = "FIVE_CV_TEST_RESULT"
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(
        save_dir,
        f"{dataset_name}_WIN_SIZE{win_size}_STRIDE_{stride}"
        f"_NUM_EXPERTS_{num_expert}_TOP_K_{top_k}_woroi.txt"
    )

    metrics_name = ['ACC', 'SEN', 'SPE', 'BAC', 'F1', 'MCC', 'AUC', 'KAPPA']
    content = []

    content.append("=== Per Outer Fold Results ===")
    for fold_idx, metrics in enumerate(fold_test_metrics):
        content.append(f"\nOuter Fold {fold_idx + 1}:")
        for name, value in zip(metrics_name, metrics):
            content.append(f"{name}: {value * 100:.10f}")

    content.append("\n\n=== Final Results (Mean ± Std) ===")
    for i, name in enumerate(metrics_name):
        values = [fold[i] for fold in fold_test_metrics]
        mean = np.mean(values) * 100
        std = np.std(values) * 100
        content.append(f"{name}: {mean:.10f}% ± {std:.10f}%")

    with open(filename, 'w') as f:
        f.write("\n".join(content))

    print(f"\nResults saved to: {filename}")


class ADNI(Dataset):
    def __init__(self, window_seqs, window_pccs, global_adjs, patch_adjs, labels):
        super(ADNI, self).__init__()
        self.window_seqs = window_seqs
        self.window_pccs = window_pccs
        self.global_adjs = global_adjs
        self.patch_adjs = patch_adjs
        self.labels = labels

    def __getitem__(self, item):
        window_seq = self.window_seqs[item]
        window_pcc = self.window_pccs[item]
        global_adj = self.global_adjs[item]
        patch_adj = self.patch_adjs[item]
        label = self.labels[item]

        return window_seq, window_pcc, global_adj, patch_adj, label, item

    def __len__(self):
        return self.window_seqs.shape[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on fMRI data.")
    parser.add_argument('--data-path', type=str, required=True, help="Path to the data file.")
    parser.add_argument('--embed-size', type=int, default=64, help="Embedding size.")
    parser.add_argument('--hidden-size', type=int, default=128, help="Hidden size.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.") 
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--layer-num', type=int, default=4, help="Number of Layers.")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size.")
    
    return parser.parse_args()


if __name__ == "__main__":
    win_size = 10
    stride = 5
    args = parse_args()
    print(args)
    
    task_name = os.path.basename(args.data_path).split('.')[0].replace('_', '-')
    if task_name.endswith('-new'):
        task_name = task_name[:-4] 
    
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE ", device)

    num_expert = 4
    top_k = 2
    lb_coef = 0.1    
    
    print("Loading data......")
    num_window = int((90 - win_size) / stride) + 1
    
    data_path = args.data_path 
    window_seqs, window_pccs, window_adjs, global_adjs, patch_adjs, labels, dataset_name = load_patch(
        data_path, w=win_size, s=stride, percentile_same_roi=80, percentile_same_win=80, percentile_diff=80
    )
    
    dataset = ADNI(window_seqs, window_pccs, global_adjs, patch_adjs, labels)
    overall_metrics = []  
    
    outer_k = 5
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=42)

    # --- Outer Fold Loop ---
    for outer_fold, (train_val_indices, test_indices) in enumerate(outer_cv.split(np.arange(len(labels)), labels)):
        print(f"\n=== Outer Fold {outer_fold + 1}/{outer_k} ===")

        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                 sampler=Data.SubsetRandomSampler(test_indices), drop_last=False)

        train_val_labels = np.array(labels)[train_val_indices]
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fold_metrics = []

        # --- Inner Fold Loop ---
        for inner_fold, (train_index, val_index) in enumerate(inner_cv.split(train_val_indices, train_val_labels)):
            print(f"Inner Fold {inner_fold + 1}/5")

            train_sub_idx = train_val_indices[train_index]
            val_sub_idx = train_val_indices[val_index]

            train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                      sampler=Data.SubsetRandomSampler(train_sub_idx), drop_last=True)
            val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                    sampler=Data.SubsetRandomSampler(val_sub_idx), drop_last=False)

            model = FTDSM(
                embed_size=args.embed_size, 
                hidden_size=args.hidden_size,
                layer_num=args.layer_num, 
                num_window=num_window,
                win_size=win_size, 
                num_experts=num_expert, 
                top_k=top_k, 
                lb_coef=lb_coef,
                device=device
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            loss_func = nn.CrossEntropyLoss().to(device)

            best_val_acc = 0.0
            best_model_path = f"BEST_MODEL_PARAM_FOR_OUTER{outer_fold}_INNER{inner_fold}_ON_{dataset_name}.pth"

            for epoch in range(1, args.epochs + 1):
                model.train()

                for train_win_seq, train_window_pccs, train_global_adjs, train_patch_adjs, train_labels, _ in train_loader:
                    train_win_seq = train_win_seq.to(device)
                    train_window_pccs = train_window_pccs.to(device)
                    train_global_adjs = train_global_adjs.to(device)
                    train_patch_adjs = train_patch_adjs.to(device)
                    train_labels = train_labels.to(device)
                    
                    output, mamba_loss, *extras = model(train_win_seq, train_window_pccs, train_global_adjs, train_patch_adjs)
                    loss = loss_func(output, train_labels) + 0.0025 * mamba_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                all_val_outputs, all_val_labels = [], []
                with torch.no_grad():
                    for val_win_seq, val_window_pccs, val_global_adjs, val_patch_adjs, val_labels, _ in val_loader:
                        val_win_seq = val_win_seq.to(device)
                        val_window_pccs = val_window_pccs.to(device)
                        val_global_adjs = val_global_adjs.to(device)
                        val_patch_adjs = val_patch_adjs.to(device)
                        val_labels = val_labels.to(device)
                        
                        output, *extras = model(val_win_seq, val_window_pccs, val_global_adjs, val_patch_adjs)
                        all_val_outputs.append(output.cpu())
                        all_val_labels.append(val_labels.cpu())
                
                all_val_outputs = torch.cat(all_val_outputs, dim=0)
                all_val_labels = torch.cat(all_val_labels, dim=0)
                val_ACC, *val_metrics = utils.stastic_indicators(all_val_outputs, all_val_labels)
                
                if epoch % 20 == 0:
                    print(f"Outer {outer_fold + 1} Inner {inner_fold + 1} Epoch {epoch}: Val ACC {val_ACC.item():.4f}")

                if val_ACC > best_val_acc:
                    best_val_acc = val_ACC
                    torch.save(model.state_dict(), best_model_path)

            model.load_state_dict(torch.load(best_model_path))
            model.eval()
            
            all_test_outputs = []
            all_test_labels = []
            all_roi_imps = []    
            all_patch_imps = []
            all_sample_indices = [] 
            
            with torch.no_grad():
                for test_win_seq, test_window_pccs, test_global_adjs, test_patch_adjs, test_labels, test_indices_batch in test_loader:
                    test_win_seq = test_win_seq.to(device)
                    test_window_pccs = test_window_pccs.to(device)
                    test_global_adjs = test_global_adjs.to(device)
                    test_patch_adjs = test_patch_adjs.to(device)
                    test_labels = test_labels.to(device)
                    
                    output, _, roi_importance, patch_importance = model(
                        test_win_seq, test_window_pccs, test_global_adjs, test_patch_adjs
                    )
                    
                    all_test_outputs.append(output.cpu())
                    all_test_labels.append(test_labels.cpu())
                    all_roi_imps.append(roi_importance.cpu().numpy())
                    all_patch_imps.append(patch_importance.cpu().numpy())
                    all_sample_indices.append(test_indices_batch.cpu().numpy()) 

            all_test_outputs = torch.cat(all_test_outputs, dim=0)
            all_test_labels = torch.cat(all_test_labels, dim=0)
            all_roi_imps = np.concatenate(all_roi_imps, axis=0)     
            all_patch_imps = np.concatenate(all_patch_imps, axis=0)
            all_sample_indices = np.concatenate(all_sample_indices, axis=0) 

            preds = torch.argmax(all_test_outputs, dim=1).numpy()
            probs = torch.softmax(all_test_outputs, dim=1).numpy()
            true_labels = all_test_labels.numpy()

            save_dir = "INTERPRETABILITY_RESULTS"
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(
                save_dir, 
                f"{dataset_name}_OuterFold{outer_fold}_Inner{inner_fold}_Importance.npz"
            )
            
            np.savez(save_path, 
                     roi_importance=all_roi_imps, 
                     patch_importance=all_patch_imps, 
                     labels=true_labels, 
                     preds=preds,
                     probs=probs,
                     sample_indices=all_sample_indices) 
            
            print(f"Saved importance scores and indices to {save_path}")

            metrics = utils.stastic_indicators(all_test_outputs, all_test_labels)
            print(f"\n>>> Test Results (Outer {outer_fold+1} Inner {inner_fold+1}) <<<")
            for name, value in zip(['ACC', 'SEN', 'SPE', 'BAC', 'F1', 'MCC', 'AUC', 'KAPPA'], metrics):
                print(f"{name}: {value * 100:.2f}%")
            fold_metrics.append(metrics)

            if os.path.exists(best_model_path):
                os.remove(best_model_path)

        avg_metrics = tuple(np.mean([m[i] for m in fold_metrics]) for i in range(len(fold_metrics[0])))
        overall_metrics.append(avg_metrics)

        print(f"\n>>> Average Test Results for Outer Fold {outer_fold+1} <<<")
        for name, value in zip(['ACC', 'SEN', 'SPE', 'BAC', 'F1', 'MCC', 'AUC', 'KAPPA'], avg_metrics):
            print(f"{name}: {value * 100:.2f}%")

    save_metrics_txt(overall_metrics, dataset_name=dataset_name, win_size=win_size, stride=stride, num_expert=num_expert, top_k=top_k)