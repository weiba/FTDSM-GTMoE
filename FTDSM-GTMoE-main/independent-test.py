import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from model import *
import utils
from utils import load_patch

def save_results(fold_test_metrics, dataset_name, win_size, stride):
    time_stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    os.makedirs("result", exist_ok=True)
    filename = os.path.join(
        "result", 
        f"{dataset_name}_idp_test-result_{time_stamp}_win{win_size}_stride{stride}_FTDSM.txt"
    )

    metrics_name = ['ACC', 'SEN', 'SPE', 'BAC', 'F1', 'MCC', 'AUC', 'KAPPA']
    content = []

    content.append("=== Per Fold Results ===")
    for fold_idx, metrics in enumerate(fold_test_metrics):
        content.append(f"\nFold {fold_idx+1}:")
        for name, value in zip(metrics_name, metrics):
            content.append(f"{name}: {value*100:.10f}")

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

        return window_seq, window_pcc, global_adj, patch_adj, label

    def __len__(self):
        return len(self.window_seqs)


def parse_args():
    parser = argparse.ArgumentParser(description="Independent testing on fMRI data.")
    parser.add_argument('--embed-size', type=int, default=64, help="Embedding size.")
    parser.add_argument('--hidden-size', type=int, default=128, help="Hidden size.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--layer-num', type=int, default=4, help="Number of Layers.")
    parser.add_argument('--lamb', type=float, default=0.5, help="Threshold for Adjacency Matrix Construction.")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size.")
    
    return parser.parse_args()


if __name__ == "__main__":
    win_size = 10
    stride = 5
    args = parse_args()
    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE ", device)
    
    num_expert = 4
    top_k = 2
    lb_coef = 0.1
    num_window = int((90 - win_size) / stride) + 1
    
    print("Loading data......")
    train_path = 'Data/Independent_data_center/train_data/EMCI_LMCI_ADNI3.npz'
    test_path = 'Data/Independent_data_center/test_data/EMCI_LMCI_test.npz'
    
    train_window_seqs, train_window_pccs, train_window_adjs, train_global_adjs, train_patch_adjs, train_labels, train_dataset_name = load_patch(
        train_path, w=win_size, s=stride, percentile_same_roi=80, percentile_same_win=80, percentile_diff=80
    )
    test_window_seqs, test_window_pccs, test_window_adjs, test_global_adjs, test_patch_adjs, test_labels, test_dataset_name = load_patch(
        test_path, w=win_size, s=stride, percentile_same_roi=80, percentile_same_win=80, percentile_diff=80
    )
    
    train_dataset = ADNI(train_window_seqs, train_window_pccs, train_global_adjs, train_patch_adjs, train_labels)
    test_dataset = ADNI(test_window_seqs, test_window_pccs, test_global_adjs, test_patch_adjs, test_labels)

    task_name = train_dataset_name
    test_sets = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_labels_array = np.array(train_labels)

    print(f"\nTraining with {args.layer_num} layers......")

    ACClist, SENlist, SPElist, BAClist, F1list, MCClist, AUClist, KAPPAlist = [], [], [], [], [], [], [], []
    
    for fold, (train_index, val_index) in enumerate(kfold.split(np.arange(len(train_dataset)), train_labels_array)):
        print(f"\n=== Fold {fold + 1}/{kfold.n_splits} ===")

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
        
        train_subset = torch.utils.data.Subset(train_dataset, train_index)
        val_subset = torch.utils.data.Subset(train_dataset, val_index)
        
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        best_val_acc = 0.0
        best_model_path = f"{fold}-{task_name}-FTDSM-best-idp-P{test_dataset_name}.pth"
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            for train_win_seq, train_window_pccs, train_global_adjs, train_patch_adjs, train_labels in train_loader:
                train_win_seq = train_win_seq.to(device)
                train_window_pccs = train_window_pccs.to(device)
                train_global_adjs = train_global_adjs.to(device)
                train_patch_adjs = train_patch_adjs.to(device)
                train_labels = train_labels.to(device)
                
                output, mamba_loss, *_ = model(train_win_seq, train_window_pccs, train_global_adjs, train_patch_adjs)
                loss = loss_func(output, train_labels) + 0.0025 * mamba_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            all_val_outputs = []
            all_val_labels = []
            with torch.no_grad():
                for val_win_seq, val_window_pccs, val_global_adjs, val_patch_adjs, val_labels in val_loader:
                    val_win_seq = val_win_seq.to(device)
                    val_window_pccs = val_window_pccs.to(device)
                    val_global_adjs = val_global_adjs.to(device)
                    val_patch_adjs = val_patch_adjs.to(device)
                    val_labels = val_labels.to(device)
                    
                    output_val, *_ = model(val_win_seq, val_window_pccs, val_global_adjs, val_patch_adjs)
                    all_val_outputs.append(output_val.cpu())
                    all_val_labels.append(val_labels.cpu())
            
            all_val_outputs = torch.cat(all_val_outputs, dim=0)
            all_val_labels = torch.cat(all_val_labels, dim=0)
            val_ACC, val_SEN, val_SPE, val_BAC, val_F1, val_MCC, val_AUC, val_KAPPA = utils.stastic_indicators(all_val_outputs, all_val_labels)
            
            if epoch % 20 == 0:
                print(f'| Epoch {epoch:3d} | Val ACC {val_ACC * 100:.2f}% | AUC {val_AUC * 100:.2f}%')
            
            if val_ACC > best_val_acc:
                best_val_acc = val_ACC
                torch.save(model.state_dict(), best_model_path)

        # --- Independent Test Phase ---
        model.eval()
        all_test_outputs = []
        all_test_labels = []
        model.load_state_dict(torch.load(best_model_path))
        
        with torch.no_grad():
            for test_win_seq, test_window_pccs, test_global_adjs, test_patch_adjs, test_labels in test_sets:
                test_win_seq = test_win_seq.to(device)
                test_window_pccs = test_window_pccs.to(device)
                test_global_adjs = test_global_adjs.to(device)
                test_patch_adjs = test_patch_adjs.to(device)
                test_labels = test_labels.to(device)
                
                output_test, *_ = model(test_win_seq, test_window_pccs, test_global_adjs, test_patch_adjs)

                all_test_outputs.append(output_test.cpu())
                all_test_labels.append(test_labels.cpu())

        all_test_outputs = torch.cat(all_test_outputs, dim=0)
        all_test_labels = torch.cat(all_test_labels, dim=0)
        
        if os.path.exists(best_model_path):
            os.remove(best_model_path)


        test_ACC, test_SEN, test_SPE, test_BAC, test_F1, test_MCC, test_AUC, test_KAPPA = utils.stastic_indicators(all_test_outputs, all_test_labels)
        print(f'>>> Fold {fold + 1} IDP Test | ACC {test_ACC * 100:.2f}% | SEN {test_SEN * 100:.2f}% | SPE {test_SPE * 100:.2f}% | AUC {test_AUC * 100:.2f}%')
        
        ACClist.append(test_ACC)
        SENlist.append(test_SEN)
        SPElist.append(test_SPE)
        BAClist.append(test_BAC)
        F1list.append(test_F1)
        MCClist.append(test_MCC)
        AUClist.append(test_AUC)
        KAPPAlist.append(test_KAPPA)

    fold_test_metrics = list(zip(ACClist, SENlist, SPElist, BAClist, F1list, MCClist, AUClist, KAPPAlist))

    print("\n>>> Final Averaged Results across Folds <<<")
    print(f"ACC: {np.mean(ACClist)*100:.2f}%")
    print(f"AUC: {np.mean(AUClist)*100:.2f}%")


    save_results(fold_test_metrics, dataset_name=task_name, win_size=win_size, stride=stride)