import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from selective_modeling.modules.graph_selective_modeling import Mamba

class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout, act=torch.relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        stdv = math.sqrt(1.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.dropout(input)            
        support = torch.matmul(x, self.weight)  

        if isinstance(adj, list):
            adj_dense = torch.stack([a.to_dense() if hasattr(a, "to_dense") else a for a in adj], dim=0)
        elif hasattr(adj, "to_dense") and not adj.is_sparse:
            adj_dense = adj
        elif hasattr(adj, "to_dense") and adj.is_sparse:
            adj_dense = adj.to_dense()
        else:
            adj_dense = adj

        output = torch.matmul(adj_dense, support)  
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)


def shannon_entropy(matrix):
    matrix = F.softmax(matrix, dim=1)
    log_matrix = torch.log2(matrix)
    elementwise_product = matrix * log_matrix
    entropy = -torch.sum(elementwise_product, dim=1)
    return entropy.squeeze()


class FTDSM(nn.Module):
    def __init__(self, embed_size, hidden_size, layer_num=5,
                 hidden_size_factor=1, sparsity_threshold=0.01,
                 num_window=17, win_size=10,
                 num_experts=4, top_k=2, use_gcn_experts=True, lb_coef=0.1, device='cuda:0'):
        super().__init__()
        self.device = torch.device(device)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.layer_num = layer_num
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.scale = 0.02
        self.win_size = win_size
        self.num_window = num_window
        
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        
        self.DGSM1 = Mamba(d_model=116, d_state=16, d_conv=4, expand=1, expand_out=10)
        self.DGSM2 = Mamba(d_model=1160, d_state=16, d_conv=4, expand=1, expand_out=1)
        
        self.w_t_list = nn.ParameterList([nn.Parameter(torch.randn(win_size) * 0.01) for _ in range(self.num_window)])
        self.w_c_list = nn.ParameterList([nn.Parameter(torch.randn(embed_size) * 0.01) for _ in range(self.num_window)])
        self.weights = nn.ModuleList()
        self.biases = nn.ModuleList()
        
        for _ in range(self.num_window):
            layer_weights = nn.ParameterList()
            layer_biases = nn.ParameterList()
            for _ in range(self.layer_num):
                layer_weights.append(nn.Parameter(self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor)))
                layer_biases.append(nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor)))
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)

        self.LayerNorm = nn.LayerNorm(1160)

        self.W = nn.Parameter(torch.empty(2 * self.win_size * self.num_window, self.win_size * self.num_window, device=self.device))
        self.b = nn.Parameter(torch.zeros(win_size * self.num_window, device=self.device))
        torch.nn.init.xavier_uniform_(self.W)
        
        self.f1 = nn.Flatten()
        self.bn2 = torch.nn.BatchNorm1d(512, eps=1e-05)
        self.bn3 = torch.nn.BatchNorm1d(256, eps=1e-05)
        self.bn4 = torch.nn.BatchNorm1d(128, eps=1e-05)
        self.l1 = nn.Linear(19720, 512)
        self.d1 = nn.Dropout(p=0.5)
        self.l2 = nn.Linear(512, 256)
        self.d2 = nn.Dropout(p=0.5)
        self.l3 = nn.Linear(256, 128)
        self.d3 = nn.Dropout(p=0.5)
        self.l4 = nn.Linear(128, 2)

        self.num_experts = num_experts
        self.top_k = top_k
        self.use_gcn_experts = use_gcn_experts
        self.expert_dim = 170   
        self.lb_coef = lb_coef

        if self.use_gcn_experts:
            self.experts = nn.ModuleList([
                GraphConvolution(in_features=self.expert_dim, out_features=self.expert_dim, dropout=0.0, act=torch.relu, bias=False)
                for _ in range(self.num_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.expert_dim, self.expert_dim),
                    nn.ReLU(),
                    nn.Linear(self.expert_dim, self.expert_dim)
                )
                for _ in range(self.num_experts)
            ])

        self.router_ln = nn.LayerNorm(self.expert_dim)
        self.router_fc = nn.Linear(self.expert_dim, self.num_experts)
        
        self.router_ln_patch = nn.LayerNorm(self.win_size)
        self.router_fc_patch = nn.Linear(self.win_size, self.num_experts)

        self.to(self.device)

    def tokenEmb(self, x):
        x = x.unsqueeze(-1)
        y = self.embeddings
        return x * y

    def fourierGC(self, x, B, N, L, win_idx):
        for l in range(self.layer_num):
            w_real = self.weights[win_idx][l][0]
            w_imag = self.weights[win_idx][l][1]
            b_real = self.biases[win_idx][l][0]
            b_imag = self.biases[win_idx][l][1]

            o_real = F.relu(
                torch.einsum('bli,ij->blj', x.real, w_real) - \
                torch.einsum('bli,ij->blj', x.imag, w_imag) + \
                b_real
            )
            o_imag = F.relu(
                torch.einsum('bli,ij->blj', x.imag, w_real) + \
                torch.einsum('bli,ij->blj', x.real, w_imag) + \
                b_imag
            )

            y = torch.stack([o_real, o_imag], dim=-1)
            y = F.softshrink(y, lambd=self.sparsity_threshold)
            x = torch.view_as_complex(y)

        return x

    def _stack_adj_dense(self, train_global_adjs, B, N):
        if isinstance(train_global_adjs, torch.Tensor):
            adj_dense = train_global_adjs
            if hasattr(adj_dense, "is_sparse") and adj_dense.is_sparse:
                adj_dense = adj_dense.to_dense()
        elif isinstance(train_global_adjs, list):
            adjs = []
            for a in train_global_adjs:
                if hasattr(a, "is_sparse") and a.is_sparse:
                    adjs.append(a.to_dense())
                else:
                    adjs.append(a)
            adj_dense = torch.stack(adjs, dim=0)
        else:
            a = train_global_adjs
            if hasattr(a, "is_sparse") and a.is_sparse:
                a = a.to_dense()
            adj_dense = a.unsqueeze(0).expand(B, -1, -1).contiguous()
        return adj_dense.to(self.device)

    def forward(self, win_seq, win_pcc, global_adjs, patch_adjs):
        B, num_window, num_roi, win_size = win_seq.shape
        x = win_seq.reshape(B, num_window, -1)
        x = self.tokenEmb(x)
        f_outs = []
        for i in range(self.num_window):
            temp = x[:, i, :, :]
            temp = temp.squeeze(1)
            temp = torch.fft.rfft(temp, dim=1, norm='ortho')
            temp = temp.reshape(B, (num_roi * win_size) // 2 + 1, self.frequency_size)
            f_out = self.fourierGC(temp, B, num_roi, win_size, i)
            f_out = f_out.reshape(B, (num_roi * win_size) // 2 + 1, self.embed_size)
            f_out = torch.fft.irfft(f_out, n=num_roi * win_size, dim=1, norm="ortho")
            f_outs.append(f_out)
        x = torch.stack(f_outs, dim=1)
        x = x.reshape(B, num_window, num_roi, win_size * self.embed_size)

        pro_list = []
        for i in range(self.num_window):
            temp = x[:, i, :, :]
            temp = temp.reshape(B, num_roi, win_size, self.embed_size)
            w_t = torch.softmax(self.w_t_list[i], dim=0)   
            w_c = torch.softmax(self.w_c_list[i], dim=0)   
            f_tc = (temp * w_c).sum(dim=-1)

            pro_list.append(f_tc)
        freq_feat = torch.stack(pro_list, dim=1) 
        freq_feat = freq_feat.reshape(B, num_window, -1)
        mamba_outs1 = []
        mamba_outs2 = []
        shannon_entropy_list1 = []
        shannon_entropy_list2 = []
        for i in range(B):
            feat_t = win_pcc[i, :, :].mean(dim=-1, keepdim=True).permute(2, 0, 1)

            adj_list = [global_adjs[i] for _ in range(self.num_window)]
            feat_f = freq_feat[i, :, :].unsqueeze(0)

            temp_out1 = self.DGSM1(feat_t, adj_list)
            temp_out2 = self.DGSM2(feat_f, adj_list)
            temp_out1 = self.LayerNorm(temp_out1)
            temp_out2 = self.LayerNorm(temp_out2)
            mamba_outs1.append(temp_out1)
            mamba_outs2.append(temp_out2)
            shannon_loss1 = shannon_entropy(temp_out1)
            shannon_loss2 = shannon_entropy(temp_out2)
            shannon_entropy_list1.append(shannon_loss1)
            shannon_entropy_list2.append(shannon_loss2)
        time_feat = torch.cat(mamba_outs1, dim=0)
        freq_feat2 = torch.cat(mamba_outs2, dim=0)

        loss_avg = torch.stack(shannon_entropy_list1).mean()
        time_feat = time_feat.reshape(B, num_window, num_roi, -1)
        freq_feat = freq_feat2.reshape(B, num_window, num_roi, -1)
        concat = torch.cat([freq_feat, time_feat], dim=-1).permute(0, 2, 1, 3).reshape(B, num_roi, -1)  
        gate = torch.sigmoid(concat @ self.W + self.b)  

        freq_feat = freq_feat.permute(0, 2, 1, 3).reshape(B, num_roi, -1)
        time_feat = time_feat.permute(0, 2, 1, 3).reshape(B, num_roi, -1)
        fusion_feat = gate * freq_feat + (1 - gate) * time_feat  
        num_patch = num_window * num_roi
        patch_feat = fusion_feat.reshape(B, num_patch, -1)  
        patch_adj_dense = self._stack_adj_dense(patch_adjs, B, num_patch)

        fusion_feat = fusion_feat.to(self.device)

        adj_dense = self._stack_adj_dense(global_adjs, B, num_roi)  

        agg = torch.matmul(adj_dense, fusion_feat) 
        router_feat = self.router_ln(agg)           
        logits = self.router_fc(router_feat)        
        probs = F.softmax(logits, dim=-1)          

        patch_agg = torch.matmul(patch_adj_dense, patch_feat)  
        patch_router_feat = self.router_ln_patch(patch_agg)           
        patch_logits = self.router_fc_patch(patch_router_feat)        
        patch_probs = F.softmax(patch_logits, dim=-1)           

        patch_importance = torch.abs(patch_logits).sum(dim=-1) 
        roi_importance = torch.abs(logits).sum(dim=-1)  

        topk_vals, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)  
        mask = torch.zeros_like(probs, device=self.device)
        mask.scatter_(-1, topk_idx, topk_vals)
        sparse_probs = mask  

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            if self.use_gcn_experts:
                out_i = expert(fusion_feat, adj_dense)  
            else:
                Bv, Nv, Dv = fusion_feat.shape
                out_i = expert(fusion_feat.view(-1, Dv))  
                out_i = out_i.view(Bv, Nv, Dv)
            expert_outputs.append(out_i)

        expert_stack = torch.stack(expert_outputs, dim=2)  

        weights = sparse_probs.unsqueeze(-1)  
        grmoe_out = torch.sum(weights * expert_stack, dim=2)  

        expert_usage = probs.sum(dim=(0, 1)) 
        if expert_usage.sum() > 0:
            expert_usage = expert_usage / (expert_usage.sum() + 1e-8)
        else:
            expert_usage = torch.ones_like(expert_usage) / self.num_experts
        target = torch.full_like(expert_usage, 1.0 / self.num_experts)
        lb_loss = F.mse_loss(expert_usage, target)
        aux_loss = self.lb_coef * lb_loss
        
        patch_expert_usage = patch_probs.sum(dim=(0, 1)) 
        if patch_expert_usage.sum() > 0:
            patch_expert_usage = patch_expert_usage / (patch_expert_usage.sum() + 1e-8)
        else:
            patch_expert_usage = torch.ones_like(patch_expert_usage) / self.num_experts

        patch_lb_loss = F.mse_loss(patch_expert_usage, target)

        final_feat = grmoe_out 

        outs = self.f1(final_feat)
        out = self.d1(F.relu(self.bn2(self.l1(outs))))
        out = self.d2(F.relu(self.bn3(self.l2(out))))
        out = self.d3(F.relu(self.bn4(self.l3(out))))
        out = self.l4(out)
        out = F.log_softmax(out, dim=1)

        aux_loss = lb_loss + patch_lb_loss + loss_avg
        return out, aux_loss, roi_importance, patch_importance