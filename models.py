import math
from dgl.nn.pytorch import RelGraphConv
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Callable,Optional
from torch import Tensor


# (predictions, labels, ids/mask) -> Tensor with one element
LOSS_TYPE = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]
EPS = 1e-6

class SelfExplainableRGCN(object):
    def __init__(self, args, device='cpu'):
        super().__init__()
        self.args = args
        self.device = device
        self.encoder = RGCN(args.in_dim, args.h_dim, args.out_dim, args.num_rels).to(self.device)
        self.mlp = MLP(in_dim=args.h_dim, h_dim=args.h_dim, out_dim=1).to(self.device)


    def fit(self, train_dataloader, val_dataloader, test_dataloader, fold):
        optimizer = torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': self.args.lr_outer},
             {'params': self.mlp.parameters(), 'lr': self.args.lr_inner}],)
        criterion = nn.CrossEntropyLoss()
        self.best_val_acc = 0.
        for epoch in range(self.args.epochs_outer+1):
            num_correct_train, num_train  = 0, 0
            epoch_loss, epoch_loss_sp, epoch_loss_cf, epoch_loss_mi = 0, 0, 0, 0
            for batched_graph, labels in train_dataloader:
                embed_graph_all, embed_sub_all = [], []
                self.encoder.train()
                self.mlp.train()
                optimizer.zero_grad()

                # graph embeddings
                batched_graph, labels = batched_graph.to(self.device), labels.to(self.device)
                embed_node = self.encoder(batched_graph, batched_graph.ndata['feat'],
                                          batched_graph.edata['etype'], batched_graph.edata['eweight'])

                # counterfactual explaination
                embed_sub, embed_res, weight_nodes, weight_edges = \
                    self.cf_explain(batched_graph, embed_node)
                embed_sub_all.append(embed_sub)

                # merge the embedding of subgraph and graph
                with batched_graph.local_scope():
                    batched_graph.ndata['h'] = embed_node
                    embed_graph = dgl.mean_nodes(batched_graph, 'h')
                    embed_graph_all.append(embed_graph)

                embed_merge = (embed_graph+embed_sub)/2
                pred_graph, pred_sub, pred_res = self.encoder.fc(embed_merge), self.encoder.fc(embed_sub), self.encoder.fc(embed_res)

                self.encoder.eval()
                self.mlp.eval()
                embed_graph_all, embed_sub_all = torch.cat(embed_graph_all, dim=0), torch.cat(embed_sub_all, dim=0)
                # embed_g, embed_s = embed_graph_all.detach().clone().requires_grad_(
                #     False), embed_sub_all.detach().clone().requires_grad_(False)

                self.mine = MINE(self.args.h_dim)
                self.mine.to(self.device)
                self.mine.train()
                loss_mi = self.mine.optimize(embed_graph_all, embed_sub_all, iters=self.args.epochs_inner)
                self.mine.eval()
                self.encoder.train()
                self.mlp.train()

                # counterfactual loss
                loss_cf = criterion(pred_sub, labels) + criterion(pred_res, torch.zeros_like(labels))
                # sparsity loss
                loss_sp = torch.mean(weight_nodes)+torch.mean(weight_edges)
                # classify loss
                loss_cl = criterion(pred_graph, labels)
                # total loss
                loss = 0.5*loss_cf + 0.2*loss_sp + loss_cl + loss_mi

                epoch_loss += loss.item()
                epoch_loss_sp+=loss_sp.item()
                epoch_loss_cf+=loss_cf.item()
                epoch_loss_mi+=loss_mi.item()
                num_correct_train += (pred_graph.argmax(1) == labels).sum().item()
                num_train += len(labels)
                loss.backward()
                optimizer.step()
            train_accuracy = num_correct_train / num_train
            train_loss = epoch_loss / len(train_dataloader)
            train_loss_sp = epoch_loss_sp / len(train_dataloader)
            train_loss_sub = epoch_loss_cf / len(train_dataloader)
            train_loss_mi = epoch_loss_mi / len(train_dataloader)

            # Validate the model
            if epoch > 0 and epoch%10 == 0: self.valid(val_dataloader, fold, epoch)
            print(f"Epoch {epoch + 1}: loss: {train_loss:.2f}, accuracy: {train_accuracy:.2f}, "
                  f"loss_sp: {train_loss_sp:.2f}, loss_cf: {train_loss_sub:.2f}, "
                  f"loss_mi: {train_loss_mi:.2f}, num_correct: {num_correct_train}, num_total: {num_train}")


    def valid(self, val_dataloader, fold, epoch):
        self.encoder.eval()
        self.mlp.eval()

        with torch.no_grad():
            num_correct_val, num_correct_sub, num_correct_res,num_val = 0, 0, 0, 0
            for batched_graph, labels in val_dataloader:
                batched_graph, labels = batched_graph.to(self.device), labels.to(self.device)
                embed_node = self.encoder(batched_graph, batched_graph.ndata['feat'],
                                          batched_graph.edata['etype'], batched_graph.edata['eweight'])
                embed_sub, embed_res, weight_nodes, weight_edges = self.cf_explain(
                    batched_graph, embed_node)

                # check cf results
                batched_subgraph, batched_resgraph = subgraph_generation(batched_graph, weight_nodes, weight_edges)

                # merge the embedding of subgraph and graph
                with batched_graph.local_scope():
                    batched_graph.ndata['h'] = embed_node
                    embed_graph = dgl.mean_nodes(batched_graph, 'h')
                embed_merge = (embed_graph + embed_sub) / 2
                pred_graph, pred_sub, pred_res = self.encoder.fc(embed_merge), self.encoder.fc(
                    embed_sub), self.encoder.fc(embed_res)

                num_correct_sub += (pred_sub.argmax(1) == labels).sum().item()
                num_correct_res += (pred_res.argmax(1) == torch.zeros_like(labels)).sum().item()
                num_correct_val += (pred_graph.argmax(1) == labels).sum().item()
                num_val += len(labels)
        val_acc, val_acc_sub, val_acc_res = num_correct_val / num_val, num_correct_sub / num_val, num_correct_res / num_val
        print(
            f"Validation: accuracy: {val_acc:.2f}, subgraph accuracy: {val_acc_sub:.2f}, resgraph accuracy: {val_acc_res:.2f}")
        print(
            f"Nodes: subgraph/resgraph/graph: {batched_subgraph.num_nodes()}/{batched_resgraph.num_nodes()}/{batched_graph.num_nodes()}")
        print(
            f"Edges: subgraph/resgraph/graph: {batched_subgraph.num_edges()}/{batched_resgraph.num_edges()}/{batched_graph.num_edges()}")
        # Save the best model for this fold
        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
        best_model_state, best_mlp_state = self.encoder.state_dict(), self.mlp.state_dict()
        torch.save({'encoder':self.encoder.state_dict(), 'mlp':self.mlp.state_dict()},
                   './trained_model/best_' + str(fold + 1) + '_' + str(epoch) + '.pt')


    def test(self, test_dataloader):
        self.encoder.eval()
        self.mlp.eval()
        all_preds, all_labels, sub_preds, graph_size, sub_size = [], [], [], 0, 0
        num_correct_test, num_correct_sub, num_correct_res, num_test = 0, 0, 0, 0
        with torch.no_grad():
            for batched_graph, labels in test_dataloader:
                if labels.size() != torch.Size([1]): labels = labels.squeeze()
                # batched_graph = trans_graph(batched_graph)
                if 'eweight' not in batched_graph.edata.keys(): batched_graph.edata['eweight'] = torch.ones_like(batched_graph.edata['etype'])
                batched_graph, labels = batched_graph.to(self.device), labels.to(self.device)
                embed_node = self.encoder(batched_graph, batched_graph.ndata['feat'],
                                          batched_graph.edata['etype'], batched_graph.edata['eweight'])
                embed_sub, embed_res, weight_nodes, weight_edges = self.cf_explain(batched_graph, embed_node)

                # check cf results
                batched_subgraph, batched_resgraph = subgraph_generation(batched_graph, weight_nodes, weight_edges)

                # merge the embedding of subgraph and graph
                with batched_graph.local_scope():
                    batched_graph.ndata['h'] = embed_node
                    embed_graph = dgl.mean_nodes(batched_graph, 'h')
                embed_merge = (embed_graph+embed_sub)/2
                pred_graph, pred_sub, pred_res = self.encoder.fc(embed_merge), self.encoder.fc(embed_sub), self.encoder.fc(embed_res)

                num_test += len(labels.tolist())
                all_preds.extend(pred_graph.argmax(1).tolist())
                sub_preds.extend(pred_sub.argmax(1).tolist())
                all_labels.extend(labels.tolist())
                sub_size+=batched_subgraph.num_nodes()
                graph_size+=batched_graph.num_nodes()


        # Calculate metrics for this fold
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')

        precision_sub = precision_score(all_labels, sub_preds, average='binary', zero_division=0)
        recall_sub = recall_score(all_labels, sub_preds, average='binary')
        f1_sub = f1_score(all_labels, sub_preds, average='binary')

        print(f"Test Results: Precision: {precision:.2f}, Recall: {recall:.2f}, F1_binary: {f1:.2f}")
        print(f"Subgraph Results: Precision: {precision_sub:.2f}, Recall: {recall_sub:.2f}, "
              f"F1_binary: {f1_sub:.2f}, Avg.GraphSize: {graph_size/num_test:.2f}, Avg.SubSize: {sub_size/num_test:.2f}")
        return precision, recall, f1, precision_sub, recall_sub, f1_sub


    def cf_explain(self, batched_graph, embed_node):
        # weights of nodes and edges
        transfer_mat = consturct_transfer_matrix_sparse(batched_graph)
        embed_edge = torch.sparse.mm(transfer_mat, embed_node)
        weight_edges = self.mlp(embed_edge).squeeze()
        weight_nodes = self.mlp(embed_node)

        # weight_edges = torch.abs(weight_edges-.5*torch.ones_like(weight_edges))
        # weight_nodes = torch.abs(weight_nodes-.5*torch.ones_like(weight_nodes))

        # subgraph and resgraph classification
        embed_sub = self.encoder(batched_graph, weight_nodes * batched_graph.ndata['feat'],
                                 batched_graph.edata['etype'], weight_edges)
        embed_res = self.encoder(batched_graph, (1 - weight_nodes) * batched_graph.ndata['feat'],
                                 batched_graph.edata['etype'],
                                 torch.ones_like(weight_edges) - weight_edges)
        weight_nodes = weight_nodes.squeeze()
        with batched_graph.local_scope():
            batched_graph.ndata['h'] = embed_sub
            embed_sub = dgl.mean_nodes(batched_graph, 'h')
        with batched_graph.local_scope():
            batched_graph.ndata['h'] = embed_res
            embed_res = dgl.mean_nodes(batched_graph, 'h')
        # return embed_sub, pred_sub, pred_res, weight_nodes, weight_edges
        return embed_sub, embed_res, weight_nodes, weight_edges


class RGCN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels):
        super().__init__()
        # Two-layer RGCN
        self.conv1 = RelGraphConv(
            in_dim,
            h_dim,
            num_rels,
            regularizer="basis",
            num_bases=num_rels,
            self_loop=False,
        )
        self.conv2 = RelGraphConv(
            h_dim,
            h_dim,
            num_rels,
            regularizer="basis",
            num_bases=num_rels,
            self_loop=False,
        )
        self.dropout = 0.5
        self.fc = nn.Linear(h_dim, out_dim)

    def forward(self, g, features, etypes, edge_weight):
        if edge_weight is not None:
            # Normalize edge weights if needed
            edge_weight = edge_weight.unsqueeze(1)  # [E, 1]
            g.edata['weight_scaler'] = edge_weight

        emb = self.conv1(g, features, etypes, norm=edge_weight)
        emb = F.relu(emb)
        emb = self.conv2(g, emb, etypes, norm=edge_weight)
        emb = F.dropout(emb, self.dropout, training=self.training)
        return emb


class MLP(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_layers=2, dropout=0.5):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.ff_bias = True  # Use bias for FF layers in default

        self.bns = nn.BatchNorm1d(h_dim, affine=False, track_running_stats=False)
        self.activation = F.relu

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_dim, h_dim, bias=self.ff_bias))
        for _ in range(self.num_layers - 2): self.fcs.append(
            nn.Linear(h_dim, h_dim, bias=self.ff_bias))  # 1s
        self.fcs.append(nn.Linear(h_dim, out_dim, bias=self.ff_bias))  # 1
        self.reset_parameters()

    def reset_parameters(self):
        for mlp in self.fcs:
            nn.init.xavier_uniform_(mlp.weight, gain=1.414)
            nn.init.zeros_(mlp.bias)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = x @ self.fcs[i].weight.t()
            if self.ff_bias: x = x + self.fcs[i].bias
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x @ self.fcs[-1].weight.t()
        if self.ff_bias: x = x + self.fcs[-1].bias
        x = F.sigmoid(x)
        return x


class MINE(nn.Module):
    def __init__(self, dim, loss='fdiv', alpha=0.01):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.T = nn.Sequential(nn.Linear(dim + dim, 100),
                               nn.ReLU(), nn.Linear(100, 100),
                               nn.ReLU(), nn.Linear(100, 1))

    def forward(self, x, z, z_marg=None):
        z_marg = z[torch.randperm(x.shape[0])]
        t = self.T(torch.concat([x,z], dim=1)).mean()
        t_marg = self.T(torch.concat([x,z_marg], dim=1))

        if self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()-1
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])
        return -t + second_term

    def optimize(self, X, Y, iters, batch_size=16):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        for iter in range(1, iters + 1):
            mu_mi = 0
            for x, y in batch(X, Y, batch_size):
                opt.zero_grad()
                loss = self.forward(x, y)
                loss.backward()
                opt.step()

                mu_mi -= loss.item()
            # if iter % (iters // 3) == 0: print(f"It {iter} - MI: {mu_mi / batch_size}")

        with torch.no_grad():
            final_mi = -self.forward(X, Y)
        # print(f"Final MI: {final_mi}")
        return final_mi
