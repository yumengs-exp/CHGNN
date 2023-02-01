import copy

import torch
import torch.nn as nn, torch.nn.functional as F
import numpy as np

import math

from torch_scatter import scatter
from torch_geometric.nn import HypergraphConv
from aug import aug_node
from torch_geometric.utils import subgraph


def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().sum(dim=1,keepdims=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X

def delete_hyperedge(reserve_id, H):
    reserve_id = reserve_id.detach().numpy()
    H_size = H.size(1)
    H = H.detach().numpy()
    new_H = [[],[]]
    for i in range(H_size):
        if H[1,i] not in reserve_id:
            new_H[0].append(H[0][i])
            new_H[1].append(H[1][i])

    new_H = torch.Tensor(new_H)
    return new_H
# HYPEREDGE ViewGenerator
class HypergraphViewGenerator(torch.nn.Module):
    def __init__(self,in_dim,out_dim, head = 1, dropout = 0.6):
        super().__init__()

        self.encoder = nn.ModuleList([HypergraphConv(in_dim,in_dim//2,head=head, droupout = dropout), HypergraphConv(in_dim//2,out_dim,head=head,dropout=dropout)])


    def forward(self,X_origin,edge_index_origin, overlappness,args):
        X = copy.deepcopy(X_origin)
        edge_index = copy.deepcopy(edge_index_origin)
        edge_index = edge_index.long()


        sel_mask = aug_node(overlappness,args)

        for m in self.encoder:
            X = m(X,edge_index)
        Xve = X[edge_index[0]]
        Xe = scatter(Xve, edge_index[1], dim = 0,reduce = 'mean') #|E|*3
        Xe = normalize_l2(F.sigmoid(Xe))
        if args.dataset in ['dblp']:
            # Xe = torch.nn.functional.softmax(Xe, dim=1)
            Xe[:, 0] += 10
        sample = F.gumbel_softmax(Xe,hard=True)

        reserve = sample[:,0]
        delete = sample[:,1]
        mask = sample[:,2]

        reserve_sample = reserve.bool()[edge_index[1].long()]
        mask_sample = torch.logical_and(mask.bool()[edge_index[1]],sel_mask[edge_index[0]])
        sample = reserve_sample | mask_sample

        edge_index = edge_index[:,sample]

        # E = E * (reserve+mask)
        # H_aug = copy.deepcopy(H_origin)
        # H_aug = aug_node(H_aug,overlappness,args)
        #
        # H= delete_hyperedge(reserve_id,H)##get sub hypergraph
        # H[mask] = H_aug[mask.bool()]

        return sample.float(),edge_index






















