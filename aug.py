import torch
import torch, numpy as np, scipy.sparse as sp
import torch_sparse
from torch_scatter import scatter
from model.Generator import Generator
# node dropping

def aug_node(overlappness,args):


    cut_off = args.cut_off_node
    p = args.node_dropping_rate

    weights = overlappness.clone()

    #random (delete)
    # weights = torch.ones(weights.shape[0])

    weights = (weights.max() -weights) / (weights.max()-weights.mean())
    if p<0. or p>1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    weights = weights * p
    weights = weights.where(weights<cut_off, torch.ones_like(weights)*cut_off)
    sel_mask = ~torch.bernoulli(1. - weights).to(torch.bool).cuda()



    return sel_mask




#edge_perturbation

def aug_edge(H,overlappness,args):

    cut_off = args.cut_off_edge
    p = args.edge_perturbation_rate

    (V, E), value = torch_sparse.from_scipy(H)
    edge_weight = scatter(overlappness[V],E, dim=0, reduce='mean')
    edge_weight = torch.log(edge_weight)

    #random (delete)
    # edge_weight = torch.ones(edge_weight.shape[0])


    edge_weight = (edge_weight.max()-edge_weight)/(edge_weight.max()-edge_weight.mean())

    if p<0. or p>1.:
        raise ValueError('Dropout probability has to be between 0 and 1, '
                         'but got {}'.format(p))

    weights = edge_weight * p
    weights = weights.where(weights<cut_off, torch.ones_like(weights)*cut_off)
    sel_mask = ~torch.bernoulli(1. - weights).to(torch.bool)
    H = H.toarray()
    H = H.T
    H[sel_mask]  = np.zeros(H.shape[1])
    H = H.T
    H = sp.csr_matrix(H)



    return H

def aug(args, H,X):
    nfeat, nclass = X.shape[1], 3
    nlayer = args.nlayer
    nhid = args.nhid

    model = Generator(nfeat, nhid, nclass, nlayer)


def view_generator(H,X,args):
    pass
