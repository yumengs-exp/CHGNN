import torch
import torch.nn as nn, torch.nn.functional as F

import torch_sparse
import math

from torch_scatter import scatter
from torch_geometric.utils import softmax
# from kmeans_pytorch import kmeans



def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def normalize_l2(X):
    """Row-normalize  matrix"""
    rownorm = X.detach().norm(dim=1, keepdim=True)
    scale = rownorm.pow(-1)
    scale[torch.isinf(scale)] = 0.
    X = X * scale
    return X



# v1: X -> XW -> AXW -> norm
class Conv(nn.Module):

    def __init__(self, args, in_channels, out_channels, heads=8, dropout=0., negative_slope=0.2):
        super().__init__()
        # TODO: bias?
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.args = args

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

    def forward(self, X, vertex, edges, homo):
        N = X.shape[0]

        # X0 = X # NOTE: reserved for skip connection

        X = self.W(X)

        Xve = X[vertex] # [nnz, C]
        Xe = scatter(Xve, edges, dim=0, reduce=self.args.first_aggregate) # [E, C]


        Xev = Xe[edges] # [nnz, C]

        att_sum = scatter(homo[edges], vertex, dim=0, reduce='sum')
        att_sum = att_sum[vertex]
        att = (homo[edges] / att_sum).unsqueeze(1)
        att = att.mm(torch.ones(att.shape[1],Xev.shape[1]).cuda())

        att_Xev = att.mul(Xev)


        Xv = scatter(att_Xev, vertex, dim=0, reduce=self.args.second_aggregate, dim_size=N) # [N, C]
        X = X + Xv

        if self.args.use_norm:
            X = normalize_l2(X)

        # NOTE: concat heads or mean heads?
        # NOTE: normalize here?
        # NOTE: skip concat here?

        return X






class HHyperGNN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, nlayer, nhead, nproj,homogeneity,tau,epcc,epec):
        """

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
            V (torch.long): V is the row index for the sparse incident matrix H, |V| x |E|
            E (torch.long): E is the col index for the sparse incident matrix H, |V| x |E|
        """
        super().__init__()
        self.conv_out = Conv(args, nhid * nhead, nclass, heads=1, dropout=args.attn_drop)
        self.convs = nn.ModuleList(
            [ Conv(args, nfeat, nhid, heads=nhead, dropout=args.attn_drop)] +
            [Conv(args, nhid * nhead, nhid, heads=nhead, dropout=args.attn_drop) for _ in range(nlayer-2)]
        )



        self.fc1 = torch.nn.Linear(nhid * nhead, nproj)
        self.fc2 = torch.nn.Linear(nproj, nhid * nhead)

        self.fc3 = torch.nn.Linear(nclass, nproj)
        self.fc4 = torch.nn.Linear(nproj, nclass)

        self.reg = torch.nn.Linear(nhid*nhead,1)
        self.nc_disc =torch.nn.Bilinear(nhid*nhead,nclass,1)
        self.ne_disc = torch.nn.Bilinear(nhid*nhead,nhid*nhead,1)
        self.ec_disc = torch.nn.Bilinear(nclass,nhid*nhead,1)

        self.homo = homogeneity
        self.tau = tau
        self.epcc = epcc
        self.epec = epec

        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act[args.activation]
        self.input_drop = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)


    def forward(self, X,V,E):


        homo = self.homo
        # homo = torch.ones(self.homo.shape[0]).cuda()
        X = self.input_drop(X)

        for conv in self.convs:
            X = conv(X, V, E, homo)
            X = self.act(X)
            X= self.dropout(X)


        Z = X

        X = self.conv_out(X, V, E, homo)


        return Z,X

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def projection_cluster(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc3(z))
        return self.fc4(z)


    def sim(self, z1:torch.Tensor, z2:torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1,z2.t())

    def cal_loss(self,z1:torch.Tensor, z2:torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1,z1))
        between_sim = f(self.sim(z1,z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss_cls(self, Z1, Z2, Y, train_idx):
        Z1 = F.log_softmax(Z1, dim=1)
        Z2 = F.log_softmax(Z2, dim=1)

        self.c1 = Z1
        self.c2 = Z2

        loss_cls = F.nll_loss(((Z1+Z2)/2)[train_idx], Y[train_idx])
        return loss_cls

    def loss_node(self,Z1,Z2):

        h1 = self.projection(Z1)
        h2 = self.projection(Z2)
        l1 = self.cal_loss(h1, h2)
        l2 = self.cal_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret

    def loss_node_ada_maxmargin(self,Z1,Z2):

        h1 = self.projection(Z1)
        h2 = self.projection(Z2)
        l1 = self.cal_loss(h1, h2)
        l2 = self.cal_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret

    def loss_cluster(self,X1,X2):


        h1 = self.projection_cluster(X1)
        h2 = self.projection_cluster(X2)

        h1 = h1.T
        h2 = h2.T
        l1 = self.cal_loss(h1, h2)
        l2 = self.cal_loss(h2, h1)

        # l1 = self.cal_loss_maxmagin(h1, h2)
        # l2 = self.cal_loss_maxmagin(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret

    def loss_cluster_maxmargin(self,X1,X2):


        h1 = self.projection_cluster(X1)
        h2 = self.projection_cluster(X2)

        h1 = h1.T
        h2 = h2.T


        l1 = self.cal_loss_maxmagin(h1, h2)
        l2 = self.cal_loss_maxmagin(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret

    def loss_cluster_ada_maxmargin(self,X1,X2,Y, train_idx):


        h1 = self.projection_cluster(X1)
        h2 = self.projection_cluster(X2)

        train_node = h1[train_idx]
        train_label = Y[train_idx]
        cnum = len(Y.unique())
        signal = scatter(train_node, train_label, dim=0, reduce='mean',dim_size=cnum)
        sim = self.sim(signal,signal)
        sim = sim - torch.diag(sim)
        sim = -sim.mean()
        h1 = h1.T
        h2 = h2.T
        if sim.cpu().item() >self.epec:
            l1 = self.cal_loss_maxmagin(h1, h2)
            l2 = self.cal_loss_maxmagin(h2, h1)
            # print('cluster')
        else:
            l1 = self.cal_loss(h1, h2)
            l2 = self.cal_loss(h2, h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()


        return ret


    def cal_loss_maxmagin(self, z1:torch.Tensor, z2:torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(torch.pow(self.sim(z1,z1),2))
        between_sim = f(torch.pow(self.sim(z1,z2),2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))



    def loss_edgehomo(self,Z,V,E,E_origin):
        Ze = Z[V]
        Ze = scatter(Ze,E,dim=0,reduce='mean',dim_size=E_origin.max()+1)

        # reg = self.disc(Ze[E], Z[V])

        logit = self.reg(Ze)
        homo  = self.homo.view(-1,1)
        loss = F.mse_loss(logit,homo)

        return loss



    def loss_hyperedge(self,Z1,Z2,V1,E1,V2,E2,E):
        Ze1 = Z1[V1] # [nnz, C]
        Ze1 = scatter(Ze1, E1, dim=0, reduce='mean',dim_size = E.max()+1) # [E, C]

        Ze2 = Z2[V2] # [nnz, C]
        Ze2 = scatter(Ze2, E2, dim=0, reduce='mean',dim_size = E.max()+1) # [E, C]

        h1 = self.projection(Ze1)
        h2 = self.projection(Ze2)
        l1 = self.cal_loss(h1,h2)
        l2 = self.cal_loss(h2,h1)
        # l1 = self.cal_loss_maxmagin(h1, h2)
        # l2 = self.cal_loss_maxmagin(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret


    def loss_hyperedge_maxmargin(self,Z1,Z2):
        Ze1 = Z1[self.H1_V] # [nnz, C]
        Ze1 = scatter(Ze1, self.H1_E, dim=0, reduce='mean',dim_size = self.E.max()+1) # [E, C]

        Ze2 = Z2[self.H2_V] # [nnz, C]
        Ze2 = scatter(Ze2, self.H2_E, dim=0, reduce='mean',dim_size = self.E.max()+1) # [E, C]

        h1 = self.projection(Ze1)
        h2 = self.projection(Ze2)
        # l1 = self.cal_loss(h1,h2)
        # l2 = self.cal_loss(h2,h1)
        l1 = self.cal_loss_maxmagin(h1, h2)
        l2 = self.cal_loss_maxmagin(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret

    def loss_hyperedge_ada_maxmargin(self,Z1,Z2,Y,train_idx,V1,V2,E1,E2,E):
        Ze1 = Z1[V1] # [nnz, C]
        Ze2 = Z2[V2] # [nnz, C]

        train_node = Z1[train_idx]
        train_label = Y[train_idx]
        cnum = len(Y.unique())
        signal = scatter(train_node, train_label, dim=0, reduce='mean',dim_size=cnum)
        signal = self.projection(signal)
        sim = self.sim(signal,signal)
        sim = sim - torch.diag(sim)
        sim = -sim.mean()

        Ze1 = scatter(Ze1, E1, dim=0, reduce='mean',dim_size = E.max()+1) # [E, C]
        Ze2 = scatter(Ze2, E2, dim=0, reduce='mean',dim_size = E.max()+1) # [E, C]
        h1 = self.projection(Ze1)
        h2 = self.projection(Ze2)

        if sim.cpu().item() >self.epec:
            l1 = self.cal_loss_maxmagin(h1, h2)
            l2 = self.cal_loss_maxmagin(h2, h1)
            # print('edge')
        else:
            l1 = self.cal_loss(h1,h2)
            l2 = self.cal_loss(h2,h1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()
        return ret




    def test(self,X,V,E):

        homo = self.homo
        X = self.input_drop(X)

        for conv in self.convs:
            X = conv(X, V, E, homo)
            X = self.act(X)
            X= self.dropout(X)

        X = self.conv_out(X, V, E, homo)
        return F.log_softmax(X, dim=1)

    def loss_crosscl_nc(self,Z,X,tau=0.5 ):
        pos = torch.exp(torch.sigmoid(self.nc_disc(Z,X))/tau)
        neg_X = torch.exp(torch.sigmoid(self.nc_disc(Z,X[torch.randperm(X.size(0))]))/tau)
        neg_Z = torch.exp(torch.sigmoid(self.nc_disc(Z[torch.randperm(Z.size(0))],X))/tau)


        loss_X = -torch.log(pos/(pos+neg_X))
        loss_Z = -torch.log(pos/(pos+neg_Z))

        loss_X = loss_X[~torch.isnan(loss_X)]
        loss_Z = loss_Z[~torch.isnan(loss_Z)]

        return (loss_X+loss_Z).mean()

    def loss_crosscl_ne(self,Z,V,E,E_origin,tau=0.5):
        Ze = Z[V]
        Ze = scatter(Ze, E, dim=0, reduce='mean', dim_size=E_origin.max() + 1)

        neg_Ze = Ze[torch.randperm(Ze.size(0))]
        neg_Z = Z[torch.randperm(Z.size(0))]

        pos = torch.exp(torch.sigmoid(self.ne_disc(Z[V], Ze[E]))/tau)
        neg_E = torch.exp(torch.sigmoid(self.ne_disc(Z[V], neg_Ze[E]))/ tau)
        neg_V = torch.exp(torch.sigmoid(self.ne_disc(neg_Z[V], Ze[E])) / tau)

        loss_E = -torch.log(pos / (pos + neg_E))
        loss_V = -torch.log(pos / (pos + neg_V))

        loss_V = loss_V[~torch.isnan(loss_V)]
        loss_E = loss_E[~torch.isnan(loss_E)]

        return (loss_V + loss_E).mean()

    def loss_crosscl_ec(self,X,Z, V,E,E_origin,tau=0.5):
        Ze = Z[V]
        Ze = scatter(Ze, E, dim=0, reduce='mean', dim_size=E_origin.max() + 1)

        neg_Ze = Ze[torch.randperm(Ze.size(0))]
        neg_X = X[torch.randperm(X.size(0))]

        pos = torch.exp(torch.sigmoid(self.ec_disc(X[V], Ze[E]))/tau)
        neg_E = torch.exp(torch.sigmoid(self.ec_disc(X[V], neg_Ze[E]))/ tau)
        neg_X = torch.exp(torch.sigmoid(self.ec_disc(neg_X[V], Ze[E])) / tau)

        loss_E = -torch.log(pos / (pos + neg_E))
        loss_X = -torch.log(pos / (pos + neg_X))

        loss_X = loss_X[~torch.isnan(loss_X)]
        loss_E = loss_E[~torch.isnan(loss_E)]

        return (loss_X + loss_E).mean()








