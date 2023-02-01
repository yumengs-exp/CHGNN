from model.HyperGNN import HyperGNN
import torch, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
from aug import aug_node,aug_edge
from torch_scatter import scatter
from model.Generator import Generator
from model.HHyperGNN import HHyperGNN
import os
def accuracy(Z, Y):
    return 100 * Z.argmax(1).eq(Y).float().mean().item()


import torch_sparse


def fetch_data(args):
    from data import data
    dataset = data.load_dataset(args)
    args.dataset_dict = dataset

    overlappness = data.load_overlappness(args)
    homogeneity = data.load_homogeneity(args)

    X, Y, G = dataset['features'], dataset['labels'], dataset['hypergraph']

    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
    X = torch.FloatTensor(np.array(X.todense()))

    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])

    # print(os.environ["CUDA_VISIBLE_DEVICES"])

    X, Y = X.cuda(), Y.cuda()
    homogeneity = homogeneity.cuda()

    return X, Y, G, overlappness,homogeneity


# def initialise(X, Y, G, overlappness, args, unseen=None):
#
#
#     G = G.copy()
#
#     if unseen is not None:
#         unseen = set(unseen)
#         # remove unseen nodes
#         for e, vs in G.items():
#             G[e] = list(set(vs) - unseen)
#
#
#
#     N, M = X.shape[0], len(G)
#     indptr, indices, data = [0], [], []
#     for e, vs in G.items():
#         indices += vs
#         data += [1] * len(vs)
#         indptr.append(len(indices))
#     H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr()  # V x E
#
#     Gnfeat = X.shape[1]
#     Gclass = 3 # drop,revserve,mask unimportant nodes
#     Ghid = args.nhid
#     Glayer = args.nlayer
#
#     H1 = Generator(args,Gnfeat, Ghid, Gclass, Glayer)
#
#     H2 = Generator(args,Gnfeat, Ghid, Gclass, Glayer)
#
#     V,E = getinput(H)
#
#
#     nfeat, nclass = X.shape[1], len(Y.unique())
#     nlayer = args.nlayer
#     nhid = args.nhid
#     nhead = args.nhead
#     nproj = args.nproj
#     tau = args.tau
#     epcc = args.epcc
#     epec = args.epec
#
#
#     model = HyperGNN(args, nfeat, nhid, nclass, nlayer, nhead, nproj, H1_V, H1_E, H2_V,H2_E, V,E, homo,tau,epcc,epec)
#     optimiser = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
#
#     model.cuda()
#
#     return model, optimiser



def initialise(X, Y, G):


    G = G.copy()





    N, M = X.shape[0], len(G)
    indptr, indices, data = [0], [], []
    for e, vs in G.items():
        indices += vs
        data += [1] * len(vs)
        indptr.append(len(indices))
    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr()  # V x E
    V, E = getinput(H)
    H = sparse_mx_to_torch_sparse_tensor(H).to_dense()
    f = '/nfs/srv/data2/yumengs/code/CHGNN_copy/CHGNN/aug/origin'
    torch.save(H.numpy(),f)
    H = H.bool().cuda()


    return H,V,E

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    #先转换成coo_matrix形式的矩阵，因为Torch支持COO（rdinate）格式的稀疏张量
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    #np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
    #这一句是提取稀疏矩阵的非零元素的索引。得到的矩阵是一个[2, 8137]的tensor。
    #其中第一行是行索引，第二行是列索引。每一列的两个值对应一个非零元素的坐标。
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))#numpy转成torch
    #这两行就是规定了数值和shape。没什么好说的。
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
    #该函数的返回值的类型是 torch.Tensor

def normalise(M):


    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)


def getinput(H):


    (V,E), value = torch_sparse.from_scipy(H)

    return V.cuda(),E.cuda()
