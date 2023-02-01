from data import parser
import torch, numpy as np, scipy.sparse as sp
import os, inspect, random, pickle
import numpy as np, scipy.sparse as sp
from tqdm import tqdm
import pickle

def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M
    where D is the diagonal node-degree matrix
    """

    d = np.array(M.sum(1))

    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)  # D inverse i.e. D^{-1}

    return DI.dot(M)



def sigmoid(z):
    return 1/(1 + np.exp(-z))

def cal_overlappness(nlist, G:dict):
    '''

    :param nlist: node list
    :param G: hyperedges
    :return: tensor[nlist]
    '''

    edgedict = dict()
    for i in nlist:
        edgedict[i]=[]

    for hyperedge in G.keys():
        e = G[hyperedge]
        for node in e:
            edgedict[node] += e

    overlappness = []
    for E in edgedict.keys():
        subgraph = edgedict[E]
        subgraph_set = set(subgraph)
        up = len(subgraph)
        down = len(subgraph_set)

        if up!=0 and down!=0:
            overlappness.append(up/down)
        else:
            overlappness.append(0)


    overlappness = torch.Tensor(overlappness)

    return overlappness

def cal_degree_of_each_pair(nlist,G:dict):
    '''

    :param nlist: node list
    :param G: hyperedges
    :return: tensor[nlist,nlist]
    '''

    degree_matrix = torch.zeros(len(nlist),len(nlist))

    for hyperedge in G.keys():
        e = G[hyperedge]
        for node_i in e:
            for node_j in e:
                degree_matrix[node_i,node_j] = degree_matrix[node_i,node_j] + 1 #对角线是节点度，其余是节点对的度

    return degree_matrix

def cal_homogeneity_hyperedge(nlist,G:dict,degree_matrix:torch.Tensor):
    '''

    :param nlist: node list
    :param G: hyperedges
    :param pair_degree: degree of each pair
    :return: tensor[|G|]
    '''

    homogeneity = []
    for hyperedge in G.keys():
        e = G[hyperedge]
        if len(e)>1:
            homo = 0
            for node_i in e:
                for node_j in e:
                    if node_i!=node_j:
                        homo = homo + degree_matrix[node_i,node_j].item()

            homo = homo/(len(e)*(len(e)-1))
            homogeneity.append(sigmoid(homo))

        else:
            homogeneity.append(1)

    homogeneity = torch.torch.Tensor(homogeneity)

    return homogeneity

def cal_homogeneity_hyperedge_with_self_loop(nlist,G:dict,degree_matrix:torch.Tensor):
    '''

    :param nlist: node list
    :param G: hyperedges
    :param pair_degree: degree of each pair
    :return: tensor[|G|]
    '''




    Vs = set(range(len(nlist)))

    # only add self-loop to those are orginally un-self-looped
    # TODO:maybe we should remove some repeated self-loops?
    for edge, nodes in G.items():
        if len(nodes) == 1 and nodes[0] in Vs:
                Vs.remove(nodes[0])

    for v in Vs:
        G[f'self-loop-{v}'] = [v]

    homogeneity = []
    for hyperedge in G.keys():
        e = G[hyperedge]
        if len(e)>1:
            homo = 0
            for node_i in e:
                for node_j in e:
                    if node_i!=node_j:
                        homo = homo + degree_matrix[node_i,node_j].item()

            homo = homo/(len(e)*(len(e)-1))
            homogeneity.append(sigmoid(homo))

        else:
            homogeneity.append(1)

    homogeneity = torch.torch.Tensor(homogeneity)

    return homogeneity


if __name__ == '__main__':

    #coauthorship/cocitation/hypergraph
    data = 'hypergraph'

    #cora/dblp for coauthorship, cora/citeseer/pubmed for cocitation, 20news,Mushroom,NTU2012,ModelNet40
    dnames = ['20newsW100', 'ModelNet40', 'NTU2012', 'Mushroom']
    for dataset_name in dnames:
        dataset = parser(data, dataset_name).parse()
        X, Y, G = dataset['features'], dataset['labels'], dataset['hypergraph']
        nlist = np.arange(len(Y))

        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        dir = os.path.join(current, data, dataset_name)
        f_overlappness = os.path.join(dir, 'overlappness')

        print('overlapness')
        overlappness = cal_overlappness(nlist, G)
        torch.save(overlappness, f_overlappness)

        print('homo')
        degree_matrix = cal_degree_of_each_pair(nlist, G)
        homogeneity = cal_homogeneity_hyperedge(nlist, G, degree_matrix)
        f_homogeneity = os.path.join(dir, 'homogeneity')
        torch.save(homogeneity, f_homogeneity)

    # f_homogeneity = os.path.join(dir,'homogeneity_selfloop')
    # torch.save(homogeneity_self_loop,f_homogeneity)


