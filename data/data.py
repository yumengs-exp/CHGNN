import os, inspect, random, pickle
import numpy as np, scipy.sparse as sp
from tqdm import tqdm
import pickle
import torch


def load_dataset(args):
    """
    parses the dataset
    """
    dataset = parser(args.data, args.dataset).parse()

    return dataset

def fewlabel(num,train_idx,Y):
    count = [0,0,0,0,0,0,0]
    new = []
    for i in train_idx:
        label = Y[i].cpu().item()
        if count[label]<num:
            new.append(i)
            count[label] +=1

    return new





def load_split(args):

    current = os.path.abspath(inspect.getfile(inspect.currentframe()))
    Dir, _ = os.path.split(current)
    file = os.path.join(Dir, args.data, args.dataset, "splits", str(args.split) + ".pickle")

    if not os.path.isfile(file): print("split + ", str(args.split), "does not exist")
    with open(file, 'rb') as H:
        Splits = pickle.load(H)
        train, test = Splits['train'], Splits['test']

    return train, test

def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True, balance=True):
    """ Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks"""
    """ randomly splits label into train/valid/test splits """
    if not balance:
        if ignore_negative:
            labeled_nodes = torch.where(label != -1)[0]
        else:
            labeled_nodes = label

        n = labeled_nodes.shape[0]
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        perm = torch.as_tensor(np.random.permutation(n))

        train_indices = perm[:train_num]
        val_indices = perm[train_num:train_num + valid_num]
        test_indices = perm[train_num + valid_num:]

        if not ignore_negative:
            return train_indices, val_indices, test_indices

        train_idx = labeled_nodes[train_indices]
        valid_idx = labeled_nodes[val_indices]
        test_idx = labeled_nodes[test_indices]

        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}
    else:
        #         ipdb.set_trace()




        indices = []

        for i in range(label.max() + 1):
            index = torch.where((label == i))[0].view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        percls_trn = round(float(train_prop/(label.max()+1)*len(label)))

        # percls_trn = 20
        val_lb = int(valid_prop*len(label))
        train_idx = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_idx = rest_index[:val_lb]
        test_idx = rest_index[val_lb:]
        split_idx = {'train': train_idx,
                     'valid': valid_idx,
                     'test': test_idx}


    return train_idx,test_idx



def split_train_idx(train_idx,Y):
    label = Y[train_idx]
    cnum = len(Y.unique())
    class_id = []
    for i in range(cnum):
        class_id.append([])
    for i in range(len(train_idx)):
        index = train_idx[i]
        idx_label = label[i]
        class_id[idx_label].append(index)
    return class_id


def load_overlappness(args):

    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    dir = os.path.join(current, args.data, args.dataset)
    f_overlappness  = os.path.join(dir,'overlappness')

    overlappness = torch.load(f_overlappness)
    return overlappness

def load_homogeneity(args):

    current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    dir = os.path.join(current, args.data, args.dataset)


    f_homogeneity = os.path.join(dir,'homogeneity')

    homogeneity = torch.load(f_homogeneity)
    return homogeneity

def get_class(Y,train_idx):
    inter_class_mask = [[] for i in range(Y.max()+1)]
    for i in range(train_idx.size()[0]):
        index = train_idx[i]
        c = Y[index]
        inter_class_mask[c].append(index.item())
    for i in range(len(inter_class_mask)):
        inter_class_mask[i] = torch.LongTensor(inter_class_mask[i])

    return inter_class_mask


class parser(object):
    """
    an object for parsing data
    """
    
    def __init__(self, data, dataset):

        
        current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        self.d = os.path.join(current, data, dataset)
        self.data, self.dataset = data, dataset

    

    def parse(self):
        """
        returns a dataset specific function to parse
        """
        
        name = "_load_data"
        function = getattr(self, name, lambda: {})
        return function()



    def _load_data(self):

        
        with open(os.path.join(self.d, 'hypergraph.pickle'), 'rb') as handle:
            hypergraph = pickle.load(handle)
            # print("number of hyperedges is", len(hypergraph))

        with open(os.path.join(self.d, 'features.pickle'), 'rb') as handle:
            features = pickle.load(handle).todense()

        with open(os.path.join(self.d, 'labels.pickle'), 'rb') as handle:
            labels = self._1hot(pickle.load(handle))

        return {'hypergraph': hypergraph, 'features': features, 'labels': labels, 'n': features.shape[0]}



    def _1hot(self, labels):
        """
        converts each positive integer (representing a unique class) into ints one-hot form

        Arguments:
        labels: a list of positive integers with eah integer representing a unique label
        """
        
        classes = set(labels)
        onehot = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        return np.array(list(map(onehot.get, labels)), dtype=np.int32)




