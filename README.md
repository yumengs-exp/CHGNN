# CHGNN





## Getting Started

### Prerequisites

Our code requires Python>=3.6. 

We recommend using a virtual environment and installing the newest versions of  [Pytorch](https://pytorch.org/) and [Pytorch-Geometric](https://github.com/rusty1s/pytorch_geometric).


You also need these additional packages:

* scipy
* path
* tqdm


## Datasets

co-authorship hypergraphs
* Cora
* DBLP

co-citation hypergraphs
* Pubmed
* Citeseer
* Cora

* For citation datasets, you can download training splits from HyperGCN.
  
other hypergraphs
* 20newsgroup
* ModelNet40
* NTU2012
* Mushroom

## Baselines
UniGNN, HyperGCN, HyperSAGE, and HGNN  can be found   at [https://github.com/OneForward/UniGNN](https://github.com/OneForward/UniGNN).

AllSetTransformer can be found at [https://github.com/jianhao2016/AllSet](https://github.com/jianhao2016/AllSet).

SimGRACE can be found at [https://github.com/junxia97/SimGRACE](https://github.com/junxia97/SimGRACE).

DGI can be found at [https://github.com/PetarV-/DGI](https://github.com/PetarV-/DGI).

TriCL can be found at [https://github.com/wooner49/TriCL](https://github.com/wooner49/TriCL).

## Semi-supervised Hypernode Classification

```sh
python train.py --data=coauthorship --dataset=cora 
```

You should probably see final accuracies like the following.  

`Average test accuracy: 76.79657001495361 ± 1.01541351159170083`








