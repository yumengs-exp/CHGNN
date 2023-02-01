import gc

import torch
import torch.nn as nn
from torch.optim import Adam
import os
import numpy as np
import time
import datetime
import path
import shutil
import config
from model.HypergraphViewGenerator import HypergraphViewGenerator
import nni
import yaml


args = config.parse()


if args.data == "coauthorship" and args.dataset =="cora":
    hp_setting_name = "cora_coauthorship"
else:
    hp_setting_name = args.dataset
params = yaml.safe_load(open('setting.yaml'))[hp_setting_name]
args.w_cl_c = params['w_cl_c']
args.w_cl_e = params['w_cl_e']
args.w_crosscl_nc = params['w_crosscl_nc']
args.w_crosscl_ne = params['w_crosscl_ne']
args.w_crosscl_ec = params['w_crosscl_ec']
args.w_homo = params['w_homo']
args.epochs = params['epochs']
args.nhid = params['nhid']
args.nproj = params['nproj']

args.epochs=2000


# gpu, seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)

use_norm = 'use-norm' if args.use_norm else 'no-norm'

#### configure output directory

dataname = f'{args.data}_{args.dataset}'
model_name  = 'CHGNN'
nlayer = args.nlayer
dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
out_dir = path.Path(f'./{args.out_dir}/{model_name}_{nlayer}_{dataname}/seed_{args.seed}')

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()

### configure logger
from logger import get_logger

baselogger = get_logger('base logger', f'{out_dir}/logging.log', not args.nostdout)
losslogger = get_logger('loss logger', f'{out_dir}/loss.log',  stdout=False)
resultlogger = get_logger('result logger', f'{out_dir}/result.log', not args.nostdout)

baselogger.info(args)
losslogger.info(args)
resultlogger.info(args)

# load data
from data import data
from prepare import *


test_accs = []
best_val_accs, best_test_accs = [], []



# load data
X, Y, G, overlappness, homogeneity = fetch_data(args)


result = []



def train_with_viewgen(view_gen1, view_gen2, view_optimizer, model, optimizer, X,V,E):

    view_gen1.train()
    view_gen2.train()
    model.train()


    view_optimizer.zero_grad()
    optimizer.zero_grad()



    sample1, edge_index1 = view_gen1(X, edge_index, overlappness, args)
    sample2, edge_index2 = view_gen2(X, edge_index, overlappness, args)

    loss_sim = F.mse_loss(sample1, sample2)
    loss_sim = args.w_sim * (1 - loss_sim)

    Z1, X1 = model(X, edge_index1[0], edge_index1[1])  # Z :Emd X: cluster
    Z2, X2 = model(X, edge_index2[0], edge_index2[1])


    loss_cl_cluster = model.loss_cluster_ada_maxmargin(X1, X2, Y, train_idx)
    loss_cl_hyperedge = model.loss_hyperedge_ada_maxmargin(Z1, Z2, Y, train_idx, edge_index1[0], edge_index2[0],
                                                           edge_index1[1], edge_index2[1], E)
    loss_cl = args.w_cl_c * loss_cl_cluster + args.w_cl_e * loss_cl_hyperedge

    loss_homo1 = model.loss_edgehomo(Z1, edge_index1[0], edge_index1[1], E)
    loss_homo2 = model.loss_edgehomo(Z2, edge_index2[0], edge_index2[1], E)
    loss_homo = args.w_homo * (loss_homo1 + loss_homo2)

    loss_crosscl_nc = model.loss_crosscl_nc(Z1, X2) + model.loss_crosscl_nc(Z2, X1)
    loss_crosscl_ne = model.loss_crosscl_ne(Z1, edge_index2[0], edge_index2[1], E) + model.loss_crosscl_ne(Z2,edge_index1[0],edge_index1[1], E)
    loss_crosscl_ec = model.loss_crosscl_ec(X1, Z2, edge_index2[0], edge_index2[1], E) + model.loss_crosscl_ec(X2,Z1,edge_index1[0],edge_index1[1],E)
    loss_crosscl = args.w_crosscl_nc * loss_crosscl_nc + args.w_crosscl_ne * loss_crosscl_ne + args.w_crosscl_ec * loss_crosscl_ec

    loss_cls = model.loss_cls(X1, X1, Y, train_idx)+model.loss_cls(X2, X2, Y, train_idx)
    loss = loss_cls + loss_cl + loss_crosscl + loss_sim + loss_homo


    losslogger.info(f'epoch:{epoch} | loss_cls:{loss_cls:.4f}  | loss_cl_hyperedge:{loss_cl_hyperedge:.4f} | loss_cl_cluster:{loss_cl_cluster:.4f} | loss_sim:{loss_sim:.4f} | loss_homo:{loss_homo:.4f} | loss_crosscl_nc:{loss_crosscl_nc:.4f} | loss_crosscl_ne:{loss_crosscl_ne:.4f} | loss_crosscl_ec:{loss_crosscl_ec:.4f} ')

    loss.backward()

    view_optimizer.step()
    optimizer.step()



    return loss

# bbbbest=0
for run in range(1, args.n_runs+1):
    run_dir = out_dir / f'{run}'
    run_dir.makedirs_p()

    # load data
    args.split = run


    train_idx, test_idx = data.load_split(args)
    train_idx = torch.LongTensor(train_idx).cuda()
    test_idx  = torch.LongTensor(test_idx).cuda()




    inter_class_mask = data.get_class(Y,train_idx)
    # model
    model = HHyperGNN(args, X.shape[1], args.nhid, len(Y.unique()), args.nlayer, args.nhead, args.nproj, homogeneity,args.tau,args.epcc,args.epec)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.cuda()
    H,V,E = initialise(X, Y, G)
    edge_index = torch.cat((V.view(1, -1), E.view(1, -1)), dim=0).cuda()

    view_gen1 = HypergraphViewGenerator(X.shape[1],3)
    view_gen2 = HypergraphViewGenerator(X.shape[1],3)
    view_optimizer = Adam([{'params':view_gen1.parameters()},{'params':view_gen2.parameters()}], lr = args.lr, weight_decay = args.wd)
    view_gen1.cuda()
    view_gen2.cuda()

    baselogger.info(f'Run {run}/{args.n_runs}, Total Epochs: {args.epochs}')
    baselogger.info(model)
    baselogger.info(view_gen1)
    baselogger.info(view_gen2)
    baselogger.info( f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}'  )

    tic_run = time.time()


    from collections import Counter
    counter = Counter(Y[train_idx].tolist())
    baselogger.info(counter)
    label_rate = len(train_idx) / X.shape[0]
    baselogger.info(f'label rate: {label_rate}')

    best_test_acc, test_acc, Z = 0, 0, None
    for epoch in range(args.epochs):
        # train

            tic_epoch = time.time()

            loss = train_with_viewgen(view_gen1, view_gen2, view_optimizer, model, optimizer, X, V, E)

            train_time = time.time() - tic_epoch

            # eval
            model.eval()
            Z = model.test(X, V, E)
            train_acc = accuracy(Z[train_idx], Y[train_idx])
            test_acc = accuracy(Z[test_idx], Y[test_idx])
            result.append(test_acc)
            best_test_acc = max(best_test_acc, test_acc)
            baselogger.info(f'epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.2f} | test acc:{test_acc:.2f} | time:{train_time * 1000:.1f}ms')

    resultlogger.info(f"Run {run}/{args.n_runs}, best test accuracy: {best_test_acc:.2f}, total time: {time.time()-tic_run:.2f}s")
    test_accs.append(test_acc)
    best_test_accs.append(best_test_acc)


resultlogger.info(f"Average test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")
