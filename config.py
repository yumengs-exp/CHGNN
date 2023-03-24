import argparse


def parse():
    p = argparse.ArgumentParser("CHGNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    p.add_argument('--data', type=str, default='cocitation', help='data name (coauthorship/cocitation/hypergraph)')


    p.add_argument('--dataset', type=str, default='citeseer', help='dataset name (e.g.: cora/dblp for coauthorship, cora/citeseer/pubmed for cocitation, 20newsW100/ModelNet40/NTU2012/Mushroom for hypergraph)')


    p.add_argument('--node_dropping_rate', type=float, default=0.2, help='node dropping rate')


    p.add_argument('--edge_perturbation_rate', type=float, default=0.1, help='edge perturbation rate')


    p.add_argument('--lamda_ec', type=float, default=1, help='loss rate')
    p.add_argument('--lamda_cc', type=float, default=1, help='loss rate')


    p.add_argument('--first-aggregate', type=str, default='mean', help='aggregation for hyperedge h_e: max, sum, mean')


    p.add_argument('--second-aggregate', type=str, default='sum', help='aggregation for node x_i: max, sum, mean')


    p.add_argument('--use-norm', action="store_true", help='use norm in the final layer')


    p.add_argument('--activation', type=str, default='relu', help='activation layer between UniConvs')


    p.add_argument('--nlayer', type=int, default=2, help='number of hidden layers')


    p.add_argument('--nhid', type=int, default=8, help='number of hidden features, note that actually it\'s #nhid x #nhead')


    p.add_argument('--nhead', type=int, default=8, help='number of conv heads')
    
    p.add_argument('--nproj', type=int, default=16, help='number of projection')




    p.add_argument('--dropout', type=float, default=0.6, help='dropout probability after UniConv layer')


    p.add_argument('--input-drop', type=float, default=0.6, help='dropout probability for input layer')


    p.add_argument('--attn-drop', type=float, default=0.6, help='dropout probability for attentions in UniGATConv')


    p.add_argument('--lr', type=float, default=0.01, help='learning rate')


    p.add_argument('--wd', type=float, default=5e-4, help='weight decay')


    p.add_argument('--epochs', type=int, default=2000, help='number of epochs to train')


    p.add_argument('--n-runs', type=int, default=10, help='number of runs for repeated experiments')


    p.add_argument('--gpu', type=int, default=0, help='gpu id to use')


    p.add_argument('--seed', type=int, default=0, help='seed for randomness')


    p.add_argument('--patience', type=int, default=200, help='early stop after specific epochs')


    p.add_argument('--nostdout', action="store_true",  help='do not output logging to terminal')

    p.add_argument('--split', type=int, default=1,  help='choose which train/test split to use')


    p.add_argument('--out-dir', type=str, default='runs/test',  help='output dir')



    p.add_argument('--cut_off_node', type=float, default=0.8, help='node dropping cut-off probability')


    p.add_argument('--cut_off_edge', type=float, default=0.8, help='edge perturbation cut-off probability')


    p.add_argument('--tau', type=float, default=0.5, help='tau')

    p.add_argument('--epcc', type=float, default=0.2, help='epcc')
    p.add_argument('--epec', type=float, default=0.2, help='epec')

    p.add_argument('--w_sim', type=float, default=1, help='w_sim')

    p.add_argument('--w_cl_e', type=float, default=0.2, help='w_cl_e')
    p.add_argument('--w_cl_c', type=float, default=1, help='w_cl_c')

    p.add_argument('--w_homo', type=float, default=1, help='w_homo')

    p.add_argument('--w_crosscl_nc', type=float, default=0.2, help='w_crosscl_nc')
    p.add_argument('--w_crosscl_ne', type=float, default=0.2, help='w_crosscl_ne')
    p.add_argument('--w_crosscl_ec', type=float, default=0.8, help='w_crosscl_ec')

    return p.parse_args()
