import numpy as np
from models import *
import torch.nn.functional as F
import torch
import deeprobust.graph.utils as utils
from torch.nn.parameter import Parameter
from tqdm import tqdm
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from copy import deepcopy
from utils import reset_args
import random
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, dropout_adj, is_undirected, to_undirected

class FeatAgent:

    def __init__(self, data_all, args):
        self.device = 'cuda'
        self.args = args
        self.data_all = data_all
        self.model = self.pretrain_model()

    def initialize_as_ori_feat(self, feat):
        self.delta_feat.data.copy_(feat)

    def finetune(self, data):
        args = self.args
        print('Finetuning ...')

        for module in self.model.bns.modules():
            if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                if args.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = args.bn_momentum
                else:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
        
        if not hasattr(self, 'model_pre_state'):
            self.model_pre_state = deepcopy(self.model.state_dict())
        if not args.not_reset:
            self.model.load_state_dict(self.model_pre_state) # reset every test sample
        assert args.debug == 1 or args.debug == 2
        model = self.model

        if args.tent:
            for param in model.parameters():
                if args.train_all:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            for module in model.bns.modules():
                if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                    if args.use_learned_stats:
                        module.track_running_stats = True
                        module.momentum = args.bn_momentum
                    else:
                        module.track_running_stats = False
                        module.running_mean = None
                        module.running_var = None
                    module.weight.requires_grad_(True)
                    module.bias.requires_grad_(True)
            # for param in model.bns.parameters():
            #     param.requires_grad = True
            
            if args.sam:
                args.loss = 'sharpness'
                optimizer = SAM(model.parameters(), torch.optim.Adam, lr=args.lr_tta, weight_decay=0)
            else:
                args.loss = 'entropy'
                optimizer = optim.Adam(model.parameters(), lr=args.lr_tta, weight_decay=0)
        else:
            for param in model.parameters():
                param.requires_grad = True
            args.lr = args.lr_feat
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
            # args.lr =0.001; args.epochs=10

        # model.lr=0.1
        edge_index = data.graph['edge_index'].to(self.device)
        feat, labels = data.graph['node_feat'].to(self.device), data.label.to(self.device) #.squeeze()

        self.feat, self.data = feat, data

        model.train()
        # for i in range(args.epochs):
        do_bw = args.sam
        for i in range(args.ep_tta):
            optimizer.zero_grad()
            loss = self.test_time_loss(model, feat, edge_index, do_bw=do_bw, optimizer=optimizer)
            if not do_bw:
                loss.backward()
                optimizer.step()
            if i == 0:
                print(f'Epoch {i}: {loss}')

        model.eval()
        output = model.predict(feat, edge_index)
        loss = self.test_time_loss(model, feat, edge_index, do_bw = False)
        print(f'Epoch {i}: {loss}')
        print('Test:')

        if args.dataset == 'elliptic':
            return self.evaluate_single(model, output, labels, data), output[data.mask], labels[data.mask]
        else:
            return self.evaluate_single(model, output, labels, data), output, labels

    # will re re-wrote by other classes
    def learn_graph(self, data):
        args = self.args
        args = self.args
        self.data = data
        nnodes = data.graph['node_feat'].shape[0]
        d = data.graph['node_feat'].shape[1]
        # optimize delta_feat !!
        delta_feat = Parameter(torch.FloatTensor(nnodes, d).to(self.device))
        self.delta_feat = delta_feat
        delta_feat.data.fill_(1e-7)
        self.optimizer_feat = torch.optim.Adam([delta_feat], lr=args.lr_feat)

        # not learned model anymore, just transform input
        model = self.model
        for param in model.parameters():
            param.requires_grad = False
        model.eval() # should set to eval

        feat, labels = data.graph['node_feat'].to(self.device), data.label.to(self.device)#.squeeze()
        edge_index = data.graph['edge_index'].to(self.device)
        self.edge_index, self.feat, self.labels = edge_index, feat, labels

        for it in tqdm(range(args.epochs)):
            self.optimizer_feat.zero_grad()
            loss = self.test_time_loss(model, feat+delta_feat, edge_index)

            loss.backward()
            if it % 100 == 0:
                print(f'Epoch {it}: {loss}')

            self.optimizer_feat.step()
            if args.debug==2:
                output = model.predict(feat+delta_feat, edge_index)
                print('Test:', self.evaluate_single(model, output, labels, data))

        with torch.no_grad():
            loss = self.test_time_loss(model, feat+delta_feat, edge_index)
        print(f'Epoch {it+1}: {loss}')

        output = model.predict(feat+delta_feat, edge_index)
        print('Test on transformed graph:')
        if args.dataset == 'elliptic':
            return self.evaluate_single(model, output, labels, data), output[data.mask], labels[data.mask]
        else:
            return self.evaluate_single(model, output, labels, data), output, labels

    def augment(self, strategy='dropedge', p=0.5, edge_index=None, edge_weight=None):
        model = self.model
        if hasattr(self, 'delta_feat'):
            delta_feat = self.delta_feat
            feat = self.feat + delta_feat
        else:
            feat = self.feat
        if strategy == 'shuffle':
            idx = np.random.permutation(feat.shape[0])
            shuf_fts = feat[idx, :]
            output = model.get_embed(shuf_fts, edge_index, edge_weight)
        if strategy == "dropedge":
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "dropnode":
            feat = self.feat + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "dropmix":
            feat = self.feat + self.delta_feat
            mask = torch.cuda.FloatTensor(len(feat)).uniform_() > p
            feat = feat * mask.view(-1, 1)
            edge_index, edge_weight = dropout_adj(edge_index, edge_weight, p=p)
            output = model.get_embed(feat, edge_index, edge_weight)

        if strategy == "dropfeat":
            feat = F.dropout(self.feat, p=p) + self.delta_feat
            output = model.get_embed(feat, edge_index, edge_weight)
        if strategy == "featnoise":
            mean, std = 0, p
            noise = torch.randn(feat.size()) * std + mean
            feat = feat + noise.to(feat.device)
            output = model.get_embed(feat, edge_index)
        return output

    # loss during test time : can be to propogate to model or data (depends on type of loss)
    def test_time_loss(self, model, feat, edge_index, edge_weight=None, mode='train', do_bw=False, optimizer=None):
        args = self.args
        loss = 0
        if 'LC' in args.loss: # label constitency
            if mode == 'eval': # random seed setting
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed(args.seed)
            if args.strategy == 'dropedge':
                # output1 = self.augment(strategy=args.strategy, p=0.5, edge_index=edge_index, edge_weight=edge_weight)
                output1 = self.augment(strategy=args.strategy, p=0.05, edge_index=edge_index, edge_weight=edge_weight) #TODO
            if args.strategy == 'dropnode':
                output1 = self.augment(strategy=args.strategy, p=0.05, edge_index=edge_index, edge_weight=edge_weight)
            if args.strategy == 'rwsample':
                output1 = self.augment(strategy=args.strategy, edge_index=edge_index, edge_weight=edge_weight)
            output2 = self.augment(strategy='dropedge', p=0.0, edge_index=edge_index, edge_weight=edge_weight)
            output3 = self.augment(strategy='shuffle', edge_index=edge_index, edge_weight=edge_weight)
            if args.margin != -1:
                loss = inner(output1, output2) - inner_margin(output2, output3, margin=args.margin)
            else:
                loss = inner(output1, output2) - inner(output2, output3)

        if 'recon' in args.loss: # data reconstruction
            model = self.model
            delta_feat = self.delta_feat
            feat = self.feat + delta_feat
            output2 = model.get_embed(feat, edge_index, edge_weight)
            loss += inner(output2[edge_index[0]], output2[edge_index[1]])

        if args.loss == "train":
            train_mask = self.data.train_mask
            loss = F.nll_loss(output[train_mask], labels[train_mask])

        if args.loss == "test":
            model, data = self.model, self.data
            output = model.forward(feat, edge_index, edge_weight)
            y = data.label.to(self.device)
            if self.args.dataset == 'elliptic':
                loss = model.sup_loss(y[data.mask], output[data.mask])
            elif args.dataset == 'ogb-arxiv':
                loss = model.sup_loss(y[data.test_mask], output[data.test_mask])
            else:
                loss = model.sup_loss(y, output)

        if "entropy" in args.loss: # TTA (TENT)
            model, data = self.model, self.data
            if hasattr(self, 'delta_feat'):
                delta_feat = self.delta_feat
                feat = self.feat + delta_feat
            else:
                feat = self.feat
            batch_size = 1000
            output = model.forward(feat, edge_index, edge_weight)
            # output = np.random.permutation(np.arange(len(output))[: batch_size])
            entropy = softmax_entropy(output)
            if args.ent_filter != None:
                mask = entropy < args.ent_filter
                selected_entropy = entropy[mask]
                print(f"num selected ent : {len(selected_entropy)}")
                print(f"total ent : {len(entropy)}")
            else:
                selected_entropy = entropy
            loss = selected_entropy.mean(0)
            
        if "sharpness" == args.loss:
            model, data = self.model, self.data
            if hasattr(self, 'delta_feat'):
                delta_feat = self.delta_feat
                feat = self.feat + delta_feat
            else:
                feat = self.feat
            batch_size = 1000
            output = model.forward(feat, edge_index, edge_weight)
            entropy = softmax_entropy(output)
            if args.ent_filter != None:
                mask = entropy < args.ent_filter
                selected_entropy = entropy[mask]
                print(f"num selected ent : {len(selected_entropy)}")
                print(f"total ent : {len(entropy)}")
            else:
                selected_entropy = entropy
            
            loss = selected_entropy.mean(0)
            
            if do_bw:
                loss.backward()

                # compute \hat{\epsilon(\Theta)} for first order approximation, Eqn. (4)
                optimizer.first_step(zero_grad=True)

                # second time backward, update model weights using gradients at \Theta+\hat{\epsilon(\Theta)}
                output = model.forward(feat, edge_index, edge_weight)
                entropy2 = softmax_entropy(output)
                                
                if args.ent_filter != None:
                    mask = entropy2 < args.ent_filter
                    selected_entropy2 = entropy2[mask]
                    print(f"num selected ent2 : {len(selected_entropy2)}")
                    print(f"total ent2 : {len(entropy2)}")
                else:
                    selected_entropy2 = entropy2

                loss2 = selected_entropy2.mean(0)
                loss2.backward()
                
                optimizer.second_step(zero_grad=False)
                
        if args.loss == 'dae':
            if hasattr(self, 'delta_feat'):
                delta_feat = self.delta_feat
                feat = self.feat + delta_feat
            else:
                feat = self.feat
            loss = model.get_loss_masked_features(feat, edge_index, edge_weight)

        return loss

    def pretrain_model(self, verbose=True):
        data_all = self.data_all
        args = self.args
        device = self.device
        if type(data_all[0]) is not list:
            feat, labels = data_all[0].graph['node_feat'], data_all[0].label
            edge_index = data_all[0].graph['edge_index']
        else:
            feat, labels = data_all[0][0].graph['node_feat'], data_all[0][0].label
            edge_index = data_all[0][0].graph['edge_index']
        # reset_args(args)
        if args.model == "GCN" or args.model == "GCNSLAPS":
            save_mem = False
            model = GCN(nfeat=feat.shape[1], nhid=args.hidden, dropout=args.dropout, nlayers=args.nlayers,
                        weight_decay=args.weight_decay, with_bn=True, lr=args.lr, save_mem=save_mem,
                        nclass=max(labels).item()+1, device=device, args=args).to(device)

        elif args.model == "GAT":
            model = GAT(nfeat=feat.shape[1], nhid=32, heads=4, lr=args.lr, nlayers=args.nlayers,
                  nclass=labels.max().item() + 1, with_bn=True, weight_decay=args.weight_decay,
                  dropout=0.0, device=device, args=args).to(device)
        elif args.model == "GCNII":
            model = GCNII(nfeat=feat.shape[1], nhid=32, lr=args.lr, nlayers=10,
                  nclass=labels.max().item() + 1, weight_decay=args.weight_decay,
                  dropout=0.0, device=device, args=args).to(device)
        elif args.model == "SAGE":
            if args.dataset == "fb100":
                model = SAGE2(feat.shape[1], 32, max(labels).item()+1, num_layers=args.nlayers,
                        dropout=0.0, lr=0.01, weight_decay=args.weight_decay,
                        device=device, args=args, with_bn=args.with_bn).to(device)
            else:
                model = SAGE(feat.shape[1], 32, max(labels).item()+1, num_layers=args.nlayers,
                        dropout=0.0, lr=0.01, weight_decay=args.weight_decay, device=device,
                        args=args, with_bn=args.with_bn).to(device)
        elif args.model == "GPR":
            model = GPRGNN(feat.shape[1], 32, max(labels).item()+1, dropout=0.0,
                    lr=0.01, weight_decay=args.weight_decay, device=device, args=args).to(device)
        elif args.model == "APPNP":
            if args.dataset == 'ogb-arxiv':
                model = APPNP(nfeat=feat.shape[1], nhid=args.hidden, dropout=0.2, nlayers=4,
                            with_bn=True, lr=0.01,
                            weight_decay=0, nclass=max(labels).item()+1, device=device, args=args).to(device)
            else:
                model = APPNP(nfeat=feat.shape[1], nhid=args.hidden, dropout=0.5, nlayers=10,
                        weight_decay=args.weight_decay,
                        nclass=max(labels).item()+1, device=device, args=args).to(device)
        else:
            raise NotImplementedError
        if verbose: print(model)

        import os.path as osp
        if args.ood:
            filename = f'saved/{args.dataset}_{args.model}_s{args.seed}.pt'
            if args.model == "GCNSLAPS":
                filename = f'saved/{args.dataset}_GCN_s{args.seed}.pt'
        else:
            filename = f'saved_no_ood/{args.dataset}_{args.model}_s{args.seed}.pt'
        if args.debug and osp.exists(filename):
            model.load_state_dict(torch.load(filename, map_location=self.device))
        else:
            # do pre-training again every-time (fast)
            train_iters = 500 if args.dataset == 'ogb-arxiv' else 200
            model.fit_inductive(data_all, train_iters=train_iters, patience=500, verbose=True)
            if args.debug:
                torch.save(model.state_dict(), filename)
        if args.model == "GCNSLAPS":
            assert args.debug > 0
            model.setup_dae(feat.shape[1], nhid=args.hidden, nclass=feat.shape[1])
            model.train_dae(feat, edge_index, None)

        if verbose: self.evaluate(model)
        return model

    def evaluate_single(self, model, output, labels, test_data, verbose=True):
        eval_func = model.eval_func
        if self.args.dataset in ['ogb-arxiv']:
            acc_test = eval_func(labels[test_data.test_mask], output[test_data.test_mask])
        elif self.args.dataset in ['cora', 'amazon-photo', 'twitch-e', 'fb100']:
            acc_test = eval_func(labels, output)
        elif self.args.dataset in ['elliptic']:
            acc_test = eval_func(labels[test_data.mask], output[test_data.mask])
        else:
            raise NotImplementedError
        if verbose:
            print('Test:', acc_test)
        return acc_test

    def evaluate(self, model):
        model.eval()
        accs = []
        y_te, out_te = [], []
        y_te_all, out_te_all = [], []
        for ii, test_data in enumerate(self.data_all[2]):
            x, edge_index = test_data.graph['node_feat'], test_data.graph['edge_index']
            x, edge_index = x.to(self.device), edge_index.to(self.device)
            output = model.predict(x, edge_index)

            labels = test_data.label.to(self.device) #.squeeze()
            eval_func = model.eval_func
            if self.args.dataset in ['ogb-arxiv']:
                acc_test = eval_func(labels[test_data.test_mask], output[test_data.test_mask])
                accs.append(acc_test)
                y_te_all.append(labels[test_data.test_mask])
                out_te_all.append(output[test_data.test_mask])
            elif self.args.dataset in ['cora', 'amazon-photo', 'twitch-e', 'fb100']:
                acc_test = eval_func(labels, output)
                accs.append(acc_test)
                y_te_all.append(labels)
                out_te_all.append(output)
            elif self.args.dataset in ['elliptic']:
                acc_test = eval_func(labels[test_data.mask], output[test_data.mask])
                y_te.append(labels[test_data.mask])
                out_te.append(output[test_data.mask])
                y_te_all.append(labels[test_data.mask])
                out_te_all.append(output[test_data.mask])
                if ii % 4 == 0 or ii == len(self.data_all[2]) - 1:
                    acc_te = eval_func(torch.cat(y_te, dim=0), torch.cat(out_te, dim=0))
                    accs += [float(f'{acc_te:.2f}')]
                    y_te, out_te = [], []
            else:
                raise NotImplementedError
        print('Test accs:', accs)
        acc_te = eval_func(torch.cat(y_te_all, dim=0), torch.cat(out_te_all, dim=0))
        print(f'flatten test: {acc_te}')

    def get_perf(self, output, labels, mask):
        loss = F.nll_loss(output[mask], labels[mask])
        acc = utils.accuracy(output[mask], labels[mask])
        print("loss= {:.4f}".format(loss.item()),
              "accuracy= {:.4f}".format(acc.item()))
        return loss.item(), acc.item()

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **logits**."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from **log_softmax**."""
    return -(x * torch.log(x+1e-15)).sum(1)

def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def sim(t1, t2):
    # cosine similarity
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (t1 * t2).sum(1)

def inner(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return (1-(t1 * t2).sum(1)).mean()

def inner_margin(t1, t2, margin):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return F.relu(1-(t1 * t2).sum(1)-margin).mean()

def diff(t1, t2):
    t1 = t1 / (t1.norm(dim=1).view(-1,1) + 1e-15)
    t2 = t2 / (t2.norm(dim=1).view(-1,1) + 1e-15)
    return 0.5*((t1-t2)**2).sum(1).mean()



class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        # print(self.base_optimizer, self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups