import sys

from torch import utils

sys.path.append('../')
import math

import numpy as np
import torch
import world
from dataloader import BasicDataset
from sklearn.cluster import KMeans
from torch import nn
from utils import cprint, cos_similarity
from multiprocessing.dummy import Pool as ThreadPool
from model.basic_model import BasicModel


class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['GCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph, self.Graph_one = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        #
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    

    
    def computer__(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def computer_DAP(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        u_degree = torch.tensor(self.dataset.users_D)
        i_degree = torch.tensor(self.dataset.items_D)
        degree = torch.cat([u_degree, i_degree], dim=0)
        all_embs_anchor_higher = torch.zeros([self.num_users + self.num_items, self.latent_dim]).to(world.device)
        all_embs_anchor_lower = torch.zeros([self.num_users + self.num_items, self.latent_dim]).to(world.device)
        embs = [all_emb]
        g_droped = self.Graph

        weight1 = self.config['alpha1']
        weight2 = self.config['alpha2']
        num_cluster = self.config['cluster_num']
        print(f'weight1={weight1},weight2={weight2}, num_cluster:{num_cluster}')
        for layer in range(self.n_layers):
            all_emb_i = torch.sparse.mm(g_droped, all_emb)
            kmeans = KMeans(n_clusters=num_cluster, random_state=9)
            cprint(f'GCN debiasing at {layer + 1}-layer')
            all_emb_cluster = kmeans.fit_predict(all_emb_i.to('cpu'))
            for k_cluster in range(num_cluster):
                index = np.where(all_emb_cluster == k_cluster)
                embs_cluster = all_emb_i[index[0]]  
                degree_cluster = degree[index[0]].unsqueeze(1) 
                for i in range(len(index[0])):
                    if degree[index[0][i]] > degree_cluster.min() and degree[index[0][i]] < degree_cluster.max():
                        degree_higher_index = torch.where(degree_cluster > degree[index[0][i]])
                        degree_lower_index = torch.where(degree_cluster < degree[index[0][i]])
                        embs_higher_mean = embs_cluster[degree_higher_index[0]].mean(0)
                        embs_lower_mean = embs_cluster[degree_lower_index[0]].mean(0)
                        all_embs_anchor_higher[index[0][i]] = embs_higher_mean
                        all_embs_anchor_lower[index[0][i]] = embs_lower_mean
                    elif degree[index[0][i]] == degree_cluster.min():
                        degree_higher_index = torch.where(degree_cluster > degree[index[0][i]])
                        embs_higher_mean = embs_cluster[degree_higher_index[0]].mean(0)
                        all_embs_anchor_higher[index[0][i]] = embs_higher_mean
                    elif degree[index[0][i]] == degree_cluster.max():
                        degree_lower_index = torch.where(degree_cluster < degree[index[0][i]])
                        embs_lower_mean = embs_cluster[degree_lower_index[0]].mean(0)
                        all_embs_anchor_lower[index[0][i]] = embs_lower_mean
            all_embs_anchor_higher = nn.functional.normalize(all_embs_anchor_higher)
            all_embs_anchor_lower = nn.functional.normalize(all_embs_anchor_lower)
            alpha_sim_higher = cos_similarity(all_emb, all_embs_anchor_higher.to(world.device))
            alpha_sim_lower = cos_similarity(all_emb, all_embs_anchor_lower.to(world.device))
            all_emb = all_emb_i - weight1 * alpha_sim_higher.unsqueeze(1).to(world.device) * all_embs_anchor_higher.to(world.device)\
                      - weight2 * alpha_sim_lower.unsqueeze(1).to(world.device) * all_embs_anchor_lower.to(world.device)
            # all_emb = all_emb_i - weight1 * all_embs_anchor_higher.to(world.device) - weight2 * all_embs_anchor_lower.to(world.device)
            # all_emb = all_emb_i - weight * alpha_sim_lower.unsqueeze(1).to(world.device) * all_embs_anchor_lower.to(world.device)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def computer_one(self):
        print('one-order neighbors')
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        u_degree = torch.tensor(self.dataset.users_D)
        i_degree = torch.tensor(self.dataset.items_D)
        degree = torch.cat([u_degree, i_degree], dim=0)
        all_embs_anchor_higher = torch.zeros([self.num_users + self.num_items, self.latent_dim]).to(world.device)
        all_embs_anchor_lower = torch.zeros([self.num_users + self.num_items, self.latent_dim]).to(world.device)
        degree_higher_num = torch.zeros([self.num_users + self.num_items, 1]).to(world.device)
        degree_lower_num = torch.zeros([self.num_users + self.num_items, 1]).to(world.device)
        embs = [all_emb]

        nodes = [i for i in range(len(all_emb))]
        weight1 = self.config['alpha1']
        weight2 = self.config['alpha2']
        print(f'weight1={weight1}, weight2={weight2}')
        for layer in range(self.n_layers):
            cprint(f'GCN debiasing at {layer + 1}-layer')
            all_emb_i = torch.sparse.mm(self.Graph, all_emb)
            for node in nodes:
                neighbors = self.Graph[node]._indices()[0]
                if len(neighbors) > 0:
                    embs_cluster = all_emb_i[neighbors]
                    degree_cluster = degree[neighbors].unsqueeze(1)

                    if degree[node] > degree_cluster.min() and degree[node] < degree_cluster.max():
                        degree_higher_index = torch.where(degree_cluster > degree[node])
                        degree_higher_num[node] = degree_cluster[degree_higher_index[0]].sum()
                        degree_lower_index = torch.where(degree_cluster < degree[node])
                        degree_lower_num[node] = degree_cluster[degree_lower_index[0]].sum()
                        embs_higher_mean = embs_cluster[degree_higher_index[0]].mean(0)
                        embs_lower_mean = embs_cluster[degree_lower_index[0]].mean(0)
                        all_embs_anchor_higher[node] = embs_higher_mean
                        all_embs_anchor_lower[node] = embs_lower_mean
                    elif degree[node] == degree_cluster.min():
                        degree_higher_index = torch.where(degree_cluster > degree[node])
                        degree_higher_num[node] = degree_cluster[degree_higher_index[0]].sum()
                        embs_higher_mean = embs_cluster[degree_higher_index[0]].mean(0)
                        all_embs_anchor_higher[node] = embs_higher_mean
                    elif degree[node] == degree_cluster.max():
                        degree_lower_index = torch.where(degree_cluster < degree[node])
                        degree_lower_num[node] = degree_cluster[degree_lower_index[0]].sum()
                        embs_lower_mean = embs_cluster[degree_lower_index[0]].mean(0)
                        all_embs_anchor_lower[node] = embs_lower_mean
            all_embs_anchor_higher = nn.functional.normalize(all_embs_anchor_higher)
            all_embs_anchor_lower = nn.functional.normalize(all_embs_anchor_lower)
            alpha_sim_higher = cos_similarity(all_emb, all_embs_anchor_higher.to(world.device))
            alpha_sim_lower = cos_similarity(all_emb, all_embs_anchor_lower.to(world.device))
            degree_impact = degree_higher_num > degree_lower_num
            all_emb = all_emb_i - weight1* alpha_sim_higher.unsqueeze(1).to(world.device) * all_embs_anchor_higher.to(world.device) \
                      - weight2* alpha_sim_lower.unsqueeze(1).to(world.device) * all_embs_anchor_lower.to(world.device)
            # all_emb = all_emb_i - weight * alpha_sim_lower.unsqueeze(1).to(world.device) * all_embs_anchor_lower.to(world.device)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, all_users, all_items, users):
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                         posEmb0.norm(2).pow(2) +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
