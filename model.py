import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn


# Three different activation function uses in the ZINB-based denoising autoencoder.
MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e4)
DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e3)
PiAct = lambda x: 1 / (1 + torch.exp(-1 * x))


# A random Gaussian noise uses in the ZINB-based denoising autoencoder.
class GaussianNoise(nn.Module):
    def __init__(self, device, sigma=1, is_relative_detach=True):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, device=device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z, device):
        super(AE, self).__init__()

        # autoencoder for intra information
        # self.dropout = nn.Dropout(0.2)
        self.Gnoise = GaussianNoise(sigma=0.1, device=device)
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.BN_1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.BN_2 = nn.BatchNorm1d(n_enc_2)
        self.z_layer = nn.Linear(n_enc_2, n_z)

        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)

        self.calcu_pi = nn.Linear(n_dec_2, n_input)
        self.calcu_disp = nn.Linear(n_dec_2, n_input)
        self.calcu_mean = nn.Linear(n_dec_2, n_input)



    def forward(self, x):

        enc_h1 = self.BN_1(F.relu(self.enc_1(self.Gnoise(x))))
        enc_h2 = self.BN_2(F.relu(self.enc_2(self.Gnoise(enc_h1))))
        z = self.z_layer(self.Gnoise(enc_h2))

        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))

        pi = PiAct(self.calcu_pi(dec_h2))
        mean = MeanAct(self.calcu_mean(dec_h2))
        disp = DispAct(self.calcu_disp(dec_h2))


        return z, pi, mean ,disp, enc_h1, enc_h2


# Final model
class GraphEncoder(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z, alpha, n_clusters, device):
        super(GraphEncoder, self).__init__()

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_input=n_input,
            n_z=n_z,
            device=device
        )

        # autoencoder for intra information
        # self.dropout = nn.Dropout(0.2)
        self.gat_model = GAT(num_features=n_input, hidden1_size=n_enc_1,
                             hidden2_size=n_enc_2, embedding_size=n_z,
                             n_clusters=n_clusters, alpha=alpha)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x, adj, M):
        z, pi, mean, disp, enc_h1, enc_h2 = self.ae(x)
        predict = self.gat_model(x, adj, M, enc_h1, enc_h2, z)

        alpha = 1.0
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2) / alpha)
        q = q ** (alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, dim=1)).t()

        tmp_q = q.data
        weight = tmp_q ** 2 / torch.sum(tmp_q, dim=0)
        p = (weight.t() / torch.sum(weight, dim=1)).t()


        return z, pi, mean ,disp, predict, q, p


class GAT(nn.Module):
    def __init__(self, num_features, hidden1_size, hidden2_size, embedding_size, n_clusters, alpha):
        super(GAT, self).__init__()
        # encoder
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GraphAttentionLayer(num_features, hidden1_size, alpha)   #256
        self.conv2 = GraphAttentionLayer(hidden1_size, hidden2_size, alpha)     #64
        self.conv3 = GraphAttentionLayer(hidden2_size, embedding_size, alpha)       #16
        self.conv4 = GraphAttentionLayer(embedding_size, n_clusters, alpha)             #

    def forward(self, x, adj, M, enc_feat1, enc_feat2, embedding):
        sigma = 0.4
        # sigma = 0.1
        # h1 = self.conv1(x, adj, M)
        # h2 = self.conv2((1-sigma)*h1 + sigma*enc_feat1, adj, M)     #z1
        # h3 = self.conv3((1-sigma)*h2 + sigma*enc_feat2, adj, M)
        # h4 = self.conv4((1-sigma)*h3 + sigma*embedding, adj, M)
        # z = F.normalize(h4, p=2, dim=1)

        h1 = self.conv1(x, adj, M)
        h2 = self.conv2((1-sigma)*h1 + sigma*enc_feat1, adj, M)     #z1
        h3 = self.conv3((1-sigma)*h2 + sigma*enc_feat2, adj, M)
        h4 = self.conv4((1-sigma)*h3 + sigma*embedding, adj, M)
        s = torch.mm(h1, h1.t())
        z1 = torch.mm(s, h4)
        z = F.normalize(z1, p=2, dim=1)



        return z

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https ://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M, concat=True):
        h = torch.mm(input, self.W)
        attn_for_self = torch.mm(h,self.a_self)
        attn_for_neighs = torch.mm(h,self.a_neighs)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs,0,1)
        attn_dense = torch.mul(attn_dense,M)
        attn_dense = self.leakyrelu(attn_dense)


        zero_vec = -9e15*torch.ones_like(adj)
        adj = torch.where(adj > 0, attn_dense, zero_vec)
        attention = F.softmax(adj, dim=1)
        h_prime = torch.matmul(attention,h)

        if concat:
            return F.elu(h_prime)
        else:
            return h_prime