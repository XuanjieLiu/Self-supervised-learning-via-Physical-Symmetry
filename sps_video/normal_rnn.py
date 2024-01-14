import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import gzip


# todo: make these parameters configurable
BATCH_SIZE = 32
log_interval = 10
IMG_CHANNEL = 3

LAST_H = 4
LAST_W = 4

FIRST_CH_NUM = 64
LAST_CN_NUM = FIRST_CH_NUM * 4

RNN_INPUT_SIZE = 3
RNN_OUT_FEATURES = 3


class Conv2dGruConv2d(nn.Module):
    def __init__(self, config):
        super(Conv2dGruConv2d, self).__init__()
        self.rnn_hidden_size = config['rnn_hidden_size']
        self.rnn_num_layers = config['rnn_num_layers']
        self.latent_code_num = config['latent_code_num']

        self.encoder = nn.Sequential(
            nn.Conv2d(IMG_CHANNEL, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(FIRST_CH_NUM, FIRST_CH_NUM * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(FIRST_CH_NUM * 2, LAST_CN_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc11 = nn.Linear(LAST_CN_NUM * LAST_H * LAST_W, self.latent_code_num)
        self.fc12 = nn.Linear(LAST_CN_NUM * LAST_H * LAST_W, self.latent_code_num)

        self.rnn = nn.RNN(
            input_size=RNN_INPUT_SIZE,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True,
        )
        self.fc2 = nn.Linear(in_features=self.rnn_hidden_size, out_features=RNN_OUT_FEATURES)

        self.fc3 = nn.Sequential(
            nn.Linear(self.latent_code_num, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, LAST_CN_NUM * LAST_H * LAST_W)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(LAST_CN_NUM, FIRST_CH_NUM * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(FIRST_CH_NUM * 2, FIRST_CH_NUM, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(FIRST_CH_NUM, IMG_CHANNEL, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def save_tensor(self, tensor, path):
        """保存 tensor 对象到文件"""
        torch.save(tensor, path)

    def load_tensor(self, path):
        """从文件读取 tensor 对象"""
        return torch.load(path)

    def reparameterize(self, mu, logvar):
        eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
        z = mu + eps * torch.exp(logvar) * 0.5
        return z

    def batch_encode_to_z(self, x):
        out = self.encoder(x)
        mu = self.fc11(out.view(out.size(0), -1))
        logvar = self.fc12(out.view(out.size(0), -1))
        z1 = self.reparameterize(mu, logvar)
        return z1, mu, logvar

    def batch_seq_encode_to_z(self, x):
        img_in = x.contiguous().view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        z1, mu, logvar = self.batch_encode_to_z(img_in)
        return [z1.view(x.size(0), x.size(1), z1.size(-1)),
                mu.view(x.size(0), x.size(1), z1.size(-1)),
                logvar.view(x.size(0), x.size(1), z1.size(-1))]

    def batch_decode_from_z(self, z):
        out3 = self.fc3(z).view(z.size(0), LAST_CN_NUM, LAST_H, LAST_W)
        frames = self.decoder(out3)
        return frames

    def batch_seq_decode_from_z(self, z):
        z_in = z.reshape(z.size(0) * z.size(1), z.size(2))
        recon = self.batch_decode_from_z(z_in)
        return recon.reshape(z.size(0), z.size(1), recon.size(-3), recon.size(-2), recon.size(-1))

    def do_rnn(self, z, hidden):
        out_r, hidden_rz = self.rnn(z.unsqueeze(1), hidden)
        z2 = self.fc2(out_r.squeeze(1))
        return z2, hidden_rz

    def predict_with_symmetry(self, z_gt, sample_points, symm_func):
        z_SR_seq_batch = []
        hidden_r = torch.zeros([self.rnn_num_layers, z_gt.size(0), self.rnn_hidden_size]).cuda()
        for i in range(z_gt.size(1)):
            """Schedule sample"""
            if i in sample_points:
                z_S = z_SR_seq_batch[-1]
            else:
                z = z_gt[:, i]
                z_S = symm_func(z)
            z_SR, hidden_r = self.do_rnn(z_S, hidden_r)
            z_SR_seq_batch.append(z_SR)
        z_x0ESR = torch.stack(z_SR_seq_batch, dim=0).permute(1, 0, 2).contiguous()[:, :-1, :]
        return z_x0ESR
