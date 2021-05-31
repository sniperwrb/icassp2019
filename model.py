import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=32, num_speakers=20, repeat_num=1):
        super(Generator, self).__init__()
        self.c_dim = num_speakers
        self.r = nn.ReLU(inplace=True)
        self.l = nn.LeakyReLU(0.1, inplace=True)
        # 1 relu
        # 2 relu 3 relu 4 add 5 relu 6 relu 7 add 8 relu 9 relu 10 add
        # 11 leak 12 relu 13 add 14 leak 15 relu 16 add
        # 17 relu 18 relu 19 add 20 relu 21 relu 22 add
        # 23 relu 24

        curr_dim = conv_dim
        c_dim = num_speakers
        self.c1 = nn.Conv2d(1, curr_dim, kernel_size=(3, 9), padding=(1, 4), bias=False)
        self.n1 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)

        self.c2 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False)
        curr_dim = curr_dim * 2
        self.n2 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c3 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n3 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c4 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n4 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)

        self.c5 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False)
        curr_dim = curr_dim * 2
        self.n5 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c6 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n6 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c7 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n7 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)

        self.c8 = nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False)
        curr_dim = curr_dim * 2
        self.n8 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c9 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n9 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c10 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n10 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)

        curr_dim_old = curr_dim + 0
        self.c11 = nn.Conv2d(curr_dim_old, curr_dim, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), bias=False)
        self.n11 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c12 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n12 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c13 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n13 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)

        self.c14 = nn.Conv2d(curr_dim+c_dim, curr_dim_old, kernel_size=(3, 5), stride=(1, 1), padding=(1, 2), bias=False)
        curr_dim = curr_dim_old
        self.n14 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c15 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n15 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c16 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n16 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)

        self.c17 = nn.ConvTranspose2d(curr_dim+c_dim, curr_dim//2, kernel_size=4, stride=(2, 2), padding=(1, 1), bias=False)
        curr_dim = curr_dim // 2
        self.n17 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c18 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n18 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c19 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n19 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)

        self.c20 = nn.ConvTranspose2d(curr_dim+c_dim, curr_dim//2, kernel_size=4, stride=(2, 2), padding=(1, 1), bias=False)
        curr_dim = curr_dim // 2
        self.n20 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c21 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n21 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c22 = nn.Conv2d(curr_dim, curr_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.n22 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)

        self.c23 = nn.ConvTranspose2d(curr_dim+c_dim, curr_dim//2, kernel_size=4, stride=(2, 2), padding=(1, 1), bias=False)
        curr_dim = curr_dim // 2
        self.n23 = nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=True)
        self.c24 = nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sm  = nn.Softmax(dim=2)

    def conv_norm(self, x, lin_nets, res_nets=[], emb=False, c=None):
        if emb:
            c = c.view(c.size(0), c.size(1), 1, 1)
            c = c.repeat(1, 1, x.size(2), x.size(3))
            xc = torch.cat([x, c], dim=1)
        else:
            xc = x
        for i in range(len(lin_nets)):
            if (i==0):
                y = lin_nets[0](xc)
            else:
                y = lin_nets[i](y)
        for i in range(len(res_nets)):
            if (i==0):
                z = res_nets[0](y)
            else:
                z = res_nets[i](z)
        if len(res_nets)>0:
            y = y + z
        return y

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        out = self.conv_norm(x  , [self.c1,  self.n1,  self.r], [])
        out = self.conv_norm(out, [self.c2,  self.n2,  self.r], 
                                  [self.c3,  self.n3,  self.r , self.c4, self.n4])
        out = self.conv_norm(out, [self.c5,  self.n5,  self.r], 
                                  [self.c6,  self.n6,  self.r , self.c7, self.n7])
        out = self.conv_norm(out, [self.c8,  self.n8,  self.r], 
                                  [self.c9,  self.n9,  self.r , self.c10, self.n10])
        out = self.conv_norm(out, [self.c11, self.n11, self.l], 
                                  [self.c12, self.n12, self.r , self.c13, self.n13])
        
        #print(out.size()) # [32=bsize, 512=channels, 12=worldfeats>>3, 32=frames>>3]
        s1, s2, s3, s4 = out.size()
        out = out.view(s1, s2//8, s3*8, s4)
        out = self.sm(out)
        out = out.view(s1, s2, s3, s4)
        
        out = self.conv_norm(out, [self.c14, self.n14, self.l], 
                                  [self.c15, self.n15, self.r , self.c16, self.n16], True, c)
        out = self.conv_norm(out, [self.c17, self.n17, self.r], 
                                  [self.c18, self.n18, self.r , self.c19, self.n19], True, c)
        out = self.conv_norm(out, [self.c20, self.n20, self.r], 
                                  [self.c21, self.n21, self.r , self.c22, self.n22], True, c)
        out = self.conv_norm(out, [self.c23, self.n23, self.r , self.c24], [], True, c)
        return out

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(96, 256), conv_dim=64, repeat_num=5, num_speakers=20):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num)) # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num)) # 8
        self.main = nn.Sequential(*layers)
        self.conv_dis = nn.Conv2d(curr_dim, 1, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False) # padding should be 0
        self.conv_clf_spks = nn.Conv2d(curr_dim, num_speakers, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False)  # for num_speaker
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_dis(h)
        out_cls_spks = self.conv_clf_spks(h)
        return out_src, out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = get_loader('/scratch/sxliu/data_exp/VCTK-Corpus-22.05k/mc/train', 16, 'train', num_workers=1)
    data_iter = iter(train_loader)
    G = Generator().to(device)
    D = Discriminator().to(device)
    for i in range(10):
        mc_real, spk_label_org, acc_label_org, spk_acc_c_org = next(data_iter)
        mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d
        mc_real = mc_real.to(device)                         # Input mc.
        spk_label_org = spk_label_org.to(device)             # Original spk labels.
        acc_label_org = acc_label_org.to(device)             # Original acc labels.
        spk_acc_c_org = spk_acc_c_org.to(device)             # Original spk acc conditioning.
        mc_fake = G(mc_real, spk_acc_c_org)
        print(mc_fake.size())
        out_src, out_cls_spks, out_cls_emos = D(mc_fake)



