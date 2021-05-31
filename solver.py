from model import Generator, Discriminator
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
from os.path import join, basename
import time
import datetime
from data_loader import to_categorical
import librosa
import utils
import sys


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader, test_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sampling_rate = config.sampling_rate

        # Model configurations.
        self.num_speakers = config.num_speakers
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_clip_thresh = config.grad_clip_thresh

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(num_speakers=self.num_speakers)
        self.D = Discriminator(num_speakers=self.num_speakers)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.G.to(self.device)
        self.D.to(self.device)
        
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def sample_spk_c(self, size):
        spk_c = np.random.randint(0, self.num_speakers, size=size)
        spk_c_cat = to_categorical(spk_c, self.num_speakers)
        return torch.LongTensor(spk_c), torch.FloatTensor(spk_c_cat)

    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss."""
        return F.cross_entropy(logit, target)

    def load_wav(self, wavfile, sr=16000):
        wav, _ = librosa.load(wavfile, sr=sr, mono=True)
        return utils.wav_padding(wav, sr=16000, frame_period=5, multiple = 4)  # TODO

    def train(self):
        """Train StarGAN."""
        jj=0

        # Set data loader.
        train_loader = self.train_loader
        data_iter = iter(train_loader)

        # Read a batch of testdata
        test_wavfiles = self.test_loader.get_batch_test_data(batch_size=4)
        f0_converteds = []
        aps = []
        coded_sp_norm_tensors = []

        sampling_rate=16000
        num_mcep=96
        frame_period=5
        for i in range(len(test_wavfiles)):
            wav = utils.load_wav(test_wavfiles[i], sampling_rate)
            wav = utils.wav_volume_rescaling(wav)
            wav = utils.trim_silence(wav, 30)

            f0, timeaxis, sp, ap = utils.world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = utils.pitch_conversion(f0=f0, 
                mean_log_src=self.test_loader.logf0_mean_src, 
                mean_log_target=self.test_loader.logf0_mean_trg)
            coded_sp = utils.world_encode_spectral_envelop(sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_norm = (coded_sp - self.test_loader.mcep_mean) / self.test_loader.mcep_std
            coded_sp_norm_tensor = torch.FloatTensor(coded_sp_norm.T).unsqueeze_(0).unsqueeze_(1).to(self.device)

            f0_converteds.append(f0_converted)
            aps.append(ap)
            coded_sp_norm_tensors.append(coded_sp_norm_tensor)

        conds = torch.FloatTensor(self.test_loader.spk_c_trg).to(self.device)


        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print("resuming step %d ..."% self.resume_iters)
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch labels.
            try:
                mc_real, spk_label_org, spk_c_org_real = next(data_iter)
            except:
                data_iter = iter(train_loader)
                mc_real, spk_label_org, spk_c_org_real = next(data_iter)

            mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d

            # Generate target domain labels randomly.
            try:
                mc_trg, spk_label_trg, spk_c_trg_real = next(data_iter)
            except:
                data_iter = iter(train_loader)
                mc_trg, spk_label_trg, spk_c_trg_real = next(data_iter)
            mc_trg.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d

            mc_real = mc_real.to(self.device)                         # Input mc.
            spk_label_org = spk_label_org.to(self.device)             # Original spk labels.
            spk_c_org_real = spk_c_org_real.to(self.device)                     # Original spk acc conditioning.
            mc_trg_real = mc_trg.to(self.device)                         # Input mc.
            spk_label_trg = spk_label_trg.to(self.device)             # Target spk labels for classification loss for G.
            spk_c_trg_real = spk_c_trg_real.to(self.device)                     # Target spk conditioning.

            loss = {}

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real mc feats.
            out_src, out_cls_spks = self.D(mc_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls_spks = self.classification_loss(out_cls_spks, spk_label_org)
            
            # Compute loss with fake mc feats.
            mc_fake = self.G(mc_real, spk_c_trg_real)
            #print(mc_real.size(),mc_fake.size())
            out_src, out_cls_spks = self.D(mc_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(mc_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * mc_real.data + (1 - alpha) * mc_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls_spks + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.D.parameters(), self.grad_clip_thresh)
            self.d_optimizer.step()

            # Logging.
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls_spks'] = d_loss_cls_spks.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                mc_fake = self.G(mc_real, spk_c_trg_real)
                out_src, out_cls_spks = self.D(mc_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls_spks = self.classification_loss(out_cls_spks, spk_label_trg)

                # Target-to-original domain.
                mc_reconst = self.G(mc_fake, spk_c_org_real)
                g_loss_rec = torch.mean(torch.abs(mc_real - mc_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls_spks
                self.reset_grad()
                g_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.G.parameters(), self.grad_clip_thresh)
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls_spks'] = g_loss_cls_spks.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if ((i+1) % self.log_step == 0) or ((i<100) and ((i+1)%10==0)):
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                sys.stdout.flush()

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            if ((i+1) % self.sample_step == 0) or (i==0):
                with torch.no_grad():
                    for j in range(len(test_wavfiles)):
                        wav_name = basename(test_wavfiles[j])
                        coded_sp_converted_norm = self.G(coded_sp_norm_tensors[j], conds).data.cpu().numpy()
                        coded_sp_converted = np.squeeze(coded_sp_converted_norm).T * self.test_loader.mcep_std + self.test_loader.mcep_mean
                        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)

                        wav_transformed = utils.world_speech_synthesis(f0=f0_converteds[j], coded_sp=coded_sp_converted, 
                                                                 ap=aps[j], fs=sampling_rate, frame_period=frame_period)
                        librosa.output.write_wav(
                            join(self.sample_dir, str(i+1)+'-'+wav_name.split('.')[0]+'-vcto-{}'.format(self.test_loader.trg_spk)+'.wav'), wav_transformed, sampling_rate)

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr=g_lr, d_lr=d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
                    


