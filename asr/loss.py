import torch
import torch.nn as nn
import numpy as np

class RatioLoss(nn.Module):
    def __init__(self, eps=1.0e-6):
        super(RatioLoss, self).__init__()
        use_cuda=torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.mse_loss = nn.MSELoss().to(device)

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_true, y_pred) + self.eps)

class PESQLoss(nn.Module):

    def __init__(self, global_weight=1.0):
        super(PESQLoss, self).__init__()
        use_cuda=torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        self.global_weight=glboal_weight
        self.local_weight = 1.0-global_weight

    def forward(self, y_pred,y_true):
        # y_true = (batch, time), y_pred = (batch, time)
        true_pesq = y_true[:,0]
        global_loss = (10**(true_pesq-4.5))*torch.mean((y_true-y_pred)**2, 1)
        mean_pesq = torch.mean(y_pred, 1)
        local_loss = torch.square(true_pesq - mean_pesq)

        return self.global_weight * global_loss + self.local_weight * local_loss

class EnhanceLoss(nn.Module):

    def __init__(self, eps=1.0e-6):
        super(EnhanceLoss, self).__init__()
        use_cuda=torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.mse_loss = nn.MSELoss().to(device)
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse_loss(y_true, y_pred) + self.eps)

class DecoderLoss(nn.Module):

    def __init__(self, blank):
        super(DecoderLoss, self).__init__()
        use_cuda=torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        self.ctc_loss = nn.CTCLoss(blank=blank).to(device)

    def forward(self, output, labels, input_lengths, label_lengths):
        return self.ctc_loss(output, labels, input_lengths, label_lengths)

class CombinedLoss(nn.Module):

    def __init__(self, enhance_weight = 0.5,
                 eps=1.0e-6, blank=42):
        super(CombinedLoss, self).__init__()
        use_cuda=torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        self.enhance_weight=enhance_weight
        self.decoder_weight=1.0 - enhance_weight

        self.enhance_loss = EnhanceLoss(device, eps)
        self.decoder_loss = DecoderLoss(device, blank)

    def forward(self, sig_pred, sig_true, outputs, labels, input_lengths, label_lengths):
        self.eloss=self.enhance_loss(sig_pred, sig_true)
        self.dloss=self.decoder_loss(outputs, labels, input_lengths, label_lengths)
        loss = self.enhance_weight * self.eloss + self.decoder_weight * self.dloss

        return loss

    def get_loss(self):
        return self.eloss, self.dloss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        use_cuda=torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.l1_loss = nn.L1Loss(reduction='none').to(self.device)

    def forward(self, input, target, mask):
        loss = self.l1_loss(input.to(self.device), target.to(self.device))
        loss = (loss * mask.to(self.device)).sum()

        non_zero_elements = mask.sum()
        l1_loss = loss/non_zero_elements

        return l1_loss

class MaskedMSELoss(nn.Module):
    def __init__(self, eps=1.0e-6):
        super(MaskedMSELoss, self).__init__()
        use_cuda=torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.mse_loss = nn.MSELoss(reduction='none').to(self.device)
        self.eps = eps

    def forward(self, input, target, mask):
        loss = self.mse_loss(input.to(self.device), target.to(self.device))
        loss = (loss * mask.to(self.device)).sum()

        non_zero_elements = mask.sum()
        mse_loss = loss/non_zero_elements

        return mse_loss

class Masked2Loss(nn.Module):
    def __init__(self, eps=1.0e-6):
        super(Masked2Loss, self).__init__()
        self.mse_loss=MaskedMSELoss()
        self.l1_loss=MaskedL1Loss()
        self.eps=eps

    def forward(self, input, target, mask):
        loss = 0.5*self.mse_loss(input, target, mask)
        + 0.5*self.l1_loss(input, target, mask)

        return loss
    
class MaskedBCELoss(nn.Module):
    def __init__(self):
        super(MaskedBCELoss, self).__init__()
        use_cuda=torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.bce_loss = nn.BCELoss(reduction='none').to(self.device)

    def forward(self, input, target, mask):
        loss = self.bce_loss(input.to(self.device), target.to(self.device))
        loss = (loss * mask.to(self.device)).sum()

        non_zero_elements = mask.sum()
        bce_loss = loss/non_zero_elements

        return bce_loss

    '''
    def forward(self, encoder_out, decoder_out, encoder_mask, decoder_mask):
        # generator loss
        # against encoder
        fake_enc = encoder_out[encoder_out.shape[0]//2:]
        fake_enc_mask = encoder_mask[encoder_out.shape[0]//2:]
        fake_enc_target = np.ones(fake_enc.shape)
        fake_enc_target = torch.from_numpy(fake_enc_target.astype(np.float32)).clone().to(self.device)
        loss1 = self.bce_loss(fake_enc, fake_enc_target)
        loss1 = (loss1 * fake_enc_mask).sum()
        non_zero_elements = fake_enc_mask.sum()
        loss1 = loss1/non_zero_elements

        # against decoder
        fake_dec = decoder_out[decoder_out.shape[0]//2:]
        fake_dec_mask = decoder_mask[decoder_out.shape[0]//2:]
        fake_dec_target = np.ones(fake_dec.shape)
        fake_dec_target = torch.from_numpy(fake_dec_target.astype(np.float32)).clone().to(self.device)
        loss2 = self.bce_loss(fake_dec, fake_dec_target)
        loss2 = (loss2 * fake_dec_mask).sum()
        non_zero_elements = fake_dec_mask.sum()
        loss2 = loss2/non_zero_elements

        # discriminator loss
        # against encoder
        enc_target = np.zeros(encoder_out.shape)
        enc_target[0:encoder_out.shape[0]//2, :, :] = 1.0
        enc_target[encoder_out.shape[0]//2:, :, :] = 0.0
        enc_target = torch.from_numpy(enc_target.astype(np.float32)).clone().to(self.device)
        loss3 = self.bce_loss(encoder_out, enc_target)
        loss3 = (loss3 * encoder_mask).sum()
        non_zero_elements = encoder_mask.sum()
        loss3 = loss3/non_zero_elements

        # against decoder
        dec_target = np.zeros(decoder_out.shape)
        dec_target[0:decoder_out.shape[0]//2, :, :] = 1.0
        dec_target[decoder_out.shape[0]//2:, :, :] = 0.0
        dec_target = torch.from_numpy(dec_target.astype(np.float32)).clone().to(self.device)
        loss4 = self.bce_loss(decoder_out, dec_target)
        loss4 = (loss4 * decoder_mask).sum()
        non_zero_elements = decoder_mask.sum()
        loss4 = loss4/non_zero_elements

        return loss1,loss2,loss3,loss4
    '''
