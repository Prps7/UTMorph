import torch
import torch.nn as nn
from monai.losses import DiceLoss,MultiScaleLoss,BendingEnergyLoss
import torch.nn.functional as F
from monai.losses import DiceLoss

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x.to('cuda:0') - y.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(1,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()
        self.l1 = torch.nn.L1Loss()
        self.l2 = nn.MSELoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered2    = self.conv_gauss(new_filter)
        diff = current - filtered2

        # visualize_tensors(diff,current,new_filter,filtered2)
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x.to('cuda')), self.laplacian_kernel(y.to('cuda')))+self.l1(x.to('cuda'), y.to('cuda'))
        return loss

