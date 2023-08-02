import torch
from torch import nn
from derivatives import curl
import derivatives as kernels
import torch.nn.functional as F
from unet_parts import *
from typing import Optional

class UNet(nn.Module):
    """UNet model with variable number of down/up steps."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: Optional[int] = 64,
        steps: Optional[int] = 4,
        bilinear: Optional[bool] = True,
    ):
        """Initialize UNet model.

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            hidden_size (int, optional): number of hidden features
            steps (int, optional): number of down/up steps
            bilinear (bool, optional): whether to use bilinear interpolation
        """
        super().__init__()

        assert steps >= 1, "steps must be greater than or equal to 1"

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc1 = kernels.Conv(in_features, hidden_size)
        self.inc2 = kernels.Conv(hidden_size, hidden_size)

        self.downs = nn.ModuleList(
            [
                kernels.Down(2 ** i * hidden_size, 2 ** (i + 1) * hidden_size)
                for i in range(steps - 1)
            ]
        )
        self.downs.append(
            kernels.Down(
                2 ** (steps - 1) * hidden_size, 2 ** steps * hidden_size // factor
            )
        )

        self.ups = nn.ModuleList(
            [
                kernels.Up(
                    2 ** (steps - i) * hidden_size,
                    2 ** (steps - i - 1) * hidden_size // factor,
                    bilinear,
                )
                for i in range(steps - 1)
            ]
        )
        self.ups.append(kernels.Up(2 * hidden_size, hidden_size, bilinear))

        self.outc = kernels.OutConv(hidden_size, out_features)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward pass.

        Args:
            x (torch.tensor): input tensor
        """
        x = self.inc1(x)
        x = self.inc2(x)
        layer_outputs = [x]

        for layer in self.downs:
            x = layer(x)
            layer_outputs.append(x)

        for layer, output in zip(self.ups, reversed(layer_outputs[:-1])):
            x = layer(x, output)

        return self.outc(x)


class FluidUNet(UNet):
    """UNet model for fluid simulation.

    Corresponds to UNet2 in the original code.
    """

    def __init__(
        self, hidden_size: Optional[int] = 20, bilinear: Optional[bool] = True
    ):
        """Initialize standard UNet model with correct number of I/O channels.

        Args:
            hidden_size (int, optional): number of hidden features
            bilinear (bool, optional): whether to use bilinear interpolation
        """
        super().__init__(13, 2, hidden_size, 4, bilinear)

    def forward(
        self,
        a: torch.tensor,
        p: torch.tensor,
        v: torch.tensor,
        flow_mask: torch.tensor,
        b_mask: torch.tensor,
        b_cond: torch.tensor,
    ) -> torch.tensor:
        """Forward pass.

        Args:
            a (torch.tensor[:, 1, :, :]): vector potential
            p (torch.tensor[:, 1, :, :]): pressure
            v (torch.tensor[:, 2, :, :]): velocity
            flow_mask (torch.tensor[:, 3, :, :]): flow mask, containing the \
                domain marker
            b_mask (torch.tensor[:, 3, :, :]): boundary mask, containing the \
                boundary marker
            b_cond (torch.tensor[:, 3, :, :]): boundary conditions, containing \
                the boundary conditions for u, v, p

        Returns:
            torch.tensor[:, 1, :, :] vector potential, torch.tensor[:, 1, :, :] \
                pressure
        """
        #assert flow_mask.shape[1] == 3, "input tensor must have 3 channels"
        #assert b_mask.shape[1] == 3, "input tensor must have 3 channels"
        #assert b_cond.shape[1] == 3, "input tensor must have 3 channels"

        x = torch.cat(
            [
                p,  # pressure
                a,  # vector potential
                v,  # velocity
                flow_mask,  # domain marker
                b_cond * b_mask,  # boundary conditions
                b_mask,  # boundary marker
                flow_mask * p,  # pressure in domain
                flow_mask * v,  # velocity in domain
                b_mask * v,  # velocity on boundary
            ],
            dim=1,
        )
        # We have x stacked as [p, a, v, ...], v is already on staggered grid.
        x = super().forward(x)
        a = 400 * torch.tanh((x[:, 0:1] + a) / 400)
        p = 10 * torch.tanh((x[:, 1:2] + p) / 10)
        return a, p

def get_Net(params):
	if params.net == "UNet1":
		pde_cnn = PDE_UNet1(params.hidden_size)
	elif params.net == "UNet2":
		#pde_cnn = PDE_UNet2(params.hidden_size)
		pde_cnn = FluidUNet(params.hidden_size)
	elif params.net == "UNet3":
		pde_cnn = PDE_UNet3(params.hidden_size)
	return pde_cnn

class PDE_UNet1(nn.Module):
	#inspired by UNet taken from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
	
	def __init__(self, hidden_size=64,bilinear=True):
		super(PDE_UNet1, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear

		self.inc = DoubleConv(13, hidden_size)
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, 3)

	def forward(self,a_old,p_old,mask_flow,v_cond,mask_cond):
		v_old = curl(a_old)
		x = torch.cat([p_old,a_old,v_old,mask_flow,v_cond*mask_cond,mask_cond,mask_flow*p_old,mask_flow*v_old,v_old*mask_cond],dim=1)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		a_new, p_new = 400*torch.tanh(x[:,0:1]/400), 10*torch.tanh(x[:,1:2]/10)
		return a_new,p_new

class PDE_UNet2(nn.Module):
	#same as UNet1 but with delta a / delta p
	
	def __init__(self, hidden_size=64,bilinear=True):
		super(PDE_UNet2, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear

		self.inc = DoubleConv(13, hidden_size)
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, 2)

	def forward(self,a_old,p_old,mask_flow,v_cond,mask_cond):
		v_old = curl(a_old)
		x = torch.cat([p_old,a_old,v_old,mask_flow,v_cond*mask_cond,mask_cond,mask_flow*p_old,mask_flow*v_old,v_old*mask_cond],dim=1)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		a_new, p_new = 400*torch.tanh((a_old+x[:,0:1])/400), 10*torch.tanh((p_old+x[:,1:2])/10)
		return a_new,p_new


class PDE_UNet3(nn.Module):
	#same as UNet2 but with scaling
	
	def __init__(self, hidden_size=64,bilinear=True):
		super(PDE_UNet3, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear

		self.inc = DoubleConv(13, hidden_size)
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, 4)

	def forward(self,a_old,p_old,mask_flow,v_cond,mask_cond):
		v_old = curl(a_old)
		x = torch.cat([p_old,a_old,v_old,mask_flow,v_cond*mask_cond,mask_cond,mask_flow*p_old,mask_flow*v_old,v_old*mask_cond],dim=1)
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		a_new, p_new = 400*torch.tanh((a_old+x[:,0:1]*torch.exp(3*torch.tanh(x[:,2:3]/3)))/400), 10*torch.tanh((p_old+x[:,1:2]*torch.exp(3*torch.tanh(x[:,3:4]/3)))/10)
		return a_new,p_new

