import torch
import torch.nn.functional as F
from torch import nn
import math
import get_param
from typing import Optional

class Conv(nn.Module):
    """Convolutional block."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize Conv block.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.tensor):
        """Forward pass.

        Args:
            x (torch.tensor): input

        Returns:
            torch.tensor: convolved input, batch normalized and ReLU activated
        """
        x = self.batch_norm(self.conv(x))
        return self.relu(x)


class Down(nn.Module):
    """Downscaling."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize Downscaling block.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
        """
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.conv_1 = Conv(in_channels, out_channels)
        self.conv_2 = Conv(out_channels, out_channels)

    def forward(self, x: torch.tensor):
        """Forward pass.

        Args:
            x (torch.tensor): input

        Returns:
            torch.tensor: downscaled input through convolutions and max \
                pooling
        """
        x = self.conv_2(self.conv_1(self.max_pool(x)))
        return x


class Up(nn.Module):
    """Upscaling."""

    def __init__(
        self, in_channels: int, out_channels: int, bilinear: Optional[bool] = True
    ):
        """Initialize Upscaling block.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            bilinear (bool, optional): whether to use bilinear interpolation \
                instead of transposed convolution
        """
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv_1 = Conv(in_channels, in_channels // 2)
            self.conv_2 = Conv(in_channels // 2, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv_1 = Conv(in_channels, out_channels)
            self.conv_2 = Conv(out_channels, out_channels)

    def forward(self, x1: torch.tensor, x2: torch.tensor):
        """Forward pass.

        Args:
            x1 (torch.tensor): smaller input
            x2 (torch.tensor): larger input

        Returns:
            torch.tensor: upscaled input through padding and convolutions
        """
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv_2(self.conv_1(x))


class OutConv(nn.Module):
    """Output convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize Output convolution block.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.tensor):
        """Forward pass.

        Args:
            x (torch.tensor): input

        Returns:
            torch.tensor: convolved input
        """
        return self.conv(x)

params = get_param.params()

def toCuda(x):
	if type(x) is tuple:
		return [xi.cuda() if params.cuda else xi for xi in x]
	return x.cuda() if params.cuda else x

def toCpu(x):
	if type(x) is tuple:
		return [xi.detach().cpu() for xi in x]
	return x.detach().cpu()


# # First order derivatives (d/dx)

# dx_kernel = toCuda(torch.Tensor([-0.5,0,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
# def dx(v):
# 	return F.conv2d(v,dx_kernel,padding=(0,1))

# dx_left_kernel = toCuda(torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
# def dx_left(v):
# 	return F.conv2d(v,dx_left_kernel,padding=(0,1))

# dx_right_kernel = toCuda(torch.Tensor([0,-1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
# def dx_right(v):
# 	return F.conv2d(v,dx_right_kernel,padding=(0,1))

# # First order derivatives (d/dy)

# dy_kernel = toCuda(torch.Tensor([-0.5,0,0.5]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
# def dy(v):
# 	return F.conv2d(v,dy_kernel,padding=(1,0))

# dy_top_kernel = toCuda(torch.Tensor([-1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
# def dy_top(v):
# 	return F.conv2d(v,dy_top_kernel,padding=(1,0))

# dy_bottom_kernel = toCuda(torch.Tensor([0,-1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
# def dy_bottom(v):
# 	return F.conv2d(v,dy_bottom_kernel,padding=(1,0))

# # Curl operator

# def rot_mac(a):
# 	return torch.cat([-dx_right(a),dy_bottom(a)],dim=1)

# # Laplace operator

# #laplace_kernel = toCuda(torch.Tensor([[0,1,0],[1,-4,1],[0,1,0]]).unsqueeze(0).unsqueeze(1)) # 5 point stencil
# #laplace_kernel = toCuda(torch.Tensor([[1,1,1],[1,-8,1],[1,1,1]]).unsqueeze(0).unsqueeze(1)) # 9 point stencil
# laplace_kernel = toCuda(0.25*torch.Tensor([[1,2,1],[2,-12,2],[1,2,1]]).unsqueeze(0).unsqueeze(1)) # isotropic 9 point stencil
# def laplace(v):
# 	return F.conv2d(v,laplace_kernel,padding=(1,1))


# # mapping operators

# map_vx2vy_kernel = 0.25*toCuda(torch.Tensor([[0,1,1],[0,1,1],[0,0,0]]).unsqueeze(0).unsqueeze(1))
# def map_vx2vy(v):
# 	return F.conv2d(v,map_vx2vy_kernel,padding=(1,1))

# map_vx2vy_left_kernel = 0.5*toCuda(torch.Tensor([[0,1,0],[0,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(1))
# def map_vx2vy_left(v):
# 	return F.conv2d(v,map_vx2vy_left_kernel,padding=(1,1))

# map_vx2vy_right_kernel = 0.5*toCuda(torch.Tensor([[0,0,1],[0,0,1],[0,0,0]]).unsqueeze(0).unsqueeze(1))
# def map_vx2vy_right(v):
# 	return F.conv2d(v,map_vx2vy_right_kernel,padding=(1,1))

# map_vy2vx_kernel = 0.25*toCuda(torch.Tensor([[0,0,0],[1,1,0],[1,1,0]]).unsqueeze(0).unsqueeze(1))
# def map_vy2vx(v):
# 	return F.conv2d(v,map_vy2vx_kernel,padding=(1,1))

# map_vy2vx_top_kernel = 0.5*toCuda(torch.Tensor([[0,0,0],[1,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(1))
# def map_vy2vx_top(v):
# 	return F.conv2d(v,map_vy2vx_top_kernel,padding=(1,1))

# map_vy2vx_bottom_kernel = 0.5*toCuda(torch.Tensor([[0,0,0],[0,0,0],[1,1,0]]).unsqueeze(0).unsqueeze(1))
# def map_vy2vx_bottom(v):
# 	return F.conv2d(v,map_vy2vx_bottom_kernel,padding=(1,1))


# mean_left_kernel = 0.5*toCuda(torch.Tensor([1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
# def mean_left(v):
# 	return F.conv2d(v,mean_left_kernel,padding=(0,1))

# mean_top_kernel = 0.5*toCuda(torch.Tensor([1,1,0]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
# def mean_top(v):
# 	return F.conv2d(v,mean_top_kernel,padding=(1,0))

# mean_right_kernel = 0.5*toCuda(torch.Tensor([0,1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(2))
# def mean_right(v):
# 	return F.conv2d(v,mean_right_kernel,padding=(0,1))

# mean_bottom_kernel = 0.5*toCuda(torch.Tensor([0,1,1]).unsqueeze(0).unsqueeze(1).unsqueeze(3))
# def mean_bottom(v):
# 	return F.conv2d(v,mean_bottom_kernel,padding=(1,0))


# def staggered2normal(v):
# 	v[:,0:1] = mean_left(v[:,0:1])
# 	v[:,1:2] = mean_top(v[:,1:2])
# 	return v

# def normal2staggered(v):#CODO: double-check that! -> seems correct
# 	v[:,0:1] = mean_right(v[:,0:1])
# 	v[:,1:2] = mean_bottom(v[:,1:2])
# 	return v

class Kernels:
    """Kernels for the PDEs."""

    def __init__(self, device: torch.device):
        """Initialize kernels.

        Args:
            device (torch.device): device to use
        """
        self.laplace_kernel = 0.25*torch.tensor([[1,2,1],[2,-12,2],[1,2,1]], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        self.dx_kernel = torch.tensor([-0.5,0,0.5], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        self.dx_left_kernel = torch.tensor([-1,1,0], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        self.dx_right_kernel = torch.tensor([0,-1,1], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        self.dy_kernel = torch.tensor([-0.5,0,0.5], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1).unsqueeze(3)
        self.dy_top_kernel = torch.tensor([-1,1,0], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1).unsqueeze(3)
        self.dy_bottom_kernel = torch.tensor([0,-1,1], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1).unsqueeze(3)
        self.map_vx2vy_kernel = 0.25*torch.tensor([[0,1,1],[0,1,1],[0,0,0]], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        self.map_vx2vy_left_kernel = 0.5*torch.tensor([[0,1,0],[0,1,0],[0,0,0]], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        self.map_vx2vy_right_kernel = 0.5*torch.tensor([[0,0,1],[0,0,1],[0,0,0]], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        self.map_vy2vx_kernel = 0.25*torch.tensor([[0,0,0],[1,1,0],[1,1,0]], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        self.map_vy2vx_top_kernel = 0.5*torch.tensor([[0,0,0],[1,1,0],[0,0,0]], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        self.map_vy2vx_bottom_kernel = 0.5*torch.tensor([[0,0,0],[0,0,0],[1,1,0]], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        self.mean_left_kernel = 0.5*torch.tensor([1,1,0], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        self.mean_top_kernel = 0.5*torch.tensor([1,1,0], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1).unsqueeze(3)
        self.mean_right_kernel = 0.5*torch.tensor([0,1,1], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        self.mean_bottom_kernel = 0.5*torch.tensor([0,1,1], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1).unsqueeze(3)

kernels_cpu = Kernels(torch.device("cpu"))
if torch.cuda.is_available():
    kernels_cuda = Kernels(torch.device("cuda"))


# @torch.compile
def first_derivative(
    x: torch.tensor, dim: str, mode: Optional[str] = "central"
) -> torch.tensor:
    """Apply finite differences with padding.

    Args:
        x (torch.tensor[:, 1, :, :]): input
        dim (str): dimension to apply to, can be 'x' or 'y'
        mode (str, optional): FD scheme to apply, defaults to central,
            other options are 'left' or 'right'

    Returns:
        torch.tensor with finite differences applied

    Raises:
        ValueError if dim is not 'x' or 'y' or
            mode not 'central', 'left', 'right'
    """
    assert x.dim() == 4, "Input must be 4D tensor."
    assert x.size(1) == 1, "Input must have 1 channel."
    
    if x.device == torch.device("cpu"):
        k = kernels_cpu
    else:
        k = kernels_cuda

    if mode == "central":
        if dim == "x":
            kernel = k.dx_kernel
            padding = (0, 1)
        else:
            kernel = k.dy_kernel
            padding = (1, 0)
    elif mode == "left":
        if dim == "x":
            kernel = k.dx_left_kernel
            padding = (0, 1)
        else:
            kernel = k.dy_top_kernel
            padding = (1, 0)
    elif mode == "right":
        if dim == "x":
            kernel = k.dx_right_kernel
            padding = (0, 1)
        else:
            kernel = k.dy_bottom_kernel
            padding = (1, 0)
    else:
        raise ValueError("Wrong mode. Can be either central, left or right.")

    return F.conv2d(x, kernel, padding=padding)


# @torch.compile
def curl(x: torch.tensor) -> torch.tensor:
    """Apply curl to x.

    Args:
        x (torch.tensor[:, 1, :, :]): input

    Returns:
        torch.tensor curl(x)
    """
    assert x.dim() == 4, "Input must be 4D tensor."
    assert x.size(1) == 1, "Input must have 1 channel."

    return torch.cat(
        [-first_derivative(x, "x", "right"), first_derivative(x, "y", "right")], dim=1
    )


#@torch.compile
def laplacian(x: torch.tensor) -> torch.tensor:
    """Apply the Laplacian to x.

    Args:
        x (torch.tensor[:, 1, :, :]): input

    Returns:
        torch.tensor Laplacian(x)
    """
    assert x.dim() == 4, "Input must be 4D tensor."
    assert x.size(1) == 1, "Input must have 1 channel."
    
    if x.device == torch.device("cpu"):
        k = kernels_cpu
    else:
        k = kernels_cuda
    
    return F.conv2d(x, k.laplace_kernel, padding=(1, 1))



# @torch.compile
def vel_map(x: torch.tensor, to: str, mode: Optional[str] = "central") -> torch.tensor:
    """Map to other points on the staggered grid (velocities).

    Args:
        x (torch.tensor[:, :, :, :]): input, velocity
        to (str): 'x' or 'y', where to map to
        mode (str, optional): 'central', 'left' or 'right'

    Returns:
        torch.tensor mapped input

    Raises:
        ValueError if args not as required
    """
    assert x.dim() == 4, "Input must be 4D tensor."
    
    if x.device == torch.device("cpu"):
        k = kernels_cpu
    else:
        k = kernels_cuda

    if mode == "central":
        if to == "x":
            kernel = k.map_vy2vx_kernel
        else:
            kernel = k.map_vx2vy_kernel
    elif mode == "left":
        if to == "x":
            kernel = k.map_vy2vx_top_kernel
        else:
            kernel = k.map_vx2vy_left_kernel
    elif mode == "right":
        if to == "x":
            kernel = k.map_vy2vx_bottom_kernel
        else:
            kernel = k.map_vx2vy_right_kernel
    else:
        raise ValueError("Mode not supported.")

    return F.conv2d(x, kernel, padding=(1, 1))


# @torch.compile
def mean(x: torch.tensor, mode: str, dim: str) -> torch.tensor:
    """Calculate mean.

    Args:
        x (torch.tensor[:, 1, :, :]): input
        mode (str): 'left' or 'right'
        dim (str): 'x' or 'y'

    Returns:
        torch.tensor mean(x)

    Raises:
        ValueError if arguments not correct.
    """
    assert x.dim() == 4, "Input must be 4D tensor."
    assert x.size(1) == 1, "Input must have 1 channel."

    if x.device == torch.device("cpu"):
        k = kernels_cpu
    else:
        k = kernels_cuda

    if mode == "left":
        if dim == "x":
            kernel = k.mean_left_kernel
            padding = (0, 1)
        else:
            kernel = k.mean_top_kernel
            padding = (1, 0)
    elif mode == "right":
        if dim == "x":
            kernel = k.mean_right_kernel
            padding = (0, 1)
        else:
            kernel = k.mean_bottom_kernel
            padding = (1, 0)
    else:
        raise ValueError("mode not supported.")

    return F.conv2d(x, kernel, padding=padding)


# @torch.compile
def convert_grid(v: torch.tensor, to: str) -> torch.tensor:
    """Map velocities to other grid.

    Args:
        v (torch.tensor[:, 2, :, :]): 2-D velocities
        to (str): 'staggered' or 'collocated'

    Returns:
        torch.tensor velocities on other grid

    Raises:
        ValueError if argument to not correct.
    """
    assert v.dim() == 4, "Input must be 4D tensor."
    assert v.size(1) == 2, "Input must have 2 channels."

    if to == "staggered":
        v[:, 0:1] = mean(v[:, 0:1], "right", "x")
        v[:, 1:2] = mean(v[:, 1:2], "right", "y")
    elif to == "collocated":
        v[:, 0:1] = mean(v[:, 0:1], "left", "x")
        v[:, 1:2] = mean(v[:, 1:2], "left", "y")
    else:
        raise ValueError("to not supported.")

    return v

def vector2HSV(vector,plot_sqrt=False):
	"""
	transform vector field into hsv color wheel
	:vector: vector field (size: 2 x height x width)
	:return: hsv (hue: direction of vector; saturation: 1; value: abs value of vector)
	"""
	values = torch.sqrt(torch.sum(torch.pow(vector,2),dim=0)).unsqueeze(0)
	saturation = torch.ones(values.shape).cuda()
	norm = vector/(values+0.000001)
	angles = torch.asin(norm[0])+math.pi/2
	angles[norm[1]<0] = 2*math.pi-angles[norm[1]<0]
	hue = angles.unsqueeze(0)/(2*math.pi)
	hue = (hue*360+100)%360
	#values = norm*torch.log(values+1)
	values = values/torch.max(values)
	if plot_sqrt:
		values = torch.sqrt(values)
	hsv = torch.cat([hue,saturation,values])
	return hsv.permute(1,2,0).cpu().numpy()
