import torch
from torch import nn
from derivatives import *

class BoundaryLossMAC(nn.Module):
    def forward(self, v_new, v_cond, cond_mask_mac):
        return torch.mean(torch.square(cond_mask_mac * (v_new - v_cond))[:, :, 1:-1, 1:-1], dim=(1, 2, 3))
    
class NavierStokesLossMAC(nn.Module):
    def __init__(self, rho, mu, dt, integration):
        super().__init__()
        self.rho = rho
        self.mu = mu
        self.dt = dt
        self.integration = integration
    
    def forward(self, v_old, v_new, p_new, flow_mask_mac):
        if self.integration == "explicit":
            v = v_old
        elif self.integration == "implicit":
            v = v_new
        elif self.integration == "imex":
            v = (v_new+v_old)/2
        
        return torch.mean(torch.square(flow_mask_mac*(self.rho*((v_new[:, 1:2]-v_old[:, 1:2])/self.dt+v[:, 1:2]*first_derivative(v[:, 1:2], "x", "central")+0.5*(vel_map(v[:, 0:1], "x", "left")*first_derivative(v[:, 1:2], "y", "left")+vel_map(v[:, 0:1], "x", "right")*first_derivative(v[:, 1:2], "y", "right")))+first_derivative(p_new, "x", "left")-self.mu*laplacian(v[:, 1:2])))[:, :, 1:-1, 1:-1], dim=(1, 2, 3)) +\
                torch.mean(torch.square(flow_mask_mac*(self.rho*((v_new[:, 0:1]-v_old[:, 0:1])/self.dt+v[:, 0:1]*first_derivative(v[:, 0:1], "y", "central")+0.5*(vel_map(v[:, 1:2], "y", "left")*first_derivative(
                    v[:, 0:1], "x", "left")+vel_map(v[:, 1:2], "y", "right")*first_derivative(v[:, 0:1], "x", "right")))+first_derivative(p_new, "y", "left")-self.mu*laplacian(v[:, 0:1])))[:, :, 1:-1, 1:-1], dim=(1, 2, 3))
                
                
class GradPRegularizer(nn.Module):
    def forward(self, p_new):
        return torch.mean(
                (first_derivative(p_new, "x", "right")**2+first_derivative(p_new, "y", "right")**2)[:, :, 2:-2, 2:-2], dim=(1, 2, 3))