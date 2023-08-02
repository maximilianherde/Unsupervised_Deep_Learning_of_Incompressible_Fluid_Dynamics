"""Implements the losses."""

from typing import Optional

import torch
from torch import nn

import derivatives as kernels


class NavierStokesLossMAC(nn.Module):
    """Implements the Navier-Stokes loss for MAC grids."""

    def __init__(
        self, rho: float, mu: float, dt: float, integration: Optional[str] = "imex"
    ):
        """Initialize the loss.

        Args:
            rho (float): The density of the fluid.
            mu (float): The viscosity of the fluid.
            dt (float): The timestep size.
            integration (str, optional): The integration scheme to use. \
                Can be 'implicit', 'explicit' or 'imex'. Defaults to 'imex'.
        """
        super().__init__()
        self.rho = rho
        self.mu = mu
        self.dt = dt
        assert integration in [
            "implicit",
            "explicit",
            "imex",
        ], 'Integration scheme must be one of "implicit", "explicit" or "imex"'
        self.integration = integration

    def forward(
        self,
        v_old: torch.tensor,
        v_new: torch.tensor,
        p: torch.tensor,
        flow_mask: torch.tensor,
    ) -> torch.tensor:
        """Compute the loss.

        Applies NO reduction!

        Args:
            v_old (torch.tensor[:, 2, :, :]): The velocity at the previous timestep.
            v_new (torch.tensor[:, 2, :, :]): The velocity at the current timestep.
            p (torch.tensor[:, 1, :, :]): The pressure at the current timestep.
            flow_mask (torch.tensor[:, 3, :, :]): The flow mask at the current timestep.

        Returns:
            torch.tensor: The loss for every batch element.
        """
        #assert flow_mask.shape[1] == 3

        if self.integration == "imex":
            vel = 0.5 * (v_old + v_new)
        elif self.integration == "explicit":
            vel = v_old
        else:
            vel = v_new

        vel_t = (v_new - v_old) / self.dt
        u = vel[:, 1:2]
        v = vel[:, 0:1]

        # TODO: check correctness of this
        mom_x = flow_mask[:, :1] * (
            self.rho
            * (
                vel_t[:, 1:2]
                + u * kernels.first_derivative(u, "x") #/ 0.01
                + 0.5
                * (
                    kernels.vel_map(v, "x", "left")
                    * kernels.first_derivative(u, "y", "left") #/ 0.01
                    + kernels.vel_map(v, "x", "right")
                    * kernels.first_derivative(u, "y", "right") #/ 0.01
                )
            )
            + kernels.first_derivative(p, "x", "left") #/ 0.01
            - self.mu * kernels.laplacian(u) #/ 0.01**2
        )

        # TODO: check correctness of this
        mom_y = flow_mask[:, :1] * (
            self.rho
            * (
                vel_t[:, 0:1]
                + v * kernels.first_derivative(v, "y") #/ 0.01
                + 0.5
                * (
                    kernels.vel_map(u, "y", "left")
                    * kernels.first_derivative(v, "x", "left") #/ 0.01
                    + kernels.vel_map(u, "y", "right")
                    * kernels.first_derivative(v, "x", "right") #/ 0.01
                )
            )
            + kernels.first_derivative(p, "y", "left") #/ 0.01
            - self.mu * kernels.laplacian(v) #/ 0.01**2
        )

        # TODO: check slicing
        loss = torch.mean(
            torch.square(mom_x)[:, :, 1:-1, 1:-1], dim=(1, 2, 3)
        ) + torch.mean(torch.square(mom_y)[:, :, 1:-1, 1:-1], dim=(1, 2, 3))

        return loss


class BoundaryLossMAC(nn.Module):
    """Implements the boundary loss for MAC grids."""

    def forward(
        self, v: torch.tensor, b_mask: torch.tensor, b_cond: torch.tensor
    ) -> torch.tensor:
        """Compute the loss.

        Applies NO reduction!

        Args:
            v (torch.tensor[:, 2, :, :]): The velocity at the current timestep.
            b_mask (torch.tensor[:, 3, :, :]): The boundary mask at the current timestep.
            b_cond (torch.tensor[:, 3, :, :]): The boundary condition at the current timestep.

        Returns:
            torch.tensor: The loss for every batch element.
        """
        #assert b_mask.shape[1] == 3
        #assert b_cond.shape[1] == 3

        # TODO: check slicing
        loss = torch.mean(
            torch.square(b_mask[:, :2] * (v - b_cond[:, :2]))[:, :, 1:-1, 1:-1],
            dim=(1, 2, 3),
        )
        return loss


class GradPRegularizer(nn.Module):
    """Implements a regularizer for the gradient of the pressure."""

    def forward(self, p: torch.tensor) -> torch.tensor:
        """Compute the regularizer.

        Applies NO reduction!

        Args:
            p (torch.tensor[])
   
        Returns:
            torch.tensor: The regularizer for every batch element.
        """
        return torch.mean(
            (
                kernels.first_derivative(p, "x", "right") ** 2
                + kernels.first_derivative(p, "y", "right") ** 2
            )[:, :, 2:-2, 2:-2],
            dim=(1, 2, 3),
        )
