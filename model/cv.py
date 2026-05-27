import torch
from torch import nn


class ConstantVelocityModel(nn.Module):
    """A baseline physics-based trajectory forecasting module that extrapolates 
    an object's future path assuming a constant velocity vector derived from 
    its two most recent observed states.
    
    Attributes:
        horizon (int): The total number of future timesteps (Tf) calculated 
            during the forward projection step before downsampling.
    """
    def __init__(self):
        super().__init__()
        self.horizon = 120  # Tf

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Predicts future 3D Cartesian coordinates based on a constant linear 
        velocity assumption.

        Args:
            data (dict): A dictionary containing input data, which must include:
                - "obs_traj" (torch.Tensor): Observed raw tracking coordinates 
                  of shape [T, B, 3].

        Returns:
            tuple: A tuple containing:
                - future (torch.Tensor): Downsampled projected trajectory coordinates 
                  of shape [B, 1, Tf // 10, 3].
                - None: Placeholder.
                - None: Placeholder.
        """
        # Permute from [T, B, 3] to [B, T, 3] for batch-major processing
        coords = torch.transpose(data["obs_traj"], 1, 0)

        # Estimate velocity from last two time steps
        vel = coords[:, -1] - coords[:, -2]  # [B, 3]
        last_pos = coords[:, -1]            # [B, 3]

        # Predict future positions: pos(t+1) = last_pos + vel * (t + 1)
        # Construct a temporal step sequence of shape [1, Tf, 1] to scale the velocity vector
        steps = torch.arange(1, self.horizon + 1, device=coords.device).float().view(1, self.horizon, 1)
        
        # Broadcast across batches and time horizons to calculate future coordinates: [B, Tf, 3]
        future = last_pos.unsqueeze(1) + steps * vel.unsqueeze(1)  # [B, Tf, 3]
        
        # Slices every 10th step along the time horizon axis to downsample predictions, 
        # matching output formatting constraints.
        return future.unsqueeze(1)[:, :, ::10], None, None
