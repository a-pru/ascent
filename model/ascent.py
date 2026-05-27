import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .transformer_blocks import Block
from .utils import flight_params_to_pos, ptsToGlobal, ptsToLocal


class Projector(nn.Module):
    """A multi-layer perceptron (MLP) module used to project feature vectors 
    from an arbitrary input dimension to a specified embedding dimension.
    
    Attributes:
        embed_dim (int): The dimensionality of the output embedding.
        proj (nn.Sequential): Sequential container of Linear and ReLU layers.
    """
    def __init__(self, dim, embed_dim=128):
        super().__init__()

        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
    
    def forward(self, data):
        """Forward pass for projecting input features.
        Args:
            data (torch.Tensor): Input tensor of shape (..., dim).
            
        Returns:
            torch.Tensor: Projected features of shape (..., embed_dim).
        """
        return self.proj(data)


class Ascent(nn.Module):
    """ASCENT Trajectory Prediction network module designed to model agent dynamics 
    and context configurations using Transformer blocks, outputting multi-modal 
    future trajectories and flight parameters.

    Attributes:
        attn_depth (int): Number of Transformer layers in the agent/scene blocks.
        k (int): Number of prediction modes (multimodal outputs).
        embed_dim (int): Internal hidden size/embedding dimension.
        global_pos_embedding (bool): If True, adds global position embedding to agent features.
        normalize_coords (bool): If True, standardizes trajectories into local reference frames.
        future_steps (int): Total number of discrete future time steps to predict.
        decoder (str/obj): Configuration or indicator for the chosen decoding strategy.
        agent_blks (nn.ModuleList): Sequence of Transformer Blocks for agent-level encoding.
        agent_xy_proj (Projector): Projector for 2D horizontal coordinates.
        agent_z_proj (Projector): Projector for vertical depth/altitude coordinates.
        agent_ts_proj (Projector): Projector for temporal step/index tracking.
        loc (nn.Sequential): MLP mapping to direct 3D future coordinate regressions.
        fp1 (nn.Sequential): MLP predicting the 1st flight parameter feature set over time (speed)
        fp2 (nn.Sequential): MLP predicting the 2nd flight parameter feature set (yaw).
        fp3 (nn.Sequential): MLP predicting the 3rd flight parameter feature set (pitch).
        pi (nn.Sequential): MLP calculating probability logit distributions across modes.
        pos_embed (nn.Sequential): MLP encoding spatial orientation attributes.
        norm (nn.LayerNorm): Layer normalization applied prior to core decoding sequences.
        type_embed (nn.Parameter): Learnable parameters identifying agent categories.
        mode1_embed (nn.Parameter): Learnable multimodal query embeddings for trajectory branching.
        obs (int): Context historical observation timeline duration length.
        obs_steps (int): Total discrete temporal ticks monitored in the history.
        cache (dict): Runtime storage dictionary for analytical metrics or state variables.
    """
    def __init__(self, config):
        super(Ascent, self).__init__()
        self.attn_depth = 2
        self.k = config["k"]
        self.embed_dim = 128
        
        self.global_pos_embedding = config["global_pos_embedding"]
        self.normalize_coords = config["normalize_coords"]
        self.future_steps = config["pred_len"] // config["pred_step"]
        self.decoder = config["decoder"]

        # Encode agent dynamics
        dpr = [x.item() for x in torch.linspace(0, 0.2, self.attn_depth)]
        self.agent_blks = nn.ModuleList(
            Block(
                dim=self.embed_dim,
                num_heads=8,
                mlp_ratio=4.0,
                qkv_bias=False,
                drop_path=dpr[i],
            )
            for i in range(self.attn_depth)
        ) 

        self.hist_embed = nn.Linear(3*12, self.embed_dim) # unused (legacy)

        # Project to feature space
        self.agent_xyz_proj = Projector(3) # unused (legacy)
        self.agent_xy_proj = Projector(2)
        self.agent_z_proj = Projector(1)
        self.agent_ts_proj = Projector(1)

        # Unused (legacy)
        self.scene_blks = nn.ModuleList(
                Block(
                    dim=self.embed_dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=False,
                    drop_path=dpr[i],
                )
                for i in range(self.attn_depth)
            )

        #############################################################
        # Decoders & Predictive Heads
        #############################################################

        # Direct 3D Cartesian sequence output head
        self.loc = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.future_steps * 3),
        )

        # Kinematic / Flight Parameter prediction heads (e.g. speed, yaw, pitch) for structured trajectory reconstruction
        self.fp1 = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.future_steps),
        )
        self.fp2 = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.future_steps*2),
        )
        self.fp3 = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.future_steps*2),
        )

        # Score head for each predicted mode
        self.pi = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

        # Positional embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(7, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.norm = nn.LayerNorm(self.embed_dim)

        # Learnable categorical class tokens
        self.type_embed = nn.Parameter(torch.Tensor(5, self.embed_dim))

        # Mode tokens
        self.mode1_embed = nn.Parameter(torch.Tensor(self.k, self.embed_dim))

        self.obs = config["obs_len"]
        self.obs_steps = config["obs_steps"]

        self.cache = {}
        self.init_weights()

        return


    def init_weights(self):
        """Initializes model parameters using random normal distributions."""
        nn.init.normal_(self.type_embed, std=0.02)
        nn.init.normal_(self.mode1_embed, std=0.02)
        return
    

    def compute_angles(self, actor_feat): 
            """Computes the heading (yaw) and elevation (pitch) angles of an actor based on 
            the displacement between its last two observed positions.

            This method projects 3D spatial velocity into trigonometric components (sine and cosine) 
            to provide a continuous, wrap-around-safe directional encoding for the model.

            Args:
                actor_feat (torch.Tensor): Tensor containing observed actor trajectories of shape 
                    (Batch, Time_Steps, 3), where the last dimension represents (X, Y, Z) coordinates.

            Returns:
                tuple: A tuple containing:
                    - actor_angles_sincos (torch.Tensor): Normalized directional features of shape (Batch, 4),
                    ordered as [cos(yaw), sin(yaw), cos(pitch), sin(pitch)].
                    - yaw (torch.Tensor): Raw yaw (azimuth) angles in radians of shape (Batch,).
                    - pitch (torch.Tensor): Raw pitch (elevation) angles in radians of shape (Batch,).
            """
            delta = actor_feat[:, -1, :] - actor_feat[:, -2, :]  # [B, 3] — [dx, dy, dz]
            # Compute yaw (azimuth) angle in XY plane
            yaw = torch.atan2(delta[:, 1], delta[:, 0])  # atan2(dy, dx)
            # Compute pitch (elevation) angle in Z relative to XY plane
            horizontal_norm = torch.norm(delta[:, :2], dim=-1) + 1e-8  # sqrt(dx^2 + dy^2)
            pitch = torch.atan2(delta[:, 2], horizontal_norm)  # atan2(dz, √(dx² + dy²))
            # Final directional encoding: sin/cos for yaw and pitch
            yaw_sincos = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)         # [B, 2]
            pitch_sincos = torch.stack([torch.cos(pitch), torch.sin(pitch)], dim=-1)  # [B, 2]
            # Concatenate for final representation: [cos(yaw), sin(yaw), cos(pitch), sin(pitch)]
            actor_angles_sincos = torch.cat([yaw_sincos, pitch_sincos], dim=-1)  # [B, 4]
            return actor_angles_sincos, yaw, pitch


    def forward(self, data):    
        """Executes full multi-modal trajectory inference.
        
        Args:
            data (dict): Dictionary comprising tracking data elements:
                - "obs_traj": Observed tracking tensor (Steps, Batch, 3)
                - "context": Ambient frame features or states (Steps, Batch, Channels)
                - "adj": System adjacency or connectivity maps.
                
        Returns:
            tuple: Contains:
                - loc (torch.Tensor): Final absolute trajectories [B, k, future_steps, 3]
                - pi (torch.Tensor): Scored probability distribution across tracks [B, k]
                - dict: Diagnostic metadata (flight parameters, centers, yaw offsets).
        """
        # Adjust dimensions for internal tracking structures
        actor_feat = torch.transpose(data["obs_traj"], 1, 0)
    
        B = actor_feat.shape[0]       
        ts = torch.arange(actor_feat.shape[1]).view(1, -1, 1).repeat(actor_feat.shape[0], 1, 1).to(actor_feat.device).float()

        ### ACTOR ENCODING ###
        actor_centers = actor_feat[:, -1]

        # Positional Embedding 
        actor_angles_sincos, yaw, pitch = self.compute_angles(actor_feat)
        pos_feat = torch.cat([actor_centers, actor_angles_sincos], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        # Coordinate transformation to egocentric local frame
        if self.normalize_coords:
            actor_feat = ptsToLocal(actor_centers, yaw, pitch, actor_feat).float() # actor_feat - actor_centers.unsqueeze(1) #

        # Route variables through specialized sub-dimension MLPs or unified 3D blocks
        actor_feat = self.agent_xy_proj(actor_feat[..., :2]) + self.agent_z_proj(actor_feat[..., 2].unsqueeze(-1))
        
        # Inject timeline markers and process through Agent Attention blocks
        actor_feat += self.agent_ts_proj(ts)
        for blk in self.agent_blks:
            actor_feat = blk(actor_feat)
        actor_feat = torch.max(actor_feat, axis=1).values

        # Layer semantic identifier and global spatial biases
        actor_types = self.type_embed[0].unsqueeze(0)
        actor_feat = actor_feat + actor_types
        if self.global_pos_embedding and self.normalize_coords:
            actor_feat += pos_embed

        ### SCENE ENCODING ###
        feat = torch.stack([actor_feat], dim=1)
        feat = self.norm(feat)
        
        ### DECODING ###
        feat = feat[:, 0]

        # Broadcast feature embeddings to multimodal query tokens for parallel mode-specific decoding
        modes1 = (self.mode1_embed.view(1, self.k, self.embed_dim).repeat(B, 1, 1)) 
        feat = feat.unsqueeze(1).repeat(1, self.k, 1) + modes1 

        # Flight parameter prediction heads and score head
        pi = self.pi(feat).squeeze(dim=-1)
        fp1 = self.fp1(feat) 
        fp2 = self.fp2(feat)
        fp3 = self.fp3(feat) 

        # Normalize trigonometric values and extract continuous angular parameters for yaw
        y_sin = fp2.view(B, self.k, -1, 2)[..., 0]
        y_cos = fp2.view(B, self.k, -1, 2)[..., 1]
        norm = torch.sqrt(y_sin**2 + y_cos**2 + 1e-8)
        y_sin = y_sin / norm
        y_cos = y_cos / norm
        fp2 = torch.atan2(y_sin, y_cos)

        # Normalize trigonometric values and extract continuous angular parameters for pitch
        y_sin = fp3.view(B, self.k, -1, 2)[..., 0]
        y_cos = fp3.view(B, self.k, -1, 2)[..., 1]
        norm = torch.sqrt(y_sin**2 + y_cos**2 + 1e-8)
        y_sin = y_sin / norm
        y_cos = y_cos / norm
        fp3 = torch.atan2(y_sin, y_cos)
        
        flight_params = torch.stack([fp1, fp2, fp3], dim=-1)
        init_pos =  torch.zeros_like(actor_centers.unsqueeze(1).repeat(1, self.k, 1).view(-1, 3))

        # Reconstruct structural 3D updates from parameter configurations
        loc = flight_params_to_pos(flight_params.view(B*self.k, -1, 3), init_pos).view(B, self.k, -1, 3)[:, :, 1:]

        # Project coordinate fields back into global space or construct spatial projections
        if self.normalize_coords:
            for k in range(self.k):
                loc[:, k] = ptsToGlobal(actor_centers, yaw, pitch, loc[:, k])
        else:
            loc = self.loc(feat).view(B, self.k, self.future_steps, 3)

        return loc, pi, {"flight_params": flight_params, "actor_centers": actor_centers, "actor_angles": yaw,}