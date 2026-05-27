import math
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import random


def seed():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.multiprocessing.set_start_method("spawn")
    np.set_printoptions(suppress=True, precision=4)
    torch.set_printoptions(linewidth=None, profile=None, sci_mode=None)
    torch.manual_seed(0)
    g = torch.Generator()
    g.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    return g


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loadConfig(exp_folder):
    with open(exp_folder + "/config.json", "r") as f:
        config = json.load(f)
    if "decoder" not in config.keys(): config["decoder"] = ""
    if "obs_steps" not in config.keys(): config["obs_steps"] = 1
    print(config)

    if "decoder" not in config.keys(): config["decoder"] = ""
    if "obs_steps" not in config.keys(): config["obs_steps"] = 1
    return config


def ptsToLocals(origin, theta, pts):
    """
    Args:
        origin: [B, 3] - origin points in global frame
        theta: [B] - rotation angles around z-axis (in radians)
        pts: [B, X, 3] - global points to be transformed to local frame
    Returns:
        local_pts: [B, X, 3] - points transformed to local frame
    """
    bs = pts.shape[0]
    origin = origin.view(bs, 1, 3)
    theta = theta.view(bs)

    pts_translated = pts.double() - origin.double()

    # Create inverse rotation matrices (rotation by -theta)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rotate_mat_inv = torch.stack([
        torch.stack([cos_theta, sin_theta, torch.zeros_like(theta)], dim=-1),
        torch.stack([-sin_theta, cos_theta, torch.zeros_like(theta)], dim=-1),
        torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=-1),
    ], dim=1) 

    local_pts = torch.matmul(pts_translated, rotate_mat_inv.transpose(1, 2).double())  # [B, X, 3]

    return local_pts


def ptsToLocal(origin, yaw, pitch, pts):
    """
    Args:
        origin: [B, 3] - origin points in global frame
        yaw: [B] - yaw angles around z-axis (in radians)
        pitch: [B] - pitch angles around y-axis (in radians)
        pts: [B, X, 3] - global points to be transformed to local frame
    Returns:
        local_pts: [B, X, 3] - points transformed to local frame
    """
    bs = pts.shape[0]
    origin = origin.view(bs, 1, 3)
    yaw = yaw.view(bs)
    pitch = pitch.view(bs)

    pts_translated = pts.double() - origin.double() 

    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    R_yaw_inv = torch.stack([
        torch.stack([cy, sy, torch.zeros_like(yaw)], dim=-1),
        torch.stack([-sy, cy, torch.zeros_like(yaw)], dim=-1),
        torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=-1),
    ], dim=1) 
    
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    R_pitch_inv = torch.stack([
        torch.stack([cp, torch.zeros_like(pitch), -sp], dim=-1),
        torch.stack([torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch)], dim=-1),
        torch.stack([sp, torch.zeros_like(pitch), cp], dim=-1),
    ], dim=1) 

    R_inv = torch.matmul(R_pitch_inv, R_yaw_inv)  
    
    local_pts = torch.matmul(pts_translated, R_inv.transpose(1, 2).double()) 

    return local_pts.float()


def ptsToGlobal(origin, yaw, pitch, pts):
    """
    Args:
        origin: [B, 3] - origin points in global frame
        yaw: [B] - yaw angles around z-axis (in radians)
        pitch: [B] - pitch angles around y-axis (in radians)
        pts: [B, X, 3] - local points to be transformed
    Returns:
        global_pts: [B, X, 3] - points transformed to global frame
    """
    bs = pts.shape[0]
    origin = origin.view(bs, 1, 3)
    yaw = yaw.view(bs)
    pitch = pitch.view(bs)

    # Rotation by pitch (around Y-axis)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    R_pitch = torch.stack([
        torch.stack([cp, torch.zeros_like(pitch), sp], dim=-1),
        torch.stack([torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch)], dim=-1),
        torch.stack([-sp, torch.zeros_like(pitch), cp], dim=-1),
    ], dim=1) 

    # Rotation by yaw (around Z-axis)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    R_yaw = torch.stack([
        torch.stack([cy, -sy, torch.zeros_like(yaw)], dim=-1),
        torch.stack([sy,  cy, torch.zeros_like(yaw)], dim=-1),
        torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=-1),
    ], dim=1) 

    # Combine: yaw after pitch
    R = torch.matmul(R_yaw, R_pitch)

    # Apply rotation then translate
    rotated_pts = torch.matmul(pts.double(), R.transpose(1, 2).double())
    global_pts = rotated_pts + origin.double()

    return global_pts.float()


def ptsToGlobalO(origin, theta, pts):
    """
    Args:
        origin: [B, 3] - origin points in global frame
        theta: [B] - rotation angles around z-axis (in radians)
        pts: [B, X, 3] - local points to be transformed
    Returns:
        global_trajectory: [B, X, 3] - points transformed to global frame
    """
    bs = pts.shape[0]
    origin = origin.view(bs, 1, 3)
    theta = theta.view(bs)

    # Create 3D rotation matrices around the z-axis
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rotate_mat = torch.stack([
        torch.stack([cos_theta, -sin_theta, torch.zeros_like(theta)], dim=-1),
        torch.stack([sin_theta,  cos_theta, torch.zeros_like(theta)], dim=-1),
        torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=-1),
    ], dim=1) 

    rotated_pts = torch.matmul(pts.double(), rotate_mat.transpose(1, 2).double()) 
    global_trajectory = rotated_pts + origin.double()

    return global_trajectory


def denormalize_points_3d(points_norm, p1, p2):
    d = torch.nn.functional.normalize(p1 - p2, dim=1)
    z = torch.tensor([0, 0, 1], device=points_norm.device, dtype=points_norm.dtype).expand_as(d)
    
    v = torch.cross(z, d) 
    c = (z * d).sum(dim=1, keepdim=True)
    s = v.norm(dim=1, keepdim=True)

    v_skew = torch.zeros(points_norm.shape[0], 3, 3, device=points_norm.device)
    v_skew[:, 0, 1], v_skew[:, 0, 2] = -v[:, 2],  v[:, 1]
    v_skew[:, 1, 0], v_skew[:, 1, 2] =  v[:, 2], -v[:, 0]
    v_skew[:, 2, 0], v_skew[:, 2, 1] = -v[:, 1],  v[:, 0]

    I = torch.eye(3, device=points_norm.device).unsqueeze(0)
    R_inv = I + v_skew + torch.bmm(v_skew, v_skew) * ((1 - c) / (s**2 + 1e-8)).unsqueeze(-1)

    return torch.bmm(points_norm, R_inv) + p1[:, None, :]


def normalize_points_3d(points):
    p1, p2 = points[:, -1], points[:, -2]
    d = torch.nn.functional.normalize(p1 - p2, dim=1)
    z = torch.tensor([0, 0, 1], device=points.device, dtype=points.dtype).expand_as(d)
    
    v = torch.cross(d, z)
    c = (d * z).sum(dim=1, keepdim=True)
    s = v.norm(dim=1, keepdim=True)
    
    # Rodrigues' rotation formula
    v_skew = torch.zeros(points.shape[0], 3, 3, device=points.device)
    v_skew[:, 0, 1], v_skew[:, 0, 2] = -v[:, 2],  v[:, 1]
    v_skew[:, 1, 0], v_skew[:, 1, 2] =  v[:, 2], -v[:, 0]
    v_skew[:, 2, 0], v_skew[:, 2, 1] = -v[:, 1],  v[:, 0]

    I = torch.eye(3, device=points.device).unsqueeze(0)
    R = I + v_skew + torch.bmm(v_skew, v_skew) * ((1 - c) / (s**2 + 1e-8)).unsqueeze(-1)

    return torch.bmm(points - p1[:, None, :], R), p1, p2


def pos_to_flight_params(positions, dt=1.0):
    """
    positions: tensor [B, T, 3] -> (x, y, z)
    Returns: tensor [B, T-1, 4] -> (speed, heading, vertical_speed, altitude)
    """

    # Velocity: difference between positions over time
    vel = (positions[:, 1:] - positions[:, :-1]) / dt  # [B, T-1, 3]
    dx, dy, dz = vel[..., 0], vel[..., 1], vel[..., 2]

    # Speed (horizontal ground speed)
    speed = torch.sqrt(dx**2 + dy**2)  # [B, T-1]

    # Heading (yaw) in radians
    heading = torch.atan2(dy, dx)  # [B, T-1]

    # Vertical speed
    vertical_speed = dz  # [B, T-1]

    # Altitude (just z at t+1)
    altitude = positions[:, 1:, 2]  # [B, T-1]

    return torch.stack([speed, heading, vertical_speed, altitude], dim=-1)  # [B, T-1, 4]


def flight_params_to_pos(flight_params, initial_pos, dt=1.0):
    """
    flight_params: [B, T, 4] -> (speed, heading, vertical_speed, altitude)
    initial_pos: [B, 3] -> starting position (x, y, z)
    Returns: [B, T+1, 3] positions
    """
    B, T, _ = flight_params.shape
    positions = [initial_pos]  # list of [B, 3]

    for t in range(T):
        speed = flight_params[:, t, 0]
        heading = flight_params[:, t, 1]
        vertical_angle = flight_params[:, t, 2]

        dx = speed * torch.cos(heading) * dt  # [B]
        dy = speed * torch.sin(heading) * dt  # [B]
        dz = speed * torch.sin(vertical_angle) * dt # [B]

        prev = positions[-1]  # [B, 3]
        new_pos = torch.stack([
            prev[:, 0] + dx,
            prev[:, 1] + dy,
            prev[:, 2] + dz,      
        ], dim=1)  # [B, 3]

        positions.append(new_pos)

    return torch.stack(positions, dim=1)  # [B, T+1, 3]


class TrajectoryDataset(Dataset):
    """Dataset and processing pipeline for handling historical and predictive multi-agent 
    trajectory paths, featuring raw coordinate parsing, relative motion mapping, and local disk caching.
    
    Modified from https://github.com/alexmonti19/dagnet and https://github.com/castacks/trajairnet
    """
    
    def __init__(
        self, data_dir, obs_len=11, obs_steps=1, pred_len=120, skip=1, pred_step=10,
        min_agent=0, delim=' ', subsample=1):
        """
        Initializes the dataset, loading trajectory history and target windows. It searches for 
        a matching pre-compiled disk `.pt` cache before falls back to manual raw-file compilation.

        Args:
            data_dir (str): Directory containing dataset text files structured as: <frame_id> <agent_id> <x> <y> ...
            obs_len (int): Total number of historical orientation time-steps parsed for input tracks.
            obs_steps (int): Subsampling temporal rate index applied onto the history window slices (e.g., 1 or 5).
            pred_len (int): Full temporal horizon size allocated for destination target sequences.
            skip (int): Frame stepping interval length utilized during sequence slicing loops.
            pred_step (int): Decimation stride index used to downsample continuous future trajectories.
            min_agent (int): Threshold setting the lower limit of valid co-existing agents required to retain a sequence.
            delim (str): Character splitting file record strings.
            subsample (int): Stride parameter modifying total visible item collections inside indexing tasks.
        """
        super(TrajectoryDataset, self).__init__()

        self.subsample = subsample
        self.max_agents_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.obs_steps = obs_steps
        self.pred_len = pred_len
        self.skip = skip
        self.pred_step = pred_step
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.seq_final_len = self.obs_len + int(math.ceil(self.pred_len/self.pred_step))
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_agents_in_seq = []
        seq_list = []
        seq_list_rel = []
        context_list = []
        agent_id_list = []
        start_idx_list = []

        assert self.obs_steps == 5 or self.obs_steps == 1

        # Setup automated unique filepath caching signature matching local data directory patterns
        cache_file = self.data_dir.replace("/processed_data/train", "").replace("/processed_data/test", "")
        cache_file = cache_file.replace("/dataset/", "/dataset/_cache/")
        cache_file = "{}_{}_{}_{}_{}.pt".format(cache_file, self.data_dir.split("/")[-1], self.obs_len, self.pred_len, self.pred_step)
  
        # Attempt to instantly bypass raw file parsing sequences by returning historical binary caches
        if os.path.exists(cache_file):
            cache_data = torch.load(cache_file)
            self.num_seq = cache_data["num_seq"]
            self.obs_traj = cache_data["obs_traj"]
            self.obs_context = cache_data["obs_context"]
            self.pred_traj = cache_data["pred_traj"]
            self.obs_traj_rel = cache_data["obs_traj_rel"]
            self.pred_traj_rel = cache_data["pred_traj_rel"]
            self.seq_start_end = cache_data["seq_start_end"]
            self.max_agents = cache_data["max_agents"]
            self.agent_id = cache_data["agent_id"]
            self.start_idx = cache_data["start_idx"]
            return 

        # Primary raw parsing and matrix conversion tracking loop
        for path in tqdm(all_files):
            # print(path)
            data = []
            with open(path, 'r') as f:
                for line in f:
                    if len(line) == 1: continue
                    line = line.strip().split(delim)
                    line = [float(i) for i in line]
                    data.append(line)
            data = np.asarray(data)
            if (len(data[:, 0]) == 0):
                print("File is empty")
                continue

            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            #print(frame_data[0], frame_data[1], frame_data[2], frame_data[3])
            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                
                agents_in_curr_seq = np.unique(curr_seq_data[:, 1]).astype(int)
                self.max_agents_in_frame = max(self.max_agents_in_frame, len(agents_in_curr_seq))
                
                curr_seq_rel = np.zeros((len(agents_in_curr_seq), 3,
                                         self.seq_final_len))
                curr_seq = np.zeros((len(agents_in_curr_seq), 3,self.seq_final_len ))
                curr_context =  np.zeros((len(agents_in_curr_seq), 2,self.seq_final_len ))
                num_agents_considered = 0

                for _, agent_id in enumerate(agents_in_curr_seq):
                    curr_agent_seq = curr_seq_data[curr_seq_data[:, 1] == agent_id, :]
                    tmp = int( curr_agent_seq[0, 0] )
                    pad_front = frames.index(curr_agent_seq[0, 0]) - idx
                    pad_end = frames.index(curr_agent_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_agent_seq = np.transpose(curr_agent_seq[:, 2:])
                    obs = curr_agent_seq[:, :obs_len]
                    pred = curr_agent_seq[:, obs_len+pred_step-1::pred_step]
                    curr_agent_seq = np.hstack((obs,pred))
                    context = curr_agent_seq[-2:,:]
                    assert(~np.isnan(context).any())
                    
                    # Convert absolute positional tracking vectors into temporal velocity increments
                    rel_curr_agent_seq = np.zeros(curr_agent_seq.shape)
                    rel_curr_agent_seq[:, 1:] = \
                        curr_agent_seq[:, 1:] - curr_agent_seq[:, :-1]

                    _idx = num_agents_considered

                    if (curr_agent_seq.shape[1]!=self.seq_final_len):
                        continue
                   
                    curr_seq[_idx, :, pad_front:pad_end] = curr_agent_seq[:3,:]
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_agent_seq[:3,:]
                    curr_context[_idx,:,pad_front:pad_end] = context
                    num_agents_considered += 1
                    start_idx = tmp         
                    
                if num_agents_considered > min_agent:
                    num_agents_in_seq.append(num_agents_considered)
                    seq_list.append(curr_seq[:num_agents_considered])
                    seq_list_rel.append(curr_seq_rel[:num_agents_considered])
                    context_list.append(curr_context[:num_agents_considered])
                    agent_id_list.append(agents_in_curr_seq[:num_agents_considered])
                    start_idx_list.append([start_idx]*num_agents_considered)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        context_list = np.concatenate(context_list, axis=0)
        agent_id_list = np.concatenate(agent_id_list, axis=0)
        start_idx_list = np.concatenate(start_idx_list, axis=0)

        # Build final processing memory objects using PyTorch Tensor allocations
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_context = torch.from_numpy(context_list[:,:,:self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.agent_id = torch.from_numpy(agent_id_list).type(torch.int)
        self.start_idx = torch.from_numpy(start_idx_list).type(torch.int)
        
        cum_start_idx = [0] + np.cumsum(num_agents_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.max_agents = -float('Inf')
        for (start, end) in self.seq_start_end:
            n_agents = end - start
            self.max_agents = n_agents if n_agents > self.max_agents else self.max_agents
        
        # Serialize compiled records to local storage disk locations
        cache_data = {
            "num_seq": self.num_seq,
            "obs_traj": self.obs_traj,
            "obs_context": self.obs_context,
            "pred_traj": self.pred_traj,
            "obs_traj_rel": self.obs_traj_rel,
            "pred_traj_rel": self.pred_traj_rel,
            "seq_start_end": self.seq_start_end,
            "max_agents": self.max_agents,
            "agent_id": self.agent_id,
            "start_idx": self.start_idx
        }
        torch.save(cache_data, cache_file)
        return

    def __len__(self):
        """Returns total valid slice sequences available inside collection index frameworks."""
        return int(self.num_seq / self.subsample)
    
    def __max_agents__(self):
        """Returns the highest number of interacting actors identified within a single sequence context frame."""
        return self.max_agents

    def __getitem__(self, index):
        """Retrieves and packages tracking tensors corresponding to a indexed scene context segment.

        Args:
            index (int): Dataset query sample identifier index.

        Returns:
            list: Contains slice sets containing observed trajectories, future trajectory arrays, 
                  relative velocities, target displacements, environment contexts, agent tags, and file entry pointers.
        """
        index = min(index * self.subsample, self.num_seq)
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :], self.obs_context[start:end, :],
            self.agent_id[start:end], self.start_idx[start:end]
        ]

        # Conditionally downsample observation timelines if designated down-sampling rates are configured
        if self.obs_steps == 5:
            for i in [0, 2, 4]:
                out[i] = out[i][..., ::self.obs_steps]
        return out


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, context_list, agent_id, start_idx) = zip(*data)

    # Data format: batch, input_size, seq_len

    # LSTM input format: seq_len, batch, input_size
    adj = torch.tensor([i for i, sublist in enumerate(obs_seq_list) for _ in sublist])
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    context = torch.cat(context_list, dim=0 ).permute(2, 0, 1)
    agent_id = [int(item) for tensor in agent_id for item in tensor]
    start_idx = [int(item) for tensor in start_idx for item in tensor]

    # TRANSFORMER input format: batch, seq_len, input_size
    """obs_traj = torch.cat(obs_seq_list, dim=0).permute(0, 2, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(0, 2, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(0, 2, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(0, 2, 1)
    context = torch.cat(context_list, dim=0 ).permute(0, 2, 1)
    seq_start_end = torch.LongTensor(seq_start_end)"""

    out = {
        "obs_traj": obs_traj,
        "obs_traj_rel": obs_traj_rel,
        "pred_traj": pred_traj,
        "context": context,
        "adj": adj,
        "agent_id": agent_id,
        "start_idx": start_idx
    }
    return out

