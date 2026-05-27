import argparse
import os 
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import random
from datetime import datetime
import numpy as np
import time
from torch.optim.lr_scheduler import MultiStepLR
import json

from model.ascent import Ascent
from model.utils import TrajectoryDataset, seq_collate, pos_to_flight_params, ptsToLocal
from test import test


def seed():
    """Sets environment seeds and deterministic flags for reproducibility."""
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
    """Worker initialization seed handling for data loaders."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cal_loss(y_hat, pi, y, aux, config):
    """Calculates multimodal trajectory error using winner-take-all regression and cross entropy."""
    B = y_hat.shape[0]
    B_range = range(B)

    l2_norm = torch.norm(y_hat[..., :3] - y.unsqueeze(1), dim=-1).sum(-1)

    # Winner-Take-All strategy: update only the closest predicted mode
    best_mode = torch.argmin(l2_norm, dim=-1)
    y_hat_best = y_hat[B_range, best_mode]
    agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :3], y)
    agent_cls_loss = F.cross_entropy(pi, best_mode.detach())
   
    loss = agent_reg_loss + (agent_cls_loss if y_hat.shape[1] > 1 else 0)
    return loss


def train():
    """Main execution pipeline for configuration setup, file streaming, training, and model evaluation."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
    g = seed()

    # CLI parameter configuration
    parser=argparse.ArgumentParser(description='Train Ascent model')
    parser.add_argument('--dataset_folder',type=str, default='/dataset/', help="Path to dataset folder (default: /dataset/)")
    parser.add_argument('--dataset_name', type=str, default='7days1', help="Dataset split name (default: 7days1)")
    parser.add_argument('--obs', type=int, default=11, help="Observation length in seconds (default: 11)")
    parser.add_argument('--obs_steps', type=int, default=1, help="Steps between historical observations (default: 1)")
    parser.add_argument('--preds', type=int, default=120, help="Prediction length in seconds (default: 120)")
    parser.add_argument('--preds_step', type=int, default=10, help="Steps between predictions (default: 10)")
    parser.add_argument('--k', type=int, default=5, help="Number of modes (default: 5)")
    args = parser.parse_args()

    config = {
        "lr": 0.001,
        "epochs": 20,
        "k": args.k,
        "obs_len": args.obs,
        "obs_steps": args.obs_steps,
        "pred_len": args.preds,
        "pred_step": args.preds_step,
        "split_xy_z": True,
        "normalize_coords": True,
        "global_pos_embedding": True,
        "dataset_name": args.dataset_name,
        "batch_size": 32,
        "decoder": "detr"
    }

    print(config)

    # Initialize timestamped logging folder structures
    exp_dir = "runs/{}".format( datetime.now().strftime("%Y-%m-%d_%H-%M-%S") )
    os.makedirs(exp_dir)

    os.system('cp -a %s %s' % ('train.py', exp_dir))
    os.system('cp -a %s %s' % ('model/', exp_dir))

    with open("{}/config.json".format(exp_dir), "w") as f:
        json.dump(config, f) 
    with open("{}/log.txt".format(exp_dir), "w") as f:
        for key, val in config.items():
            f.write(f"{key} : {val}\n")
        f.write("\n")

    dataset_folder = args.dataset_folder
    dataset_name = args.dataset_name
    if "kbtp" not in dataset_name and "kagc" not in dataset_name:
        delim = " "
        subsample = 1
    else:
        delim = ","
        subsample = 2

    del args

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datapath = os.getcwd() + dataset_folder + dataset_name + "/processed_data/"

    print("Loading Train Data from ", datapath + "train")
    dataset_train = TrajectoryDataset(datapath + "train", obs_len=config["obs_len"], obs_steps=config["obs_steps"], pred_len=config["pred_len"], pred_step=config["pred_step"], delim=delim, subsample=subsample)

    print("Loading Test Data from ", datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=config["obs_len"], obs_steps=config["obs_steps"], pred_len=config["pred_len"], pred_step=config["pred_step"], delim=delim)

    loader_train = DataLoader(dataset_train, batch_size=config["batch_size"], num_workers=4, shuffle=True, collate_fn=seq_collate, worker_init_fn=seed_worker, generator=g)
    loader_test = DataLoader(dataset_test, batch_size=256, num_workers=4, shuffle=False, collate_fn=seq_collate, worker_init_fn=seed_worker, generator=g)

    model = Ascent(config)
    model.to(device)

    # checkpoint = torch.load('model_11.pt',map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)

    st = time.time()
    ade = -1
    best_fde = -1

    # Main optimization training loop over specified epochs
    for epoch in range(1, config["epochs"]+1):
        model.train()
        tot_batch_count = 0
        tot_loss = 0
        for data in tqdm(loader_train, disable=False):
            for k in data.keys():
                if torch.is_tensor(data[k]): data[k] = data[k].to(device)

            optimizer.zero_grad()
            y_hat, pi, aux = model(data)

            loss = cal_loss(y_hat, pi, torch.transpose(data["pred_traj"], 1, 0), aux, config)

            tot_loss += loss.item()
            tot_batch_count += 1
            loss.backward()
            optimizer.step()
        scheduler.step()

        loss = tot_loss/tot_batch_count

        model_path = "{}/model_{}_{:02d}.pt".format(exp_dir, dataset_name, epoch) 
        print("Epoch: {:2d} Train Loss: {:6.4f} Saving model at {}".format(epoch, loss, model_path))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_path)

        model.eval()

        # Evaluate current validation tracking performance metrics
        test_ade, test_fde = test(model, loader_test, device)
        print("[{:8.2f}] EPOCH: {} Test ADE: {:5.3f}  Test FDE: {:5.3f}".format(time.time()-st, epoch, test_ade, test_fde))
        with open("{}/log.txt".format(exp_dir), "a") as f:
            f.write("[{:7.2f}] EPOCH: {} Test ADE: {:5.3f}  Test FDE: {:5.3f}\n".format(time.time()-st, epoch, test_ade, test_fde))

        if best_fde < 0 or best_fde > test_fde: 
            ade = test_ade
            best_fde = test_fde

    print("BEST FDE: {:5.3f} - {:5.3f} & {:5.3f}".format(best_fde, ade, best_fde))
    with open("{}/log.txt".format(exp_dir), "a") as f:
            f.write("BEST FDE: {:5.3f} - {:5.3f} & {:5.3f}".format(best_fde, ade, best_fde))

    return


if __name__=='__main__':
    train()