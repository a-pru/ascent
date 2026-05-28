import argparse
import os 
from tqdm import tqdm 
import torch
import numpy as np     
from torch.utils.data import DataLoader
import json

from model.ascent import Ascent
from model.utils import TrajectoryDataset, seq_collate, ptsToLocal, loadConfig, seed, seed_worker
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

agent_base = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#d62728",  # red
]

def agent_palette(base_color, n=5):
    """Generate n shades from a single agent base color."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "", ["black", base_color, "white"]
    )
    return [cmap(x) for x in np.linspace(0.25, 0.85, n)]


def plotWrapper(aerial, type, data, label=None, alpha=None, zorder=None, s=None, marker=None, color=None, linewidth=None, linestyle="--", axis="z"):
    if axis == "z":
        idx1, idx2 = 0, 1
    else:
        idx1, idx2 = 0, 2
    if aerial:
        theta = torch.tensor(np.radians(18.093))
        rotation_matrix = torch.tensor([
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta),  torch.cos(theta)]
        ]).float()
        data = data[:, :2] @ rotation_matrix.T
        data = data * 1000

    if type == "scatter":
        return plt.scatter(data[:, idx1], data[:, idx2], marker=marker, label=label, alpha=alpha, zorder=zorder, s=s, color=color)
    elif type == "plot":
        return plt.plot(data[:, idx1], data[:, idx2], color=color, zorder=zorder, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
    assert False


def visualize():
    g = seed()
    plt.rcParams["pdf.fonttype"] = 42
    torch.set_printoptions(linewidth=None, profile=None, sci_mode=None)
    np.set_printoptions(suppress=True, precision=4)
    
    ##Dataset params
    parser=argparse.ArgumentParser(description='Visualize Ascent model output')
    parser.add_argument('--dataset_folder',type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--exp_folder', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--axis', type=str, default="z")
    args = parser.parse_args()

    exp_folder = args.exp_folder
    config = loadConfig(exp_folder)

    dataset_folder = args.dataset_folder
    dataset_name = args.dataset_name if args.dataset_name != '' else config["dataset_name"]
    train_dataset_name = config["dataset_name"]
    epoch = args.epoch
    axis = args.axis

    assert axis in ["z", "y"]

    del args

    if "kbtp" not in dataset_name and "kagc" not in dataset_name:
        delim = " "
    else:
        delim = ","

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##Load test and train data
    datapath = os.getcwd() + dataset_folder + dataset_name + "/processed_data/"
    print("Loading Test Data from ", datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=config["obs_len"], obs_steps=config["obs_steps"], pred_len=config["pred_len"], pred_step=config["pred_step"], delim=delim, subsample=10)
    loader_test = DataLoader(dataset_test, batch_size=32, num_workers=1, shuffle=False, collate_fn=seq_collate, worker_init_fn=seed_worker, generator=g)

    model = Ascent(config)
    model.to(device)
    chkpt_path = '{}/model_{}_{:02d}.pt'.format(exp_folder, train_dataset_name, epoch)
    print("LOAD", chkpt_path)
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    alpha = np.linspace(1.0, 0.2, 12)
    scene_index = 0
    actor_in_fig_counter = 0
    start_idx = -1

    for data in tqdm(loader_test):
        for k in data.keys():
            if torch.is_tensor(data[k]): data[k] = data[k].to(device)

        y_hat, pi, __ = model(data)

        for i, pred in enumerate(y_hat):
            scene_index += 1
            actor_in_fig_counter += 1

            # only plot scenarios with at least 4 agents, and only plot each scenario once (since we iterate over agents)
            if data["start_idx"].count(data["start_idx"][i]) < 3.5:
                continue

            if data["start_idx"][i] != start_idx:
                if start_idx > 0:
                    scaling = 1
                    if axis == "z":
                        plt.plot((0, 1.45*scaling), (0.0, 0.0), linewidth=10, alpha=0.3, color=[0.1, 0.1, 0.1])
                        plt.xlim([-5.5*scaling, 7*scaling])
                        plt.ylim([-5*scaling, 5*scaling])   
                        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v/1000:.0f}"))
                        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f"{v/1000:.0f}"))
                        plt.xlabel(r"$x$ [km]", fontsize=12)
                        plt.ylabel(r"$y$ [km]", fontsize=12)
                    else:
                        plt.xlim([-4.5, 6])
                        plt.ylim([0, 2])      
                        plt.xlabel(r"$x$ [km]", fontsize=12)
                        plt.ylabel(r"$z$ [km]", fontsize=12)
                    plt.xticks(fontsize=12)
                    plt.yticks(fontsize=12)

                if start_idx > 0:
                    plt.legend(fontsize=8)
                    plt.title(title_str[:-1], fontsize=8)
                    
                    plt.show()
                start_idx = data["start_idx"][i] 
                fig, ax = plt.subplots(figsize=(10, 6))
                actor_in_fig_counter = 0
                #print("new fig", data["start_idx"].count(data["start_idx"][i]), data["start_idx"], data["start_idx"][i])

                title_str = ""

            ########################################

            history = data["obs_traj"][:, i].cpu()
            pred = pred.cpu().detach()
            gt = data["pred_traj"][:, i].cpu()

            # Plot history
            plotWrapper(False, type="plot", data=history, zorder=997, axis=axis, linewidth=3, linestyle="-", color="black", alpha=1)

            # Plot predictions
            colors = agent_palette(agent_base[actor_in_fig_counter%4])
            scores = torch.nn.functional.softmax(pi[i]).cpu()
            for pred_idx in range(pred.shape[0]):
                if scores[pred_idx] < 0.05: continue
                sc = plotWrapper(False, type="scatter", data=pred[pred_idx], label="{} - {} ({:5.3f})".format(data["agent_id"][i], pred_idx, scores[pred_idx].item()), alpha=alpha, zorder=999, axis=axis, color=colors[pred_idx])
                pred_tmp = torch.cat([history[-1].unsqueeze(0), pred[pred_idx]], dim=0)
                plotWrapper(False, type="plot", data=pred_tmp, linestyle="--", color=sc.get_facecolor()[0], zorder=999, axis=axis, linewidth=3)

            # Plot ground truth
            gt_tmp = torch.cat([history[-1].unsqueeze(0), gt], dim=0)
            plotWrapper(False, type="plot", data=gt_tmp, linestyle="-", alpha=0.7, linewidth=9, color="#808080", zorder=998, axis=axis)

            ########################################

            gt = gt.unsqueeze(0)
            fde, fidx = torch.min(torch.norm(pred[:, -1] - gt[:, -1], dim=-1), dim=0)
            title_str += "Agent ID {} FDE: {:5.3f} (best mode idx: {})\n".format(data["agent_id"][i], fde, fidx)

            
    return


if __name__=='__main__':
    visualize()