import argparse
import os
from tqdm import tqdm
import numpy as np
import json
import torch
from torch.utils.data import DataLoader
import time

from model.cv import ConstantVelocityModel
from model.ascent import Ascent
from model.utils import TrajectoryDataset, seq_collate, loadConfig


def main():
    """Main execution block to set up parsing arguments, load configurations, 
    initialize data loaders, restore the model checkpoint, and kick off validation testing.
    """
    torch.set_printoptions(linewidth=None, profile=None, sci_mode=None)
    np.set_printoptions(suppress=True, precision=4)
    
    ##Dataset params
    parser=argparse.ArgumentParser(description='Visualize Ascent model')
    parser.add_argument('--dataset_folder',type=str, default='/dataset/')
    parser.add_argument('--dataset_name', type=str, default='')
    parser.add_argument('--exp_folder', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=1)
    args = parser.parse_args()


    parser.add_argument('--store', action="store_true")
    args = parser.parse_args()

    exp_folder = args.exp_folder
    config = loadConfig(exp_folder)

    dataset_folder = args.dataset_folder
    dataset_name = args.dataset_name if args.dataset_name != '' else config["dataset_name"]
    train_dataset_name = config["dataset_name"]
    epoch = args.epoch

    del args

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datapath = os.getcwd() + dataset_folder + dataset_name + "/processed_data/"
    print("Loading Test Data from ", datapath + "test")
    dataset_test = TrajectoryDataset(datapath + "test", obs_len=config["obs_len"], obs_steps=config["obs_steps"], pred_len=config["pred_len"], pred_step=config["pred_step"], delim=' ')
    loader_test = DataLoader(dataset_test, batch_size=256, num_workers=4, shuffle=False, collate_fn=seq_collate)

    model = Ascent(config)
    model.to(device)
    model.eval()
    
    # Locate and map the chosen epoch parameters into the architecture instance
    chkpt_path = '{}/model_{}_{:02d}.pt'.format(exp_folder, train_dataset_name, epoch)
    print("LOAD", chkpt_path)
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Baseline alternative model configuration path for debugging evaluation runs
    #model = ConstantVelocityModel() 
    #model.to(device)
    #model.eval()

    test_ade, test_fde = test(model, loader_test, device, filter=False)
    
    print("Test ADE: {:5.3f}  Test FDE: {:5.3f}".format(test_ade, test_fde))
    return


def test(model, loader_test, device, filter=False):
    """Evaluates performance by calculating Average Displacement Error (ADE), 
    Final Displacement Error (FDE), and inference latencies over the test set.

    Args:
        model (nn.Module): The model architecture being assessed.
        loader_test (DataLoader): Data framework feeding validation trajectories.
        device (torch.device): Device allocation hardware tag (CPU/CUDA).
        filter (bool): If True, filters targets based on spatial coordinate distance boundaries.

    Returns:
        tuple: (mean_ade, mean_fde) computed using best-of-K tracking evaluations.
    """
    tot_ade = []
    tot_fde = []
    max = 0
    maxx = 100000
    latencies = []
    for data in tqdm(loader_test, disable=False):
        for k in data.keys():
            if torch.is_tensor(data[k]):
                data[k] = data[k].to(device)
                if k != "adj": data[k] = data[k][:, :maxx] 

        # Synchronize GPU threads before timing to secure precise latency metrics
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            y_hat, __, __ = model(data)
        torch.cuda.synchronize()
        latencies.append( (time.time()-start) )

        y = torch.transpose( data["pred_traj"], 1, 0).unsqueeze(1)

        m1 = torch.norm( torch.transpose(data["obs_traj"], 1, 0)[..., :2], dim=-1).max().item()
        m2 = torch.norm( torch.transpose(data["pred_traj"], 1, 0)[..., :2], dim=-1).max().item()
        if m1 > max: max = m1
        if m2 > max: max = m2
        
        # Eliminate static sequences or anomalies within small specified radius thresholds
        if filter:
            mask = (torch.norm( torch.transpose(data["obs_traj"], 1, 0)[..., :2], dim=-1).max(-1).values < 5.8) & (torch.norm( torch.transpose(data["pred_traj"], 1, 0)[..., :2], dim=-1).max(-1).values < 5.8)
            mask  = ~mask
            y_hat = y_hat[mask]
            y = y[mask]
            if mask.sum() == 0: continue
            if torch.min(torch.norm(y_hat[:, :, -1] - y[:, :, -1], dim=-1), dim=1).values.mean().isnan():
                print(mask.sum(), mask.shape)
                print(y_hat[:, :, -1] )

        # Calculate metrics choosing the path closest to target values across all K predictions
        tot_ade.append( torch.min(torch.norm(y_hat - y, dim=-1).mean(dim=-1), dim=1).values.mean().item() )
        tot_fde.append( torch.min(torch.norm(y_hat[:, :, -1] - y[:, :, -1], dim=-1), dim=1).values.mean().item() )
    print(max)
    print( "mean latency: {:5.3f}ms".format(np.mean(np.array(latencies))*1000) )
    tot_ade = np.mean(np.array(tot_ade))
    tot_fde = np.mean(np.array(tot_fde))
    return tot_ade, tot_fde


if __name__=='__main__':
    main()
