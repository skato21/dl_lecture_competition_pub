import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed

def collate_fn_test(batch):
    X_batch = torch.stack([item[0].float() for item in batch])
    subject_idxs_batch = torch.tensor([item[1] for item in batch])
    return X_batch, subject_idxs_batch

@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)
    
    # ------------------
    #    Dataloader
    # ------------------    
    test_set = ThingsMEGDataset("test", data_dir="data", resample_rate=100, filter_params={'order': 5, 'cutoff': 0.3, 'btype': 'low'}, scaling=True, baseline_correction=50)
    
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn_test
    )

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        num_classes=test_set.num_classes,
        seq_len=test_set.seq_len,
        in_channels=test_set.num_channels,
        hid_dim=128,
        p_drop=0.5,
        weight_decay=1e-4
    ).to(args.device)
    
  
    model_path = os.path.join(savedir, "model_best.pt")
    map_location = torch.device(args.device)
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    # ------------------
    #  Start evaluation
    # ------------------ 
    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device).float()).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission.npy"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")

if __name__ == "__main__":
    run()