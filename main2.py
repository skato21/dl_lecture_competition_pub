import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from scipy.signal import resample, butter, filtfilt
from torchmetrics import Accuracy
from transformers import Wav2Vec2Model, Wav2Vec2Config, CLIPModel
from torchvision import transforms
from PIL import Image
from glob import glob

from src.utils import set_seed

# ThingsMEGDataset Class
class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        self.split = split
        self.data_dir = data_dir
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{self.split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy")
        X = torch.from_numpy(np.load(X_path)).float()  # Ensure data is float
        X = X.unsqueeze(0)  # Add a singleton dimension to match expected input shape

        subject_idx_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy")
        subject_idx = torch.from_numpy(np.load(subject_idx_path)).long()

        if self.split in ["train", "val"]:
            y_path = os.path.join(self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy")
            y = torch.from_numpy(np.load(y_path)).long()

            if y.dim() == 0:  # If y is a scalar, add a dimension
                y = y.unsqueeze(0)

            #print(f"Sample {i}: X shape: {X.shape}, y shape: {y.shape}")
            return X, y, subject_idx
        else:
            #print(f"Sample {i}: X shape: {X.shape}")
            return X, subject_idx
        
    @property
    def num_channels(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[0]
    
    @property
    def seq_len(self) -> int:
        return np.load(os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")).shape[1]


# Function to preprocess the data
def preprocess_data(data, target_sampling_rate=256):
    current_sampling_rate = 200
    num_samples = int(data.shape[-1] * (target_sampling_rate / current_sampling_rate))
    data = resample(data, num_samples, axis=-1)
    
    lowcut = 0.5
    highcut = 40.0
    nyquist = 0.5 * target_sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype='band')
    data = filtfilt(b, a, data, axis=-1)
    
    data = (data - np.mean(data, axis=-1, keepdims=True)) / np.std(data, axis=-1, keepdims=True)
    baseline_period = 50
    baseline = np.mean(data[:, :baseline_period], axis=-1, keepdims=True)
    data = data - baseline
    
    return data

# Function to load image paths
def load_image_paths(file_path):
    base_dir = "/workspace/dl_lecture_competition_pub/data/Images/"
    with open(file_path, 'r') as f:
        paths = [base_dir + line.strip() for line in f]
    return paths

# Custom dataset for images
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Wav2Vec2EEGClassifier with Dropout and L2 Regularization
class Wav2Vec2EEGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(Wav2Vec2EEGClassifier, self).__init__()
        # Customize the configuration
        config = Wav2Vec2Config(
            conv_stride=(2, 2, 2, 2, 2, 2, 2),
            conv_kernel=(3, 3, 3, 3, 3, 2, 2),
            conv_padding=(1, 1, 1, 1, 1, 0, 0),
            mask_time_length=1  # Ensure mask length is smaller than sequence length
        )
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(config.hidden_size, num_classes)
    
    def forward(self, x):
        # Ensure x is of shape [batch_size, channels, sequence_length]
        #print(f"Input shape before reshape: {x.shape}")
        if x.dim() == 4:  # Case when input is [batch_size, 1, channels, sequence_length]
            x = x.squeeze(1)  # Remove the singleton dimension
            #print(f"Input shape after squeeze: {x.shape}")
        
        batch_size, channels, sequence_length = x.shape
        x = x.view(batch_size * channels, sequence_length)  # Reshape to [batch_size*channels, sequence_length]
        #print(f"Input shape after reshape: {x.shape}")
        
        x = self.wav2vec2(x).last_hidden_state
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.view(batch_size, channels, -1).mean(dim=1)  # Combine back into [batch_size, num_classes]
        return x
    

# CLIPPretrainedModel with Dropout
class CLIPPretrainedModel(nn.Module):
    def __init__(self, embed_dim):
        super(CLIPPretrainedModel, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, embed_dim)
    
    def forward(self, x):
        with torch.no_grad():
            x = self.clip_model.get_image_features(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Custom loss function with L1 regularization
def custom_loss_function(output, target, model, l1_lambda=0.01):
    # Print the shapes for debugging
    #print(f"Output shape: {output.shape}")
    #print(f"Target shape: {target.shape}")

    # Adjust target shape if necessary
    if target.dim() > 1:
        target = target.squeeze()  # Reduce dimensions if target is multi-dimensional
        #print(f"Adjusted Target shape: {target.shape}")

    # Cross-entropy loss
    ce_loss = F.cross_entropy(output, target)

    # L1 regularization
    l1_loss = sum(param.abs().sum() for param in model.parameters())
    
    # Combine losses
    loss = ce_loss + l1_lambda * l1_loss
    return loss

def accuracy(preds, target):
    # Ensure target shape matches preds shape
    if target.dim() > 1:
        target = target.squeeze()
    
    #print(f"Preds shape: {preds.shape}")
    #print(f"Adjusted Target shape for accuracy: {target.shape}")

    _, preds_max = torch.max(preds, 1)
    correct = (preds_max == target).sum().item()
    return correct / target.size(0)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    train_set = ThingsMEGDataset("train", args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # Load image paths
    train_image_paths = load_image_paths('/workspace/dl_lecture_competition_pub/data/train_image_paths.txt')
    val_image_paths = load_image_paths('/workspace/dl_lecture_competition_pub/data/val_image_paths.txt')

    # Create image datasets and loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    train_image_dataset = ImageDataset(train_image_paths, transform=transform)
    val_image_dataset = ImageDataset(val_image_paths, transform=transform)
    
    train_image_loader = torch.utils.data.DataLoader(train_image_dataset, batch_size=32, shuffle=True)
    val_image_loader = torch.utils.data.DataLoader(val_image_dataset, batch_size=32, shuffle=False)

    eeg_model = Wav2Vec2EEGClassifier(num_classes=train_set.num_classes).to(args.device)
    clip_model = CLIPPretrainedModel(embed_dim=eeg_model.wav2vec2.config.hidden_size).to(args.device)

    optimizer = torch.optim.Adam(list(eeg_model.parameters()) + list(clip_model.parameters()), lr=args.lr, weight_decay=1e-5)
    max_val_acc = 0
    accuracy = Accuracy(task="multiclass", num_classes=train_set.num_classes, top_k=10).to(args.device)

    # Pretraining with image data
    for epoch in range(args.pretrain_epochs):
        print(f"Pretrain Epoch {epoch+1}/{args.pretrain_epochs}")
        clip_model.train()
        for images in tqdm(train_image_loader, desc="Pretrain with images"):
            images = images.to(args.device)
            _ = clip_model(images)
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        eeg_model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)
            y_pred = eeg_model(X)
            loss = custom_loss_function(y_pred, y, eeg_model, l1_lambda=0.01)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = accuracy(y_pred, y.squeeze())  # Adjusted shape
            train_acc.append(acc.item())

        eeg_model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            with torch.no_grad():
                y_pred = eeg_model(X)
            val_loss.append(F.cross_entropy(y_pred, y.squeeze()).item())  # Adjusted shape
            val_acc.append(accuracy(y_pred, y.squeeze()).item())  # Adjusted shape

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(eeg_model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(eeg_model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    eeg_model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    eeg_model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(eeg_model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")

if __name__ == "__main__":
    run()