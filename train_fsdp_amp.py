import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchvision import models
from torch.optim import Adam
from torch.amp import GradScaler, autocast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
import numpy as np
import time
import csv
import subprocess

# Dataset class
class LeukemiaDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Setup
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# GPU Utilization Logger
def get_gpu_utilization():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,nounits,noheader"
        ]).decode("utf-8").strip().split("\n")
        return [list(map(int, line.split(", "))) for line in output]
    except:
        return []

# CSV Logger
def log_results(run_type, gpu_type, num_gpus, epochs, total_time, speedup=None, efficiency=None):
    os.makedirs("csv_files", exist_ok=True)
    log_file = os.path.join("csv_files", f"fsdp_amp_gpu_results_{gpu_type}.csv")
    file_exists = os.path.exists(log_file)

    with open(log_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Parallelism", "GPUType", "NumGPUs", "Epochs", "TotalTime(s)", "Speedup", "Efficiency"])
        writer.writerow([
            run_type,
            gpu_type,
            num_gpus,
            epochs,
            round(total_time, 2),
            round(speedup, 2) if speedup else "",
            round(efficiency, 2) if efficiency else ""
        ])

# Train Function
def train(rank, world_size):
    setup(rank, world_size)

    X_path = "/home/loganathan.k/leukemia/Team7_dataset/train/processed/X_train.npy"
    y_path = "/home/loganathan.k/leukemia/Team7_dataset/train/processed/y_train.npy"

    dataset = LeukemiaDataset(X_path, y_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(rank)
    model = FSDP(model, device_id=rank, use_orig_params=True, forward_prefetch=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    num_epochs = 5
    total_training_time = 0.0

    # Logs
    epoch_losses = []
    epoch_accuracies = []
    epoch_times = []
    gpu_utils_per_epoch = []

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for images, labels in loader:
            images, labels = images.to(rank), labels.to(rank)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_time = time.time() - start_time
        total_training_time += epoch_time

        epoch_accuracy = correct / total
        epoch_losses.append(total_loss)
        epoch_accuracies.append(epoch_accuracy)
        epoch_times.append(epoch_time)

        print(f"[GPU {rank}] Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f} | Acc: {epoch_accuracy:.4f} | Time: {epoch_time:.2f}s")

        if rank == 0:
            gpu_stats = get_gpu_utilization()
            gpu_utils_per_epoch.append(gpu_stats)

    print(f"[GPU {rank}] Total training time: {total_training_time:.2f} seconds")

    if rank == 0:
        try:
            baseline = float(os.environ.get("BASELINE_1GPU", "0"))
            if baseline > 0:
                speedup = baseline / total_training_time
                efficiency = speedup / world_size
                print(f"[GPU {rank}] Speedup: {speedup:.2f}")
                print(f"[GPU {rank}] Efficiency: {efficiency:.2f}")
            else:
                speedup = efficiency = None
                print("[GPU 0] No valid baseline set.")
        except:
            speedup = efficiency = None
            print("[GPU 0] Could not calculate speedup or efficiency.")

        gpu_type = os.environ.get("GPU_TYPE", "unknown")
        log_results("FSDP+AMP", gpu_type, world_size, num_epochs, total_training_time, speedup, efficiency)

        print("Before Save Model")
        # Save model
        model_path = f"models/{gpu_type}/model_fsdp_amp_{gpu_type}_{world_size}gpu.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.module.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Save training metrics
        print("Before train metrics")
        metrics_path = f"training_metrics/{gpu_type}_metrics_fsdp_amp_{world_size}gpu.npz"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        np.savez(metrics_path,
                 losses=np.array(epoch_losses),
                 accuracies=np.array(epoch_accuracies),
                 epoch_times=np.array(epoch_times),
                 gpu_utils=np.array(gpu_utils_per_epoch))
        print(f"Training metrics saved to {metrics_path}")

    cleanup()

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train(rank, world_size)
