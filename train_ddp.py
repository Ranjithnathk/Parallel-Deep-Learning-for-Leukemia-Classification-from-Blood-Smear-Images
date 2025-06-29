import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchvision import models
from torch.optim import Adam
import time
import numpy as np
import csv
import subprocess  # for GPU utilization logging

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

# DDP Setup
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Logging results to CSV
def log_results(run_type, num_gpus, epochs, total_time, gpu_type, speedup=None, efficiency=None):
    os.makedirs("csv_files", exist_ok=True) 
    log_file = os.path.join("csv_files", f"ddp_gpu_results_{gpu_type}.csv")
    file_exists = os.path.exists(log_file)

    with open(log_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Parallelism", "GPUType", "NumGPUs", "Epochs", 
                "TotalTime(s)", "Speedup", "Efficiency"
            ])
        writer.writerow([
            run_type,
            gpu_type,
            num_gpus,
            epochs,
            round(total_time, 2),
            round(speedup, 2) if speedup else "",
            round(efficiency, 2) if efficiency else ""
        ])

# Optional GPU Util Logging (only rank 0)
def get_gpu_utilization():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,nounits,noheader"]
        ).decode("utf-8").strip().split("\n")
        utils = [list(map(int, line.split(", "))) for line in output]
        return utils  # List of [gpu_util, mem_used, mem_total] per GPU
    except:
        return []

# Train function
def train(rank, world_size):
    setup(rank, world_size)

    # Load data
    X_path = "/home/loganathan.k/leukemia/Team7_dataset/train/processed/X_train.npy"
    y_path = "/home/loganathan.k/leukemia/Team7_dataset/train/processed/y_train.npy"

    dataset = LeukemiaDataset(X_path, y_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

    # Model setup
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    total_training_time = 0.0
    num_epochs = 5

    # Logging containers
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
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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

        # Log GPU util (only rank 0)
        if rank == 0:
            gpu_stats = get_gpu_utilization()
            gpu_utils_per_epoch.append(gpu_stats)

    print(f"[GPU {rank}] Total training time: {total_training_time:.2f} seconds")

    if rank == 0:
        try:
            baseline_1gpu_time = float(os.environ.get("BASELINE_1GPU", "0"))
            if baseline_1gpu_time > 0:
                speedup = baseline_1gpu_time / total_training_time
                efficiency = speedup / world_size
                print(f"[GPU {rank}] Speedup: {speedup:.2f}")
                print(f"[GPU {rank}] Efficiency: {efficiency:.2f}")
            else:
                print("[GPU 0] No valid baseline set, skipping speedup/efficiency.")
                speedup = efficiency = None
        except:
            print("[GPU 0] Error calculating speedup.")
            speedup = efficiency = None

        gpu_type = os.environ.get("GPU_TYPE", "unknown")
        log_results("DDP", world_size, num_epochs, total_training_time, gpu_type, speedup, efficiency)

        # Save model
        model_path = f"models/{gpu_type}/model_ddp_{gpu_type}_{world_size}gpu.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.module.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Save training curves
        metrics_path = f"training_metrics/{gpu_type}_metrics_{world_size}gpu.npz"
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        np.savez(metrics_path,
                 losses=np.array(epoch_losses),
                 accuracies=np.array(epoch_accuracies),
                 epoch_times=np.array(epoch_times),
                 gpu_utils=np.array(gpu_utils_per_epoch))
        print(f"Training metrics saved to {metrics_path}")

    cleanup()

# Entry point
if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train(rank, world_size)