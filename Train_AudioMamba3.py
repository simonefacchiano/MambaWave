import sys
sys.path.append('/home/simone')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import librosa
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


from VideoMamba.DiffWave_simone3 import DiffWave, forward_diffusion_sample, interpolate_audio_signal

import torch
import torch.nn.functional as F

import soundfile as sf

from torch.cuda.amp import autocast, GradScaler # this was used for mixed-precision, to overcome problems of GPU memory

def collate_fn(batch):
    input_audio, target_audio = zip(*batch)

    # Convert tuples into lists of tensors
    input_audio = [torch.tensor(x, dtype=torch.float32) for x in input_audio]
    target_audio = [torch.tensor(x, dtype=torch.float32) for x in target_audio]

    # We find the max length of each set of input/target
    max_len_input = max(x.size(0) for x in input_audio)
    max_len_target = max(x.size(0) for x in target_audio)

    # PADDING
    input_audio_padded = [F.pad(x, (0, max_len_input - x.size(0)), "constant", 0) for x in input_audio]
    target_audio_padded = [F.pad(x, (0, max_len_target - x.size(0)), "constant", 0) for x in target_audio]

    # Stack the tensors
    input_audio_padded = torch.stack(input_audio_padded)
    target_audio_padded = torch.stack(target_audio_padded)

    return input_audio_padded, target_audio_padded



class AudioDataset(Dataset):
    """
    This is a custom class developed for the training of MambaWave.
    It also contains an optional parameter *normalize* to perform
    normalization of both input and output audios in the range [-1, +1] 
    """
    def __init__(self, input_dir, target_dir, normalize=True):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.filenames = os.listdir(input_dir)
        self.normalize = normalize

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.filenames[idx])
        target_path = os.path.join(self.target_dir, self.filenames[idx])

        # Load input/ target
        input_audio, _ = librosa.load(input_path, sr=None)  # Assicurati che sr=None per mantenere la frequenza originale
        target_audio, _ = librosa.load(target_path, sr=None)

        # If requested, normalize
        if self.normalize:
            input_audio = self.normalize_audio(input_audio)
            target_audio = self.normalize_audio(target_audio)

        # Convert into torch.tensor
        input_audio = torch.tensor(input_audio, dtype=torch.float32)
        target_audio = torch.tensor(target_audio, dtype=torch.float32)

        return input_audio, target_audio

    def normalize_audio(self, audio):
        """
        Custom function to normalize audio signals in the range [-1, +1]
        """
        return 2 * (audio - audio.min()) / (audio.max() - audio.min()) - 1
    


def create_dataloader(input_dir, target_dir, batch_size=4, shuffle=True, num_workers=0):
    dataset = AudioDataset(input_dir, target_dir, normalize=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader

def create_mask(tensor, pad_value=0):
    return tensor != pad_value

class CombinedMSE_MAELoss(nn.Module):
    """
    Custom loss that combines MSE with L1 losses.
    It didn't work. We leave it here for future reference
    """
    def __init__(self, alpha=0.5):
        super(CombinedMSE_MAELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.alpha = alpha  # weight

    def forward(self, outputs, targets):
        mse_loss = self.mse(outputs, targets)
        mae_loss = self.mae(outputs, targets)
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * mae_loss
        return combined_loss
    
def log_mse_loss(pred, target, epsilon=1e-8):
    mse_loss = torch.mean((pred - target) ** 2)
    return torch.log(mse_loss + epsilon)    


###### Train ######

def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Function to load the checkpoint of a given model
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    # Load train/eval loss
    train_loss = checkpoint.get('train_loss', None)
    val_loss = checkpoint.get('val_loss', None)

    if train_loss is None:
        print("Warning: Train Loss not found in checkpoint.")
    if val_loss is None:
        print("Warning: Validation Loss not found in checkpoint.")
    
    return model, optimizer, epoch, train_loss, val_loss



def validate(model, dataloader, criterion, device):
    """
    Function to perform the evaluation of the model
    """
    model.eval()
    val_loss = 0.0

    # tqdm to show progress during evaluation phase
    with torch.no_grad():
        for batch_idx, (cond, target_audio) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Validating", unit="batch"):
            cond = cond.to(device)  # This is the 24 kHz audio conditioning
            target_audio = target_audio.to(device)  # This is the 48 kHz input audio

            batch_size = target_audio.size(0)
            audio_length = target_audio.size(1)

            diffusion_step = torch.randint(0, len(model.params.noise_schedule), (batch_size,), dtype=torch.long).to(device)

            # Forward diffusion according to the papers of Diffusion Models
            noisy_target_audio, true_noise = forward_diffusion_sample(target_audio, diffusion_step, model.params.noise_schedule.to(device))

            # Forward pass of the model
            predicted_noise = model(noisy_target_audio, diffusion_step, cond)

            # Match dimensions (not really important)
            min_length = min(predicted_noise.size(-1), true_noise.size(-1))
            predicted_noise = predicted_noise[..., :min_length]
            true_noise = true_noise[..., :min_length]

            # Compute the loss
            loss = criterion(predicted_noise, true_noise)
            val_loss += loss.item()

    # Average loss on the validation set
    val_loss /= len(dataloader)
    print(f'Validation Loss: {val_loss:.4f}')
    return val_loss
    

def train(model, train_dataloader, val_dataloader, optimizer, criterion, epoch, device, checkpoint_dir):
    """
    Function to train the model
    """

    model.train()
    train_loss = 0.0
    scaler = GradScaler()

    for batch_idx, (cond, target_audio) in enumerate(train_dataloader):

        # Move on the correct device
        cond = cond.to(device)  # 24 kHz conditioning
        target_audio = target_audio.to(device)  # 48 kHz input

        # Compute the size
        batch_size = target_audio.size(0)
        audio_length = target_audio.size(1)

        # Sample a diffusion step
        if len(model.params.noise_schedule) > 0:
            diffusion_step = torch.randint(0, len(model.params.noise_schedule), (batch_size,), dtype=torch.long).to(device)
        else:
            raise ValueError("Noise schedule is empty or not properly initialized.")

        # Diffusion phase (adding noise the input)
        noisy_target_audio, true_noise = forward_diffusion_sample(target_audio, diffusion_step, model.params.noise_schedule.to(device))

        # Reset grad
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast():
            predicted_noise = model(noisy_target_audio, diffusion_step, cond)

            # Check the shapes
            if predicted_noise.shape != true_noise.shape:
                print("Mismatch in noise prediction and true noise dimensions:", predicted_noise.shape, true_noise.shape)
                continue  # Skip this batch or handle appropriately

            # Compute loss
            loss = criterion(predicted_noise, true_noise)

        # Backward pass with mixed precision
        scaler.scale(loss).backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights with the scaler
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        # Print the loss every 10 batches
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Training Loss: {loss.item()}')

        # Save checkpoints every 1000 batch
        if batch_idx % 1000 == 0:
            checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}_batch_{batch_idx}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"** Checkpoint saved: {checkpoint_path}")

    # Average training loss
    train_loss /= len(train_dataloader)
    print(f'Epoch {epoch}, Average Training Loss: {train_loss:.4f}')

    # Save the checkpoint at the end of the epoch
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, checkpoint_path)
    print(f"** End of Epoch {epoch} Checkpoint saved: {checkpoint_path}")

            


class Params:
    def __init__(self, residual_channels, noise_schedule_params, unconditional, n_mels, residual_layers, dilation_cycle_length, device):
        self.residual_channels = residual_channels
        self.unconditional = unconditional
        self.n_mels = n_mels
        self.residual_layers = residual_layers
        self.dilation_cycle_length = dilation_cycle_length

        # Noise scheduling
        start, stop, num_points = noise_schedule_params
        self.noise_schedule = torch.linspace(start, stop, num_points).float().to(device)


def main():
    # Hyperparameters
    epochs = 5
    batch_size = 2
    learning_rate = 0.0004
    step_size = 1
    gamma = 0.9
    checkpoint_path = None # Place here the last checkpoint path you wish to use

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize
    params = Params(
        residual_channels=32,
        noise_schedule_params=(1e-6, 0.005, 500), # here: https://dzdata.medium.com/intro-to-diffusion-model-part-4-62bd94bd93fd
        unconditional=False,
        n_mels=10,
        residual_layers=11,
        dilation_cycle_length=5000, # dilation of the dilated convolution
        device=device
    )
    
    # Create model + optimizer
    model = DiffWave(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Other optimizers I tried:
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) <-- quella usata finora
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Create scheduler + define the loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = torch.nn.MSELoss() 
    
    # Other loss functions I tried:
    # torch.nn.L1Loss()
    # log_mse_loss --> custom loss function

    # Restart from the given checkpoint path, if present
    start_epoch = 1
    train_loss, val_loss = None, None
    if checkpoint_path:
        model, optimizer, start_epoch, train_loss, val_loss = load_checkpoint(checkpoint_path, model, optimizer)
        print(f"Checkpoint loaded. Starting from epoch {start_epoch+1} with train loss {train_loss} and validation loss {val_loss}")

    # Data + path for saving the checkpoints
    train_input_dir = '/media/nvme_4tb/simone_data/VoiceBank/clean_trainset_28spk_wav_24khz'
    train_target_dir = '/media/nvme_4tb/simone_data/VoiceBank/clean_trainset_28spk_wav'
    val_input_dir = '/media/nvme_4tb/simone_data/VoiceBank/clean_trainset_28spk_wav_24khz_validation'
    val_target_dir = '/media/nvme_4tb/simone_data/VoiceBank/clean_trainset_28spk_wav_validation'
    checkpoints_dir = '/media/nvme_4tb/simone_data/VoiceBank/checkpoints_NEW2'

    # Dataloader:
    train_dataloader = create_dataloader(train_input_dir, train_target_dir, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = create_dataloader(val_input_dir, val_target_dir, batch_size=batch_size, shuffle=False, num_workers=0)

    # Start training phase
    for epoch in range(start_epoch, epochs + 1):
        train(model, train_dataloader, val_dataloader, optimizer, criterion, epoch, device, checkpoints_dir)
        scheduler.step()  # Aggiorna il learning rate secondo lo scheduler
        print(f"Epoch {epoch}/{epochs} completed. Learning Rate: {scheduler.get_last_lr()[0]}")
        
        # Start validation phase
        val_loss = validate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch}/{epochs} completed. Validation Loss: {val_loss:.4f}")

if __name__ == "__main__":
    main()