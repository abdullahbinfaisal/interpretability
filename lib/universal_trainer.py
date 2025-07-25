import time
import torch
from overcomplete.sae.trackers import DeadCodeTracker
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Assume I have a batch inside a dataloader corresponding to: image_model1, activations_model1 etc

def train_usae(names, models, dataloader, criterion, optimizers, schedulers=None,
              nb_epochs=20, clip_grad=1.0, monitoring=1, device="cpu"):
    """
    Train a Sparse Autoencoder (SAE) model.

    Parameters
    ----------
    names: List of Model Names [string]
    models : Dict of Models (Sparse Autoencoders { nn.Module })
        The SAE model to train.
    dataloader : DataLoader
        DataLoader providing the training data.
    criterion : callable
        Loss function.
    optimizers : Dict of { optim.Optimizer }
        Optimizer for updating model parameters.
    scheduler : callable, optional
        Learning rate scheduler. If None, no scheduler is used, by default None.
    nb_epochs : int, optional
        Number of training epochs, by default 20.
    clip_grad : float, optional
        Gradient clipping value, by default 1.0.
    monitoring : int, optional
        Whether to monitor and log training statistics, the options are:
         (0) silent.
         (1) monitor and log training losses.
         (2) monitor and log training losses and statistics about gradients norms and z statistics.
        By default 1.
    device : str, optional
        Device to run the training on, by default 'cpu'.

    Returns
    -------
    defaultdict
        Logs of training statistics.
    """
    total_loss_per_epoch = []  # Total loss over all models
    individual_loss_per_epoch = defaultdict(list)  # Per-model losses per epoch

    for epoch in range(nb_epochs):

        start_time = time.time()
        epoch_loss = 0.0
        epoch_model_losses = {name: 0.0 for name in names}  # Reset per-model epoch loss
        dead_tracker = None
        rotator = 0

        for name in names:
            models[name].to(device)

        # tqdm dataloader
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{nb_epochs}")
        
        for batch_count, batch in progress_bar:
            # Reset Everything
            total_loss = 0.0
            for name in names:
                optimizers[name].zero_grad()
                models[name].eval()

            # Current SAE
            current = names[rotator]

            # Current SAE model
            sae = models[current]
            sae.train()

            # Encoder Forward Pass
            z_pre, z = sae.encode(batch[f"activations_{names[rotator]}"].squeeze().to(device))
            #print("Z Shape: ", z.shape)

            # Decoder across all models & accumulate loss
            for n, m in models.items():
                
                if n == current:
                    x_hat = m.decode(z)
                else:
                    with torch.no_grad():
                        x_hat = m.decode(z.detach())
                
                target = batch[f"activations_{n}"].squeeze().to(device)
                loss = criterion(x_hat, target)

                epoch_model_losses[n] += loss.item()
                total_loss += loss

            # Backward + Optimize
            total_loss.backward()
            optimizers[current].step()
            if schedulers:
                schedulers[current].step()

            # Rotator Update
            rotator += 1
            rotator = rotator % len(names)

            # Dead feature tracking
            if dead_tracker is None:
                dead_tracker = DeadCodeTracker(z.shape[1], device)
            dead_tracker.update(z)

            # Update metrics and tqdm bar
            batch_loss = total_loss.item()
            epoch_loss += batch_loss
            progress_bar.set_postfix(loss=batch_loss)
            
        # Epoch summary
        #print(dead_tracker)
        dead_features = dead_tracker.get_dead_ratio()
        print(f"\n[Epoch {epoch+1}] Loss: {epoch_loss:.4f} | Time: {time.time() - start_time:.2f}s | Dead Features: {dead_features*100:.1f}%")
            

        total_loss_per_epoch.append(epoch_loss)
        for name in names:
            individual_loss_per_epoch[name].append(epoch_model_losses[name])
            print(f"{name} Loss: {individual_loss_per_epoch[name]}")

    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_per_epoch, label="Total Loss", linewidth=2)

    for name in names:
        plt.plot(individual_loss_per_epoch[name], label=f"{name} Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    return 
