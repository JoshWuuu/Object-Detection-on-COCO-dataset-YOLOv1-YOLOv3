import config
import torch

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    """
    Train the model on the training set.
    
    Inputs:
    - train_loader: obj, torch.utils.data.DataLoader
    - model: obj, torch.nn.Module
    - optimizer: obj, torch.optim
    - loss_fn: obj, loss function
    - scaler: obj, torch.cuda.amp.GradScaler
    - scaled_anchors: tensor, shape (3, 3, 2)

    """
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )   

        with torch.cuda.amp.autocast():
            # output shape (batch_size, 3, grid_size, grid_size, classes + 5)
            out = model(x)
            # calculate loss for each scale prediction and anchor box
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)
