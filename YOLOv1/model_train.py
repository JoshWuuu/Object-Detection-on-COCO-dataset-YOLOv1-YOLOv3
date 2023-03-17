from tqdm import tqdm
from metrics import *
import config

def train_fn(train_loader, model, optimizer, loss_fn):
    """
    train function

    Inputs:
    - train_loader: DataLoader for training dataset
    - model: model to train
    - optimizer: optimizer to use
    - loss_fn: loss function to use

    Returns:
    - mean_loss: float, mean loss of the epoch
    """
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update progress bar
        loop.set_postfix(loss=loss.item())
    
    mean_loss = sum(mean_loss) / len(mean_loss)
    print(f"Mean loss was {mean_loss}.")

    return mean_loss

