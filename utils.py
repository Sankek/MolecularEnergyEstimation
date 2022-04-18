import os
import os.path as osp
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import torch
from kaggle.api.kaggle_api_extended import KaggleApi


def save_config(config, path):
    with open(path, 'w') as f:
        json.dump(config, f)


def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_model(model_state, model_config, epoch):
    model = model_state['model']
    optimizer = model_state['optimizer']
    train_losses = model_state['train_losses']
    val_losses = model_state['val_losses'] if 'val_losses' in model_state else None
    output_call = model_state['output_call']
    name = model_config['model_name']
    path = model_config['save_path']

    if not osp.exists(path):
        os.mkdir(path)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'output_call': output_call
        }
    torch.save(state, osp.join(path, name) + f'_{epoch}ep.pth')
    save_config(model_config, osp.join(path, name) + f'_config.json')


def load_model(model_state, model_config, epoch):
    name = model_config['model_name']
    path = model_config['save_path']
    device = model_config['device']

    state = torch.load(osp.join(path, name) + f'_{epoch}ep.pth', map_location=device)
    model_state['model'].load_state_dict(state['model'])
    model_state['optimizer'].load_state_dict(state['optimizer'])
    model_state['train_losses'] = state['train_losses']
    if 'val_losses' in model_state:
        model_state['val_losses'] = state['val_losses']
    model_state['output_call'] = state['output_call']
    model_state['trained_epochs'] = epoch


def load_losses(path, name, epoch, device='cpu'):
    state = torch.load(osp.join(path, name) + f'_{epoch}ep.pth', map_location=device)
    train_losses = state['train_losses']
    if 'val_losses' in state:
        val_losses = state['val_losses']
        return train_losses, val_losses

    return train_losses


def predict_test(model_state, model_config, dataloader):
    model = model_state['model']
    output_call = model_state['output_call']
    test_loader = dataloader['test']
    device = model_config['device']

    out_list = []
    for data in tqdm(test_loader):  
        with torch.no_grad():
            data = data.to(device)
            out = output_call(data)
            out_list.append(out.cpu().numpy().reshape(-1))

    energies = np.hstack(out_list)

    return energies


def make_submission(y_pred, competition_name, description='API Submission'):
    submission_data = {
        'id': np.arange(1, len(y_pred)+1),
        'energy': y_pred
    }

    pd.DataFrame(submission_data).to_csv('submission.csv', index=False)

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

    kaggle_api.competition_submit('submission.csv', description, competition_name)


def change_lr(model_state, lr):
    optimizer = model_state['optimizer']
    for g in optimizer.param_groups:
        g['lr'] = lr

    model_state['lr'] = lr


def plot_loss(model_state, start_from=1):
    train_losses = model_state['train_losses']
    val_losses = model_state['val_losses'] if 'val_losses' in model_state else None

    fig, ax = plt.subplots()
    epochs = len(train_losses)
    ax.plot(np.arange(start_from, epochs+1), train_losses[start_from-1:], label='train')
    if val_losses:
        ax.plot(np.arange(start_from, epochs+1), val_losses[start_from-1:], label='val')
    ax.legend()
    ax.grid()
    plt.show()
