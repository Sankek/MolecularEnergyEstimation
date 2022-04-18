import os
import os.path as osp
import re
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

from utils import load_losses

def get_models_max_epoch(models_path):
    model_groups = os.listdir(models_path)
    model_max_epoch_dict = defaultdict(int)
    for model_group in model_groups:
        path = osp.join(models_path, model_group)
        files = os.listdir(path)
        files = [f for f in files if f.endswith('.pth')]

        for model_name, epoch in re.findall('(.*?)_(\d+)ep.pth', ''.join(files)):
            epoch = int(epoch)
            if epoch > model_max_epoch_dict[model_name]:
                model_max_epoch_dict[model_name] = epoch

    return model_max_epoch_dict


def write_tensorboard_losses(models_path, device='cpu'):
    writer = SummaryWriter('runs/compare_models')

    model_max_epoch_dict = get_models_max_epoch(models_path)
    model_groups = os.listdir(models_path)
    for model_group in model_groups:
        for model_name, epoch in model_max_epoch_dict.items():
            if not model_name.startswith(model_group):
                continue
            path = osp.join(models_path, model_group)
        
            train_losses, val_losses = load_losses(path, model_name, epoch, device=device)

            print(f'loaded {model_name}')
            for i in range(1, len(train_losses)+1):
                writer.add_scalars('train', {f'{model_name}':train_losses[i-1]}, i)
                if val_losses is not None:
                    writer.add_scalars('val', {f'{model_name}':val_losses[i-1]}, i)

    writer.flush()
    writer.close()