import torch

from utils import change_lr, save_model, plot_loss

from tqdm.notebook import tqdm

def train(model_state, model_config, dataloader):
    model = model_state['model']
    output_call = model_state['output_call']
    optimizer = model_state['optimizer']
    criterion = model_state['criterion']
    train_loader = dataloader['train']
    device = model_config['device']

    model.train()

    epoch_loss = 0
    for data in tqdm(train_loader):  
        data = data.to(device)
        out = output_call(data) 
        loss = criterion(out, data.y.reshape(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        epoch_loss += loss.item()  

    return epoch_loss / len(train_loader)

def validate(model_state, model_config, dataloader):
    model = model_state['model']
    output_call = model_state['output_call']
    criterion = model_state['criterion']
    val_loader = dataloader['val']
    device = model_config['device']

    model.eval()

    loss = 0
    with torch.no_grad():
        for data in tqdm(val_loader):
            data = data.to(device)
            out = output_call(data) 
            loss += criterion(out, data.y.reshape(-1, 1)).item()
        return loss / len(val_loader) 

def train_loop(model_state, model_config, dataloader, epochs):
    for epoch in range(1, epochs+1):
        train_loss = train(model_state, model_config, dataloader)
        model_state['train_losses'].append(train_loss)
        info = f'*Epoch {epoch}* Train Loss: {train_loss:.4f}'
        if 'val' in dataloader:
            val_loss = validate(model_state, model_config, dataloader)
            model_state['val_losses'].append(val_loss)
            info += f', Validation Loss: {val_loss:.4f}'
        print(info)

def long_train(model_state, model_config, dataloader, epochs=1, new_lr=None):
    start_epoch = model_state['trained_epochs']
    if new_lr is not None:
        change_lr(model_state, new_lr)
    lr = model_state['lr']
    print(f"---------- TRAINING {start_epoch}:{start_epoch+epochs}, LR = {lr} ----------")
    train_loop(model_state, model_config, dataloader, epochs)
    epoch = start_epoch + epochs
    model_state['trained_epochs'] = epoch

    if model_config['save']:
        save_model(model_state, model_config, epoch)
    plot_loss(model_state, start_from=max(epoch-epochs, 1))
    
    model_state['trained_epochs'] = epoch