import torch.optim as optim
import numpy as np
import torch
from lifelines.utils import concordance_index
def c_index(risk_pred, y, e):
    ''' Performs calculating c-index

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)

def adjust_learning_rate(optimizer, epoch, lr, lr_decay_rate):
    ''' Adjusts learning rate according to (epoch, lr and lr_decay_rate)

    :param optimizer: (torch.optim object)
    :param epoch: (int)
    :param lr: (float) the initial learning rate
    :param lr_decay_rate: (float) learning rate decay rate
    :return lr_: (float) updated learning rate
    '''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1+epoch*lr_decay_rate)
    return optimizer.param_groups[0]['lr']

def train(train_loader, test_loader, model, loss_func, config):
    optimizer = eval('optim.{}'.format(config['optimizer']))(model.parameters(), lr=config['learning_rate'])

    patience = 50
    best_c_index,flag = 0,0
    for epoch in range(1, config['epochs']+1):
        # adjusts learning rate
        lr = adjust_learning_rate(optimizer, epoch,
                                  config['learning_rate'],
                                  config['lr_decay_rate'])
        # train step
        model.train()
        for X, y, e in train_loader:
            # makes predictions
            pred = model(X)
            train_loss = loss_func(pred, y, e, model)
            train_c = c_index(-pred, y, e)
            # updates parameters
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # valid step
        model.eval()
        for X, y, e in test_loader:
            # makes predictions
            with torch.no_grad():
                pred = model(X)
                valid_loss = loss_func(pred, y, e, model)
                valid_c = c_index(-pred, y, e)
                if best_c_index < valid_c:
                    best_c_index = valid_c
                    flag = 0
                    # saves the best model
                    torch.save(model.state_dict(), config["model_path"])
                else:
                    flag += 1
                    if flag >= patience:
                        return best_c_index
        # notes that, train loader and valid loader both have one batch!!!
        print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tc-index: {:.8f}({:.8f})\tlr: {:g}'.format(
            epoch, train_loss.item(), valid_loss.item(), train_c, valid_c, lr), end='', flush=False)
    return best_c_index