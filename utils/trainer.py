from sklearn.metrics import confusion_matrix, roc_auc_score
from operator import add

import torch

def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred.cpu().numpy()
    y_pred_auc = y_pred.reshape(-1)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    score_auc = roc_auc_score(y_true, y_pred_auc)
    score_f1 = 2*tp/(2*tp+fn+fp)
    score_acc = (tp + tn) / (tp + tn + fn + fp)

    return [specificity, sensitivity, score_auc, score_f1, score_acc]

def train(net, loader, opt, loss, device, cutout_prob=0.5, scheduler=None):
    net.train()
    epoch_loss = 0.0
    count = 0
    for x, y in loader:
        opt.zero_grad()

        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        preds = net.forward(x)
        loss_value = loss(preds, y)
        loss_value.backward()
        opt.step()
        epoch_loss += loss_value.item()

    if scheduler != None:
        scheduler.step()
    epoch_loss = epoch_loss/len(loader)
    
    return epoch_loss


def evaluate(net, loader, loss, device):
    net.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            preds = net.forward(x)
            loss_value = loss(preds, y)

            epoch_loss += loss_value.item()
            epoch_loss = epoch_loss/len(loader)

    return epoch_loss

def score(model, loader, device):
    model.eval()
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    for x, y in loader:
        with torch.no_grad():
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            preds = model(x)
            preds = torch.sigmoid(preds)

            score = calculate_metrics(y, preds)
            metrics_score = list(map(add, metrics_score, score))

    for i in range(len(metrics_score)):
        metrics_score[i] /= len(loader)

    return metrics_score