import torch.nn as nn 
import torch


# import torch.nn.functional as F
# from torch.autograd import Variable


cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss(**kwargs)
        

    def forward(self, inputs, targets):
        
        BCE_loss = self.cross_entropy(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        return F_loss


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox