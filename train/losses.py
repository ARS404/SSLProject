import torch
import torch.nn.functional as F


class CrossEntropy(object):
    def __init__(self, weight):
        self.weight = weight
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def __call__(self, logits, labels):
        loss = self.loss_fn(logits.squeeze(), labels.squeeze())
        loss = loss * self.weight
        return loss
    
    def get_name(self):
        return f"Cross Entropy (lambda = {self.weight})"


class DiceLoss(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, logits, labels):
        batch_size, channels = logits.shape[:2]
        predict = logits.view(batch_size, channels, -1)
        target = labels.view(batch_size, 1, -1)

        predict = F.softmax(predict, dim=1)

        target_onehot = torch.zeros(predict.size()).cuda()
        target_onehot.scatter_(1, target, 1)

        intersection = torch.sum(predict * target_onehot, dim=2)  
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  
        
        dice_coef = (2 * intersection + 1e-5) / (union + 1e-5)  
        dice_loss = 1 - torch.mean(dice_coef)
        return dice_loss * self.weight

    def get_name(self):
        return f"Dice Loss (lambda = {self.weight})"