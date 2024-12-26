import torch


class CrossEntropy(object):
    def __init__(self, weight):
        self.weight = weight
        self.loss_fn - torch.nn.CrossEntropyLoss(ignore_index=0)

    def __call__(self, predicted, labels):
        loss = self.loss_fn(predicted.squeeze(), labels.squeeze())
        loss = loss * self.weight
        return loss
    
    def get_name(self):
        return f"Cross Entropy (lambda = {self.weight})"
        