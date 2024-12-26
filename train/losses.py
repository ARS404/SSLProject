import torch


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
        