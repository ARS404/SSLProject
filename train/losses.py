import evaluate


class mean_iou(object):
    def __init__(self, weight):
        self.loss = evaluate.load("mean_iou")
        self.weight = weight

    def __call__(self, pred_labels, true_labels):
        return self.weight * self.loss(pred_labels, true_labels)
    
    def get_name(self):
        pass
        