from torchmetrics import JaccardIndex

class MIOU(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.jaccard = JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=0).cuda()

    def __call__(self, logits, labels):
        return self.jaccard(logits, labels.squeeze()).cpu()
    
    def get_name(self):
        return "MIOU"
        
