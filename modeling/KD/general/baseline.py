import torch
import torch.nn as nn
import torch.nn.functional as F


class Scratch(nn.Module):
    def __init__(self, args, backbone):
        super().__init__()

        self.args = args
        self.backbone = backbone
        self.training = True

    def get_ratings(self, param):
        if self.args.task == "ctr":
            return self.backbone(param)
        else:
            return self.backbone.get_ratings(param)

    def do_something_in_each_epoch(self, epoch):
        return
    
    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def get_params_to_update(self):
        return [{"params": self.backbone.parameters(), 'lr': self.args.lr, 'weight_decay': self.args.wd}]

    def forward(self, *params):
        if self.args.task == "ctr":
            data, labels = params
            output = self.backbone(data)
            base_loss = self.backbone.get_loss(output, labels)
        else:
            output = self.backbone(*params)
            base_loss = self.backbone.get_loss(output)
        loss = base_loss
        return loss, base_loss.detach(), torch.tensor(0.)

    @property
    def param_to_save(self):
        return self.backbone.state_dict()
    
    @property
    def score_mat_to_save(self):
        with torch.no_grad():
            training = self.backbone.training
            self.backbone.eval()
            score_mat = self.backbone.get_all_ratings().cpu()
            self.backbone.train(training)
            return score_mat
