from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


class WarmupLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps**0.5
            * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]


# https://github.com/seominseok0429/pytorch-warmup-cosine-lr/blob/master/warmup_scheduler/scheduler.py
class BaseGradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warmup_epoch):
        self.warmup_epoch = warmup_epoch
        self.multiplier = multiplier
        self.after_scheduler = None
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.) for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_epoch)
        else:
            return super(BaseGradualWarmupScheduler, self).step(epoch)


class WarmupCosineAnnealingLR(BaseGradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, warmup_epoch, total_epoch, eta_min, last_epoch=-1):
        super().__init__(optimizer, multiplier, warmup_epoch)
        self.after_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epoch - 1,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.01)
    scheduler = WarmupCosineAnnealingLR(optim, multiplier=8, warmup_epoch=5, total_epoch=100, eta_min=0, last_epoch=-1)
    a = []
    b = []
    for epoch in range(1, 100):
        scheduler.step(epoch)
        a.append(epoch)
        b.append(optim.param_groups[0]['lr'])
        print(epoch, optim.param_groups[0]['lr'])

    plt.plot(a, b)
    plt.show()
