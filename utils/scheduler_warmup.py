from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """
    from https://github.com/ildoonet/pytorch-gradual-warmup-lr

    Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_step: target learning rate is reached at total_step, gradually
        after_scheduler: after target_step, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_step, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_step = total_step
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_step = 0
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_step > self.total_step:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_step) / self.total_step) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_step / self.total_step + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, step=None):
        if step is None:
            step = self.last_step+1
        self.last_step = step if step != 0 else 1  # ReduceLROnPlateau is called at the end of step, whereas others are called at beginning
        if self.last_step <= self.total_step:
            if self.multiplier == 1.0:
                warmup_lr = [base_lr * (float(self.last_step) / self.total_step) for base_lr in self.base_lrs]
            else:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_step / self.total_step + 1.) for base_lr in
                        self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if step is None:
                self.after_scheduler.step(metrics)
            else:
                self.after_scheduler.step(metrics)

    def step(self, step=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if step is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(step - self.total_step)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(step)
        else:
            self.step_ReduceLROnPlateau(metrics, step)
