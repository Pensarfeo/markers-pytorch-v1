from torch.optim import lr_scheduler
import numpy as np

def format_param(name, optimizer, param):
    """Return correctly formatted lr/momentum for each param group."""
    if isinstance(param, (list, tuple)):
        if len(param) != len(optimizer.param_groups):
            raise ValueError("expected {} values for {}, got {}".format(len(optimizer.param_groups), name, len(param)))
        return param
    else:
        return [param] * len(optimizer.param_groups)

class StepLR(lr_scheduler.LRScheduler):
    def __init__(self,
                 optimizer,
                 lrs,
                 momentums,
                 iters, 
                 last_epoch=-1,
                 verbose=False):
        
        self.lrs = lrs
        self.iters = []
        lastIter = 0
        for it in iters:
          lastIter = lastIter + it
          self.iters.append(lastIter)

        self.momentums = momentums
        print(self.lrs, self.iters, self.momentums)
        # Attach optimizer
        if not isinstance(optimizer, lr_scheduler.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        
        startMomentum = momentums[0]

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group['momentum'] = startMomentum

        super().__init__(optimizer, last_epoch, verbose)




    def scale_fn(self, x):
        return 1

#     def _triangular_scale_fn(self, x):
#         return 1.

#     def _triangular2_scale_fn(self, x):
#         return 1 / (2. ** (x - 1))

#     def _exp_range_scale_fn(self, x):
#         return self.gamma**(x)

    def get_lr_momentum(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        lr = np.interp(self.last_epoch, self.iters, self.lrs)
        lrs = [lr]*3
        momentum = np.interp(self.last_epoch, self.iters, self.momentums)
        ms = [momentum]*3

        return lrs, ms
    
    def get_lr(self):
        lrs, ms = self.get_lr_momentum()

        for param_group, momentum in zip(self.paramGroups, ms):
          param_group['momentum'] = momentum

        return lrs



    def state_dict(self):
        state = super().state_dict()
        # We are dropping the `_scale_fn_ref` attribute because it is a `weakref.WeakMethod` and can't be pickled
        state.pop("_scale_fn_ref")
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False, warmup = None):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # # Attach optimizer
        # if not isinstance(optimizer, Optimizer):
        #     raise TypeError('{} is not an Optimizer'.format(
        #         type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

        self.paramGroups = self.optimizer.param_groups

        if warmup is not None:
            self.step = self.warmUp
            self.warmUpIters = warmup[2]

            self.warmUpLrMin = warmup[0][0]
            self.warmUpLrMax = warmup[0][1]
            self.warmUpLrStep = (warmup[0][1] - warmup[0][0])/self.warmUpIters

            self.warmUpMomentumMin = warmup[1][0]
            self.warmUpMomentumDelta = (warmup[1][1] - warmup[1][0]) 
            self.warmUpMomentumStep = self.warmUpMomentumDelta/self.warmUpIters
            self.warmUpMomentumMax = warmup[1][1]
            self.last_epoch = 0
        
        for param_group in self.paramGroups:
          param_group['lr'] = self.warmUpLrMin
          param_group['momentum'] = self.warmUpMomentumMin
    
    def warmUp(self, _):
      lr = self.warmUpLrMin + self.warmUpLrStep*self.last_epoch
      
      momentum = self.warmUpMomentumMin + self.warmUpMomentumStep*self.last_epoch

      self.last_epoch += 1
      if self.last_epoch == self.warmUpIters:
          self.step = self.stepDecay
      
      for param_group in self.paramGroups:
        param_group['lr'] = lr
        param_group['momentum'] = momentum

      return lr, momentum

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def stepDecay(self, metrics = 0):
      # convert `metrics` to float, in case it's a zero-dim Tensor
      current = float(metrics)

      if self.is_better(current, self.best):
          self.best = current
          self.num_bad_epochs = 0
      else:
          self.num_bad_epochs += 1

      if self.in_cooldown:
          self.cooldown_counter -= 1
          self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

      if self.num_bad_epochs > self.patience:
          self._reduce_lr(self.last_epoch + 1)
          self.cooldown_counter = self.cooldown
          self.num_bad_epochs = 0

      self._last_lr = [group['lr'] for group in self.paramGroups]
      self._last_m = [group['momentum'] for group in self.paramGroups]

      return float(self._last_lr[0]), float(self._last_m[0])

    def setParams(self, lr, momentum):
      for i, param_group in enumerate(self.paramGroups):
          param_group['lr'] = lr
          param_group['momentum'] = momentum
        
    def _reduce_lr(self, epoch):
      for i, param_group in enumerate(self.paramGroups):
        old_lr = float(param_group['lr'])
        new_lr = max(old_lr * self.factor, self.min_lrs[i])
        # print(new_lr, self.warmUpLrMax, self.warmUpMomentumDelta)
        reduceMomenturm = (new_lr/self.warmUpLrMax)*self.warmUpMomentumDelta
        # print(reduceMomenturm)
        new_m = self.warmUpMomentumMax - reduceMomenturm

        if old_lr - new_lr > self.eps:
          param_group['lr'] = new_lr
          param_group['momentum'] = new_m
          if self.verbose:
            epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
            print('Epoch {}: reducing learning rate of group {} to {:.4e}.'.format(epoch_str, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')
        inf = float('inf')
        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)