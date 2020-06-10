# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.optim

from . import FairseqOptimizer, register_optimizer


@register_optimizer('adai')
class FairseqAdai(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = Adai(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--adai-betas', default='(1e-4, 0.99)', metavar='B',
                            help='betas for Adam optimizer')
        parser.add_argument('--adai-eps', type=float, default=1e-4, metavar='D',
                            help='epsilon for Adam optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'betas': eval(self.args.adai_betas),
            'eps': self.args.adai_eps,
            'weight_decay': self.args.weight_decay,
        }

    @property
    def supports_flat_params(self):
        return True


class Adai(torch.optim.Optimizer):
    r"""Implements Adai algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): beta0 and beta2 (default: (1e4, 0.99))
        eps (float, optional): the smooth term added to the beta1 tensor (default: 1e-03)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-3, betas=(1e04, 0.99), eps=1e-03,
                 weight_decay=0,  nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0]:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, nesterov=nesterov)
        super(Adai, self).__init__(params, defaults)
    

    def __setstate__(self, state):
        super(Adai, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
        
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['beta1_prod'] = torch.ones_like(p.data, memory_format=torch.preserve_format)
                    #state['beta1'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1_prod = state['beta1_prod']
                
                beta0, beta2 = group['betas']

                state['step'] += 1
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)
                    
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                beta1 = (1. - exp_avg_sq_hat.mul(beta0)).clamp(0., 1 - group['eps'])
                
                beta1_prod.mul_(beta1)
                bias_correction1 = 1 - beta1_prod

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).addcmul_(1 - beta1, grad)
                exp_avg_hat = exp_avg / bias_correction1

                step_size = group['lr']

                p.data.add_(-step_size, exp_avg_hat)
        return loss

