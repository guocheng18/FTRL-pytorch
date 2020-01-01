import torch
from torch.optim.optimizer import Optimizer


class FTRL(Optimizer):
    """ Implements FTRL online learning algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        alpha (float, optional): alpha parameter (default: 1.0)
        beta (float, optional): beta parameter (default: 1.0)
        l1 (float, optional): L1 regularization parameter (default: 1.0)
        l2 (float, optional): L2 regularization parameter (default: 1.0)

    .. _Ad Click Prediction: a View from the Trenches: 
        https://www.eecs.tufts.edu/%7Edsculley/papers/ad-click-prediction.pdf
    """

    def __init__(self, params, alpha=1.0, beta=1.0, l1=1.0, l2=1.0):
        if not 0.0 < alpha:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))
        if not 0.0 < beta:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        if not 0.0 <= l1:
            raise ValueError("Invalid l1 parameter: {}".format(l1))
        if not 0.0 <= l2:
            raise ValueError("Invalid l2 parameter: {}".format(l2))

        defaults = dict(alpha=alpha, beta=beta, l1=l1, l2=l2)
        super(FTRL, self).__init__(params, defaults)

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
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state["z"] = torch.zeros_like(p.data)
                    state["n"] = torch.zeros_like(p.data)

                z, n = state["z"], state["n"]

                theta = (n + grad ** 2).sqrt() / group["alpha"] - n.sqrt()
                z.add_(grad - theta * p.data)
                n.add_(grad ** 2)

                p.data = (
                    -1
                    / (group["l2"] + (group["beta"] + n.sqrt()) / group["alpha"])
                    * (z - group["l1"] * z.sign())
                )
                p.data[z.abs() < group["l1"]] = 0

        return loss
