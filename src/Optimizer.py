from torch.optim import Optimizer

class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, beta = 0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data = p.data - (group['lr'] * d_p)
        
        return loss

class PerturbedSGD(Optimizer):
    """Perturbed SGD optimizer"""

    def __init__(self, params, lr=0.01, alpha=0.0):

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, alpha=alpha)
        self.idx = 0

        super(PerturbedSGD, self).__init__(params, defaults)
    
    def step(self, y_ik, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure

        yik_update = y_ik.parameters()
        # internal sgd update
        for group in self.param_groups:
            #get the lr
            lr = group['lr']
            alpha = group['alpha']

            for param, y_param in zip(group['params'], yik_update):
                param.grad.data = param.grad.data + alpha*(param.data - y_param.data)
                param.data = param.data -lr*param.grad.data
                
       
        return group['params'], loss
    
