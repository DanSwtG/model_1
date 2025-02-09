#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.optim import Adam

class MAML:
    """
    Мета-обучение с использованием MAML
    """
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_optimizer = Adam(self.model.parameters(), lr=outer_lr)
    
    def adapt(self, support_set):
        """
        Быстрая адаптация модели на support_set.
        """
        fast_weights = list(self.model.parameters())
        for data, target in support_set:
            output = self.model(data)
            loss = F.mse_loss(output, target)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grad)]
        return fast_weights
    
    def meta_update(self, query_set, fast_weights):
        """
        Обновление модели на query_set.
        """
        self.outer_optimizer.zero_grad()
        for data, target in query_set:
            output = self.model(data, fast_weights)
            loss = F.mse_loss(output, target)
            loss.backward()
        self.outer_optimizer.step()

