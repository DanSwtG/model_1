#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    """
    Многозадачная модель для прогнозирования продаж, классификации популярности и анализа трендов.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiTaskModel, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.regression_head = nn.Linear(hidden_dim, output_dim)  # Прогнозирование продаж
        self.classification_head = nn.Linear(hidden_dim, 2)      # Классификация популярности
        self.trend_head = nn.Linear(hidden_dim, 1)               # Анализ трендов
    
    def forward(self, x):
        shared_features = self.shared_layer(x)
        sales_pred = self.regression_head(shared_features)
        popularity_pred = self.classification_head(shared_features)
        trend_pred = self.trend_head(shared_features)
        return sales_pred, popularity_pred, trend_pred

