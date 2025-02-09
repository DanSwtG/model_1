#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import optuna
from optuna.trial import Trial

def objective(trial: Trial):
    # Гиперпараметры для настройки
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    model = UltraAdvancedSalesForecastModel(
        num_skus=Config.NUM_SKUS,
        num_regions=Config.NUM_REGIONS,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=hidden_dim,
        output_dim=Config.OUTPUT_DIM,
        num_heads=Config.NUM_HEADS,
        num_layers=num_layers,
        seasonality_period=Config.SEASONALITY_PERIOD
    )

    # Обучение и оценка модели
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    for epoch in range(Config.EPOCHS):
        for batch in DataLoader(dataset, batch_size=Config.BATCH_SIZE):
            optimizer.zero_grad()
            output, trend, seasonality, confidence = model(batch)
            loss = criterion(output, batch.target)
            loss.backward()
            optimizer.step()
    
    return loss.item()

# Оптимизация гиперпараметров
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

