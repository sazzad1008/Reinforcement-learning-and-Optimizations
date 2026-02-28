# PPO project

This folder contains a standalone, from-scratch PPO implementation for a synthetic
complex-data task:

- `config.py`: central training and model configuration
- `data.py`: complex feature generation and split helpers
- `model.py`: NumPy MLP actor-critic model
- `trainer.py`: PPO training loop and evaluation metrics
- `run.py`: executable entrypoint

Run:

```bash
python -m PPO.run
```
