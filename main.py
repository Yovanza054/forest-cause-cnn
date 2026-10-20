import os
from src.train import train
from src.evaluate import evaluate
from src.alert_map import draw_alert_map

os.makedirs("outputs", exist_ok=True)
model, x_te, y_te, history = train()
evaluate(model, x_te, y_te, history)
draw_alert_map(model)
