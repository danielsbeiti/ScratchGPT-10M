import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

losses = pd.read_csv('loss.csv')

plt.figure(figsize=(10, 5))
plt.plot(losses['iter'], losses['train'], label='Train Loss')
plt.plot(losses['iter'], losses['val'], label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')