# %%
import matplotlib.pyplot as plt
import numpy as np

avg_train_losses = np.random.rand(10)
avg_test_losses = np.random.rand(10)
# 绘制损失曲线

plt.figure(figsize=(8, 5))
plt.plot(avg_train_losses, "o", label="Train Loss")
plt.plot(avg_test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.grid(True)
plt.show()
plt.close()

# %%

import torchvision.models as models

print(models.resnet18())

# %%
