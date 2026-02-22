import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import cv2
import sklearn
import pandas as pd
import seaborn as sns
import scipy

print("=" * 50)
print("PYTHON VERSION:")
print(sys.version)

print("=" * 50)
print("NUMPY TEST:")
a = np.random.rand(3, 3)
print("Array:\n", a)
print("Mean:", np.mean(a))

print("=" * 50)
print("TORCH TEST:")
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("GPU tensor multiplication OK. Shape:", y.shape)
else:
    print("Running on CPU only.")

print("=" * 50)
print("TORCHVISION TEST:")
print("Torchvision version:", torchvision.__version__)

print("=" * 50)
print("OPENCV TEST:")
print("OpenCV version:", cv2.__version__)

print("=" * 50)
print("SKLEARN / PANDAS / SEABORN / SCIPY TEST:")
print("sklearn:", sklearn.__version__)
print("pandas:", pd.__version__)
print("seaborn:", sns.__version__)
print("scipy:", scipy.__version__)

print("=" * 50)
print("MATPLOTLIB TEST (saving test_plot.png)...")

plt.plot([0, 1, 2], [0, 1, 4])
plt.title("Test Plot")
plt.savefig("test_plot.png")
plt.close()

print("Plot saved successfully.")

print("=" * 50)
print("ALL CHECKS COMPLETED.")