import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("output/benchmark.csv")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df["Image"], df["Convolution(ms)"], marker='o', label='Convolution (ms)')
plt.plot(df["Image"], df["MaxPooling(ms)"], marker='s', label='Max Pooling (ms)')

plt.title("CUDA Kernel Execution Time per Image")
plt.xlabel("Image")
plt.ylabel("Time (ms)")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/benchmark_plot.png")
plt.show()
