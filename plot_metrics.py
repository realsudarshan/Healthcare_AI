import matplotlib.pyplot as plt
import os

epochs = list(range(1, 11))
train_loss = [0.6988, 0.3013, 0.2253, 0.1963, 0.1703, 0.1499, 0.1302, 0.1279, 0.1155, 0.1138]
val_loss = [0.4501, 0.3189, 0.2786, 0.2497, 0.2177, 0.1994, 0.2112, 0.1900, 0.1912, 0.1997]

train_acc = [0.7604, 0.8773, 0.9068, 0.9221, 0.9357, 0.9425, 0.9503, 0.9512, 0.9570, 0.9532]
val_acc = [0.8405, 0.8959, 0.9097, 0.9097, 0.9108, 0.9210, 0.9226, 0.9267, 0.9262, 0.9277]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss
ax1.plot(epochs, train_loss, 'b-', label='Training Loss', marker='o')
ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', marker='o')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# Plot Accuracy
ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', marker='o')
ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', marker='o')
ax2.set_title('Training and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

plt.tight_layout()

# Save to artifact directory
out_dir = r"C:\Users\sudar\.gemini\antigravity\brain\47a75e81-8372-4486-b739-fd982c3f56de\scratch"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "training_metrics.png"), dpi=300)
print(f"Saved plot to {os.path.join(out_dir, 'training_metrics.png')}")
