# IMPORT LIBRARIES
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# STEP 1: LOAD DATA
file = h5py.File('metr-la.h5', 'r')
data = file['df/block0_values'][:]   # raw data

print("Dataset shape:", data.shape)
# print("Sample data first 5 rows:\n", data[:5])

# STEP 2: ANALYZE UNCERTAINTY (BEFORE ANY PROCESSING)

# Count missing values (zeros)
zero_count = np.sum(data == 0)
print("Number of zero (missing) values:", zero_count)


# Data Distribution
plt.figure(figsize=(6,4))
plt.hist(data.flatten(), bins=50)
plt.title("Data Distribution (Detect Missing & Outliers)")
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.show()


# Raw Signal (Noise & Outliers)
sensor_id = 0

plt.figure(figsize=(12,5))

# Line plot (overall trend)
plt.plot(data[:200, sensor_id], label='Signal', alpha=0.6)

# Scatter plot (points)
plt.scatter(range(200), data[:200, sensor_id], s=10)

plt.title("Noise & Outliers")
plt.xlabel("Time Steps")
plt.ylabel("Speed (mph)")
plt.legend()
plt.show()

# STEP 3: NORMALIZATION (FOR MODEL)
scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(data)


# STEP 4: CONVERT TO TENSOR
data_tensor = torch.tensor(data_scaled, dtype=torch.float32)


# STEP 5: DEFINE AUTOENCODER
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(207, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 207),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = Autoencoder()


# STEP 6: TRAIN MODEL
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epochs = 20

for epoch in range(epochs):
    output = model(data_tensor)
    loss = loss_fn(output, data_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# STEP 7: RECONSTRUCT DATA
reconstructed = model(data_tensor).detach().numpy()


# STEP 8: FULL UNCERTAINTY REMOVAL
# Replace entire dataset (no mask)
cleaned_data = reconstructed


# STEP 9: INVERSE SCALING
cleaned_data_original = scaler.inverse_transform(cleaned_data)


# # STEP 10: EVALUATION
# mse = mean_squared_error(data, cleaned_data_original)
# print("Overall MSE:", mse)


# STEP 11: FINAL COMPARISON GRAPH
plt.figure(figsize=(12,5))

# Before
plt.plot(data[:200, sensor_id],
         label='Before (uncertain data)', alpha=0.6)

# After
plt.plot(cleaned_data_original[:200, sensor_id],
         label='After (GenAI cleaned)', linewidth=2)

plt.legend()
plt.title("Before vs After Uncertainty Removal")
plt.xlabel("Time Steps")
plt.ylabel("Speed (mph)")
plt.show()

# print("Sample cleaned_data_original first 5 rows:\n", cleaned_data_original[:5])
