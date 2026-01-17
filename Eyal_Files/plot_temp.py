import matplotlib.pyplot as plt
import numpy as np

# Create some sample data for the plots
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# Create a figure and subplot grid with 3 rows and 1 column, with equal heights for all subplots
fig, axs = plt.subplots(3, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [1, 1, 1]})

# Plot the first row with a colorbar
cax = axs[0].imshow(np.random.rand(10, 10), cmap='viridis')
fig.colorbar(cax, ax=axs[0], orientation='vertical', label='Colorbar Label')

# Plot the second row
axs[1].plot(x, y2, color='blue')
axs[1].set_ylabel('Y2 Label')

# Plot the third row
axs[2].plot(x, y3, color='green')
axs[2].set_xlabel('X Label')
axs[2].set_ylabel('Y3 Label')

# Adjust subplot spacing
plt.tight_layout()

# Show the plot
plt.show()
