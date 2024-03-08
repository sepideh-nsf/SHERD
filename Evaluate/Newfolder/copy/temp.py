import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

# Define the x and y variables
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Define the group membership (e.g., 'Group A', 'Group B')
groups = ['Group A', 'Group A', 'Group B', 'Group B', 'Group B']

# Combine the data into a Pandas DataFrame
data = pd.DataFrame({'x': x, 'y': y, 'groups': groups})

# Create the scatter plot using Seaborn
sns.scatterplot(data=data, x='x', y='y', hue='groups', palette=['red', 'blue'], s=100)

# Show the plot
plt.show()


# Generate a random dataset
data = np.random.normal(loc=10, scale=2, size=100)

# Calculate the mean and variance of the dataset
mean = np.mean(data)
variance = np.var(data)

# Plot a histogram of the dataset with the mean and variance as annotations
fig, ax = plt.subplots()
ax.hist(data, bins=10)
ax.axvline(mean, color='r', linestyle='--', label='Mean')
ax.axvline(mean + np.sqrt(variance), color='g', linestyle='--', label='Standard deviation')
ax.axvline(mean - np.sqrt(variance), color='g', linestyle='--')
ax.legend()
plt.show()