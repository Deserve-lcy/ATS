import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Parameters for the Beta distributions
alpha1, beta1 = 2, 5
alpha2, beta2 = 3, 5

# Generate x values
x = np.linspace(0, 1, 1000)

# Calculate the probability density function for each Beta distribution
y1 = beta.pdf(x, alpha1, beta1)
y2 = beta.pdf(x, alpha2, beta2)

# Calculate the mean (P-value) for each Beta distribution
real_P1 = alpha1 / (alpha1 + beta1)
real_P2 = alpha2 / (alpha2 + beta2)

# Plot the updated Beta distributions with mean in the legend
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label=f'real_P1 = {real_P1:.4f}', color='blue')
plt.plot(x, y2, label=f'real_P2 = {real_P2:.4f}', color='orange')
plt.title('Beta Distributions')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
