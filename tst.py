import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# Generowanie próbek
data = expon(scale=5).rvs(100)  # scale = 1/lambda, lambda = 0.2, scale = 5

# Definicja jądra Gaussowskiego z odbiciem i normalizacją
def reflected_gaussian_kernel(x, xi, h):
    return 0.5 * (np.exp(-((x - xi)/h)**2 / 2) + np.exp(-((x + xi)/h)**2 / 2)) / (h * np.sqrt(2 * np.pi))

# Jądrowa estymacja gęstości z modyfikacją odbijającą i normalizacją
def kde_reflected(data, bandwidth, grid):
    densities = np.zeros_like(grid)
    for xi in data:
        densities += reflected_gaussian_kernel(grid, xi, bandwidth)
    densities /= len(data)
    return densities

# Siatka punktów do estymacji
grid = np.linspace(0, max(data)*1.5, 1000)

# Wybór szerokości pasma
bandwidth = 1.06 * data.std() * np.power(len(data), -1/5)  # reguła Silvermana

# Obliczenie estymacji gęstości
density = kde_reflected(data, bandwidth, grid)

# Rysowanie wyników
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.5, label='Histogram')
plt.plot(grid, density, label='KDE with Reflective Boundary')
plt.title('Kernel Density Estimation with Reflective Boundary')
plt.legend()
plt.show()
