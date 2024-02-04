import numpy as np
import matplotlib.pyplot as plt


# APEC
def apec(x, a, b, g, eps=1e-5):
    return a + (b - x) / (g - np.exp(-x) + eps)


# MAPEC
def mapec(x, a, b, g, d):
    return a + (b - x) / (g - np.exp(-x)) + (x * d)


x_values = np.linspace(-10, 10, 500)
params_apec = [
    (0, 0, -1.375),  # Default
    (0.75, -0.75, -2.0),
    (0.25, -0.25, -1.0),
    (0.00, 0.00, -0.75),
    (-0.75, 0.75, -0.5),
]
params_mapec = [
    (0, 0, -1, 0),  # Default
    (0.75, -0.75, -2.0, -0.5),
    (0.25, -0.25, -1.0, -0.25),
    (0.00, 0.00, -0.75, 0.25),
    (-0.75, 0.75, -0.5, 0.5),
]

# Plot APEC
plt.figure(figsize=(12, 6))
for a, b, g in params_apec:
    y_values = apec(x_values, a, b, g)
    plt.plot(x_values, y_values, label=f"APEC: a={a}, b={b}, g={g}")
plt.title("APEC Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()

# Plot MAPEC
plt.figure(figsize=(12, 6))
for a, b, g, d in params_mapec:
    y_values = mapec(x_values, a, b, g, d)
    plt.plot(x_values, y_values, label=f"MAPEC: a={a}, b={b}, g={g}, d={d}")
plt.title("MAPEC Activation Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
