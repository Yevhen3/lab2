import numpy as np
import matplotlib.pyplot as plt
from scipy.special import chebyt
from scipy.integrate import quad

# Визначення функції f(x)
def f(x):
    return np.exp(-np.sqrt(2*x)) * np.sin(x)**2

# Визначення кількості базисних функцій n
n = 10

def approximation(x):
    t = (2*x - 2.5) / 2.5
    c = [quad(lambda x: f((2.5*x + 2.5)/2) * chebyt(i)(x), -1, 1)[0] for i in range(n)]

    S = sum(c[i]*chebyt(i)(t) for i in range(n))
    return S

# Генерування значень x_values
x_values = np.linspace(0, 2.5, 1000)

# Обчислення значень функції f(x) та апроксимації S(x)
f_values = f(x_values)
S_values = [approximation(x) for x in x_values]

# Нормалізація значень f(x) та S(x) до проміжку [-1, 1]
f_values_normalized = 2*(f_values - np.min(f_values))/(np.max(f_values) - np.min(f_values)) - 1
S_values_normalized = 2*(S_values - np.min(S_values))/(np.max(S_values) - np.min(S_values)) - 1

# Створення масиву nodes
nodes = np.linspace(0, 2.5, 10)

# Обчислення значень функції f(x) та апроксимації S(x) у вузлах
f_nodes = f(nodes)
S_nodes = [approximation(x) for x in nodes]

# Нормалізація значень f(x) та S(x) у вузлах до проміжку [-1, 1]
f_nodes_normalized = 2*(f_nodes - np.min(f_nodes))/(np.max(f_nodes) - np.min(f_nodes)) - 1
S_nodes_normalized = 2*(S_nodes - np.min(S_nodes))/(np.max(S_nodes) - np.min(S_nodes)) - 1

# Побудова графіка
plt.plot(x_values, f_values_normalized, label='f(x)')
plt.plot(x_values, S_values_normalized, '--', label='S(x)')
plt.plot(nodes, f_nodes_normalized, 'ro')
plt.plot(nodes, S_nodes_normalized, 'bo')
plt.legend()
plt.show()

for i in range(len(nodes)):
    print(f"f({nodes[i]}) = {f_nodes_normalized[i]}, S({nodes[i]}) = {S_nodes_normalized[i]}, error = {abs(f_nodes_normalized[i] - S_nodes_normalized[i])}")

