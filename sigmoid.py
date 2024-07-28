import numpy as np

def sigmoid(x, der=False):
    if der == True:
        return x * (1-x)
    return 1 / (1 + np.exp(-x))

def nonlin(x, deriv=False):
    if (deriv == True):
        return (x) * (1 - (x))
    return 1 / (1 + np.exp(-x))

# входные данные
x = np.array([[1, 0, 1],
              [1, 0, 0],
              [0, 1, 0],
              [0, 1, 0]])
# выходные данные
y = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)
syn0 = 2 * np.random.random((3, 1)) - 1

# обучение
l1 = []

for iter in range(10000):
    l0 = x
    l1 = sigmoid(np.dot(l0, syn0))

    l1_error = y - l1
    l1_delta = l1_error * sigmoid(l1, True)

    syn0 += np.dot(l0.T, l1_delta)

print("Выходные даннные после обучения: ")
print(l1)

# проверяем как обучилась
new_one = np.array([1, 0, 1])
l1_new = nonlin(np.dot(new_one, syn0))
print("Новые данные: ")
print(l1_new)

