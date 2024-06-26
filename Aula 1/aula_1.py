import numpy as np

# Função de ativação - Função degrau
def step_function(x):
    return 1 if x > 0 else 0

# Perceptron (neuronio aritificial)
def perceptron(input_data, weights):
    # Calcula a soma ponderada das entradas e pesos
    weighted_sum = np.dot(input_data, weights)
    # Aplica a função de ativação
    output = step_function(weighted_sum)
    return output

# Pesos sinapticos (pesos do perceptron)
weights = np.array([0.5, 0.5]) # Inicialização aleatória

# Dados de entrada
input_data = np.array([0.2, 0.3])  # Corrigido: np_array para np.array

# Calcula a saída do perceptron
output = perceptron(input_data, weights)

# Exibe a saída
print("Saída do perceptron: ", output)