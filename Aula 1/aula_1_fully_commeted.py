# Biblioteca utilizada para operações com arrays e cálculos matemáticos. É muito eficiente para manipulação 
# de grandes conjuntos de dados numéricos.
import numpy as np

# Função de ativação - Função degrau
# step_function(x): Essa função de ativação é uma função degrau. Ela recebe um valor x e retorna 1 se x 
# for maior que 0, caso contrário, retorna 0. Funções de ativação são usadas para determinar a saída de 
# um neurônio dado o seu input ponderado.

# Função de Ativação: Definição e Propósito
# Definição: Uma função de ativação é uma função matemática aplicada à soma ponderada das entradas de um neurônio, 
# determinando a saída desse neurônio.
# Propósito: Introduzir não-linearidade no modelo. Sem funções de ativação, uma rede neural composta de várias 
# camadas lineares equivaleria a uma única camada linear, limitando sua capacidade de resolver problemas complexos.
def step_function(x):
    return 1 if x > 0 else 0

# Perceptron (neuronio aritificial)
# Essa função representa um neurônio artificial básico chamado perceptron.
def perceptron(input_data, weights):
    # Calcula a soma ponderada das entradas e pesos
    # Calcula o produto escalar (soma ponderada) entre os dados de entrada input_data e os pesos weights. 
    # O produto escalar é uma operação matemática que combina as entradas e os pesos para obter uma única 
    # soma ponderada.

    # Os pesos em uma rede neural são ajustados durante o processo de treinamento. Esse processo 
    # envolve a apresentação dos dados de treinamento à rede neural, cálculo das saídas preditas pela rede, 
    # comparação dessas saídas com os resultados desejados (rótulos ou targets), e ajuste dos pesos para minimizar 
    # a diferença entre as saídas preditas e os resultados reais
    weighted_sum = np.dot(input_data, weights)
    # Aplica a função de ativação
    # Aplica a função de ativação ao valor resultante da soma ponderada. Isso determina se a saída do perceptron é 1 ou 0.
    output = step_function(weighted_sum)
    return output

# Pesos sinapticos (pesos do perceptron)
# Um array numpy contendo os pesos sinápticos iniciais para o perceptron. Neste exemplo, os pesos são [0.5, 0.5]. 
# Em uma aplicação real, os pesos geralmente são inicializados aleatoriamente e ajustados durante o treinamento.
weights = np.array([0.5, 0.5]) # Inicialização aleatória

# Dados de entrada
# Um array numpy contendo os dados de entrada para o perceptron. Neste exemplo, os dados de entrada são [0.2, 0.3].
input_data = np.array([0.2, 0.3])  # Corrigido: np_array para np.array

# Calcula a saída do perceptron
output = perceptron(input_data, weights)

# Exibe a saída
print("Saída do perceptron: ", output)