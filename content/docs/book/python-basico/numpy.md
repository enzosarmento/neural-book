---
weight: 3
title: "NumPy"
---

# NumPy

O **NumPy** é a biblioteca fundamental para cálculos numéricos em Python.  
Ela fornece arrays multidimensionais de alta performance e funções matemáticas otimizadas.  
Em Engenharia, NumPy é usada para representar **vetores, matrizes**, resolver **sistemas lineares**, fazer **simulações numéricas** e preparar dados para modelos de machine learning.

> **Objetivo deste capítulo:** apresentar os conceitos essenciais do NumPy, com exemplos práticos.

---

## Índice rápido
- [Criando arrays](#criando-arrays)  
- [Sequências numéricas: `arange` vs `linspace`](#sequências-numéricas)  
- [Aleatoriedade e reprodutibilidade](#aleatoriedade-e-reprodutibilidade)  
- [Shape, `ndim` e `dtype`](#anatomia-de-um-array-shape-ndim-e-dtype)  
- [Indexação e slicing](#indexação-e-fatiamento-slicing-acessando-seus-dados)  
- [Operações vetorizadas & broadcasting](#operações-vetorizadas--broadcasting)  
- [Álgebra linear básica](#álgebra-linear-básica)  
- [Estatísticas e funções úteis](#funções-estatísticas-e-agregações)
- [Referências rápidas](#referências-rápidas)

---

## Criando arrays

A forma mais comum de criar um array é a partir de uma lista Python.

```python
import numpy as np

# Criando um array 1D (vetor) a partir de uma lista de números decimais
a = np.array([1.2, 2.4, 3.5])

# Criando um array 2D (matriz 2x2) a partir de uma lista de listas
M = np.array([[1, 2], [3, 4]])

# Vamos inspecionar nossos arrays
print("Vetor 'a':", a)
print("Tipo de dado de 'a':", a.dtype) # dtype = data type (tipo de dado)
print("Formato de 'a':", a.shape)      # shape = formato (dimensões)

print("\nMatriz 'M':\n", M)
print("Tipo de dado de 'M':", M.dtype)
print("Formato de 'M':", M.shape)
```

**O que significam dtype e shape?**

- Informa o tipo de dado armazenado no array. `float64` representa números de ponto flutuante com 64 bits de precisão (o padrão para decimais). `int64` representa inteiros. Manter todos os elementos com o mesmo tipo é um dos segredos da eficiência do NumPy. 
- `shape`: É uma tupla que descreve as dimensões do array.
  - `(3,)` significa que é um array de 1 dimensão com 3 elementos (um vetor de comprimento 3).
  - `(2, 2)` significa que é um array de 2 dimensões, com 2 linhas e 2 colunas (uma matriz 2x2).

---

## Sequências numéricas
Para análises e gráficos, frequentemente precisamos gerar sequências de números. O NumPy oferece duas funções principais para isso:

`np.arange(início, fim, passo)`

Funciona de forma muito parecida com o `range` do Python. Ele gera números em um intervalo com um passo definido. **Importante: o valor `fim` não é incluído na sequência.**

```python
# Gera números de 0 a 1 (sem incluir o 1), com passo de 0.2
tempo = np.arange(0, 1, 0.2)
print(tempo)  # Saída: [0. , 0.2, 0.4, 0.6, 0.8]
```

**Quando usar?** Ideal quando o tamanho do passo entre os pontos é o mais importante.

`np.linspace(início, fim, quantidade_de_pontos)`

Gera um número específico de pontos igualmente espaçados entre o `início` e o `fim`. **Diferente do `arange`, o valor `fim` é incluído por padrão.**

```python
# Gera 5 pontos igualmente espaçados entre 0 e 1 (incluindo ambos)
posicao = np.linspace(0, 1, 5)
print(posicao)  # Saída: [0. , 0.25, 0.5 , 0.75, 1. ]
```

**Quando usar?** Ideal quando o número total de pontos é mais importante que o passo exato.

---

## Aleatoriedade e reprodutibilidade

Para simulações (como o Método de Monte Carlo) ou para gerar dados de teste, precisamos de números aleatórios. A prática moderna no NumPy é usar um "Gerador".

```python
# Cria um gerador de números aleatórios. A 'seed' é uma semente.
# Usar a mesma seed (ex: 42) garante que os mesmos números "aleatórios" sejam gerados sempre.
# Isso é fundamental para que seus experimentos e simulações sejam REPRODUTÍVEIS.
rng = np.random.default_rng(seed=42)

# Gera 6 números inteiros aleatórios entre 50 (inclusivo) e 100 (exclusivo)
inteiros_aleatorios = rng.integers(low=50, high=100, size=6)

# Gera 5 números decimais aleatórios uniformemente distribuídos entre 0.0 e 1.0
reais_aleatorios = rng.random(5)

print("Inteiros gerados:", inteiros_aleatorios)
print("Reais gerados:", reais_aleatorios)
```

> **Dica**: Sempre defina uma seed quando estiver desenvolvendo um código que usa aleatoriedade. Isso garante que você possa rodar o código novamente e obter exatamente os mesmos resultados, o que é crucial para depuração e validação.

---

## Anatomia de um Array: `shape`, `ndim` e `dtype`

Todo array NumPy possui atributos essenciais que descrevem sua estrutura.

```python
# Cria um vetor com 12 elementos (de 0 a 11)
x = np.arange(12)
print("Vetor original:", x)
print("Shape:", x.shape)  # Formato -> (12,)
print("Ndim:", x.ndim)    # Número de dimensões -> 1
print("Dtype:", x.dtype)  # Tipo do dado -> int64

# Agora, vamos remodelar esse vetor para uma matriz 3x4
# O número total de elementos (12) deve ser mantido.
X = x.reshape(3, 4)
print("\nMatriz remodelada:\n", X)
print("Shape:", X.shape)  # Formato -> (3, 4)
print("Ndim:", X.ndim)    # Número de dimensões -> 2
```
- **Transformações úteis:**
   - `.reshape(linhas, colunas)`: Modifica o formato do array, sem alterar os dados.
   - `.flatten()` ou `.ravel()`: Transforma qualquer matriz em um vetor 1D.
   - `.astype(novo_tipo)`: Converte o tipo de dado do array (ex: de inteiro para float). `X.astype(np.float32)` economiza memória se a precisão de 64 bits não for necessária.

---

## Indexação e Fatiamento (Slicing): Acessando seus Dados

Esta é a forma de selecionar elementos, linhas, colunas ou sub-regiões de um array. A indexação em Python **começa em zero**.

Indexação 1D (Vetores):

```python
a = np.array([10, 20, 30, 40, 50])
print("Primeiro elemento:", a[0])      # Pega o elemento no índice 0 -> 10
print("Último elemento:", a[-1])     # Pega o último elemento -> 50
print("Do segundo ao quarto:", a[1:4]) # Pega do índice 1 até o 4 (sem incluir o 4) -> [20, 30, 40]
```

Indexação em 2D (Matrizes) - `matriz[linha, coluna]`

```python
m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print("Elemento na primeira linha, primeira coluna:", m[0, 0]) # -> 1

# O símbolo ':' significa "todos os elementos daquela dimensão"
print("Toda a segunda linha:", m[1, :])  # Pega a linha de índice 1 -> [4, 5, 6]
print("Toda a terceira coluna:", m[:, 2]) # Pega a coluna de índice 2 -> [3, 6, 9]
```

Indexação Booleana (Filtragem por Condição)

Essa é uma técnica extremamente poderosa para filtrar dados.

```python
v = np.array([1, 2, 3, 4, 5, -1, -2])

# 1. Criar uma condição (máscara)
mascara = v > 2
print("Máscara booleana:", mascara) # -> [False, False, True, True, True, False, False]

# 2. Usar a máscara para selecionar apenas os elementos onde a condição é True
print("Elementos maiores que 2:", v[mascara]) # -> [3, 4, 5]

# Pode ser feito em uma única linha:
print("Elementos positivos:", v[v > 0]) # -> [1, 2, 3, 4, 5]
```

---

## Operações vetorizadas & broadcasting

Operações matemáticas entre arrays são aplicadas elemento a elemento, sem a necessidade de loops explícitos. Isso é chamado de **vetorização** e é uma das principais razões para a eficiência do NumPy.

```python
x = np.array([1, 2, 3])
y = np.array([10, 20, 30])

# Operações elemento a elemento (muito mais rápido que um laço for)
soma = x + y          # -> [11, 22, 33]
multiplicacao = x * y # -> [10, 40, 90]
potencia = x ** 2     # -> [1, 4, 9]

# Operação com um escalar (um único número)
multiplicado_por_2 = x * 2 # -> [2, 4, 6]

print("Soma:", soma)
print("Multiplicado por 2:", multiplicado_por_2)
```

**Broadcasting: Operando com Arrays de Formatos Diferentes**
O **Broadcasting** é um mecanismo poderoso que permite ao NumPy realizar operações em arrays de formatos diferentes. A regra geral é: se uma dimensão em um dos arrays for 1, ela pode ser "esticada" para corresponder à dimensão do outro array.

**Exemplo prático**: Somar um vetor a cada linha de uma matriz.

```python
# Matriz 2x3
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# Vetor com 3 elementos
b = np.array([10, 20, 30])

# O NumPy "estica" ou "transmite" (broadcasts) o vetor 'b'
# para cada linha da matriz 'A' antes de somar.
# A forma de 'b' (3,) é compatível com a segunda dimensão de 'A' (2, 3).
C = A + b

print("Matriz A:\n", A)
print("\nVetor b:", b)
print("\nResultado A + b:\n", C)
# [[11, 22, 33],
#  [14, 25, 36]]
```

Isso evita a necessidade de criar uma matriz com cópias do vetor `b`, economizando memória e sendo mais eficiente.

---

## Álgebra linear básica

Produto matricial, transposta, determinante, inversa, resolução de sistemas:

```python
A = np.array([[3, 1],
              [1, 2]])
b = np.array([9, 8])

# produto matricial
C = A @ A   # ou np.dot(A, A)

# resolver A x = b
x = np.linalg.solve(A, b)

# inversa e determinante (use com cuidado)
invA = np.linalg.inv(A)
detA = np.linalg.det(A)

print("x =", x)
```

> **Nota:** usar `np.linalg.solve` é preferível a calcular `inv(A)` e multiplicar, por estabilidade e desempenho.

---

## Funções estatísticas e agregações

Analisar um conjunto de dados é uma tarefa comum. NumPy oferece funções estatísticas otimizadas.

```python
# Dados de um ensaio (ex: resistência em MPa)
dados = np.array([10.2, 20.5, 31.0, 39.8, 50.1])

print("Média:", np.mean(dados))
print("Mediana:", np.median(dados))
print("Desvio Padrão:", np.std(dados))
print("Valor Mínimo:", np.min(dados))
print("Valor Máximo:", np.max(dados))
print("Soma Total:", np.sum(dados))
```

**Agregações por Eixo em Matrizes**
Você pode calcular estatísticas ao longo de linhas ou colunas de uma matriz usando o argumento `axis`.
* `axis=0`: Opera "para baixo" (ao longo das linhas), resultando em um valor por coluna.
* `axis=1`: Opera "para o lado" (ao longo das colunas), resultando em um valor por linha.

```python
M = np.arange(12).reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

# Calcula a média de cada coluna
media_colunas = np.mean(M, axis=0)
print("Matriz M:\n", M)
print("\nMédia por coluna:", media_colunas) # -> [4., 5., 6., 7.]

# Calcula a soma de cada linha
soma_linhas = np.sum(M, axis=1)
print("Soma por linha:", soma_linhas) # -> [ 6, 22, 38]
```

---

## Referências rápidas

- Documentação oficial: [numpy.org/doc](https://numpy.org/doc/)  
- Quickstart: [numpy.org/devdocs/user/quickstart](https://numpy.org/devdocs/user/quickstart.html)

---

## Observações finais

Dominar NumPy torna muito mais eficiente o trabalho com **Pandas**, **SciPy** e frameworks de redes neurais.  
Este capítulo serve como **fundação** para todo o conteúdo posterior do *NeuralBook*.
