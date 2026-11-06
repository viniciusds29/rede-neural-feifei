# Rede neural de 2 camadas em NumPy (CS231n – Lecture 4)

Reprodução e explicação da rede neural dos slides 30–34 da Lecture 4 de
Fei-Fei Li, Yunzhu Li e Ruohan Gao (Stanford CS231n, 2023).

## Alunos

- Vinicius Oliveira de Souza
- Ana Soraia Barroso Silva

## Arquivos

- `rede_neural_slide.py`: código idêntico ao slide (20 linhas).
- `rede_neural_completa.py`: mesmo algoritmo, com semente fixa, gráfico da loss e *gradient check*.
- `loss.png`: curva da loss.

## Como rodar

```bash
pip install -r requirements.txt
python rede_neural_slide.py        # imprime "t loss" a cada iteração
python rede_neural_completa.py     # gradient check + treino + loss.png
```

## Explicação do código

**1. Define the network (linhas 4–6).** 64 exemplos (`N`), 1000 entradas (`D_in`),
100 neurônios ocultos (`H`) e 10 saídas (`D_out`). `x` e `y` são dados aleatórios (não há
tarefa real, o objetivo é mostrar o mecanismo de treino). Os pesos `w1` (1000×100) e
`w2` (100×10) começam aleatórios. Não há bias.

**2. Forward pass (linhas 9–11).**

```python
h = 1 / (1 + np.exp(-x.dot(w1)))   # camada oculta: sigmoide
y_pred = h.dot(w2)                 # saída linear
loss = np.square(y_pred - y).sum() # erro quadrático: L = Σ (y_pred − y)²
```

**3. Gradientes analíticos (linhas 14–17).** É a backpropagation escrita à mão, aplicando
a regra da cadeia de trás para frente:

| Código | Derivação |
|---|---|
| `grad_y_pred = 2.0 * (y_pred - y)` | `∂L/∂y_pred = 2(y_pred − y)` |
| `grad_w2 = h.T.dot(grad_y_pred)` | `y_pred = h·w2` ⇒ `∂L/∂w2 = hᵀ · ∂L/∂y_pred` |
| `grad_h = grad_y_pred.dot(w2.T)` | `∂L/∂h = ∂L/∂y_pred · w2ᵀ` |
| `grad_w1 = x.T.dot(grad_h * h * (1 - h))` | `σ'(z) = h(1 − h)`, então `∂L/∂w1 = xᵀ · (∂L/∂h ⊙ h(1 − h))` |

Os `.T` e a ordem das multiplicações vêm das dimensões: cada gradiente tem o mesmo
formato do peso correspondente.

**4. Gradient descent (linhas 19–20).** `w -= 1e-4 * grad`: cada peso anda um pouco na
direção oposta ao gradiente. `1e-4` é a taxa de aprendizado, e o laço roda 2000 iterações.

## Resultados

Com `np.random.seed(0)`, a loss cai de ≈ 24 636 (t=0) para ≈ 2 383 (t=100), ≈ 41 (t=1000)
e ≈ 4,35 (t=1999). O código do slide original não fixa semente, então os valores exatos
mudam a cada execução, mas o comportamento esperado é o mesmo: loss decrescendo de forma
rápida no início e mais lenta depois.

![Curva da loss](loss.png)

**Por que a loss cai tanto com dados aleatórios?** A rede tem 1000×100 + 100×10 = 101 000
parâmetros para ajustar só 64×10 = 640 saídas, então ela memoriza o lote (overfitting).
Num problema real, o que importa é a generalização para dados novos.

**Gradient check.** O `rede_neural_completa.py` compara o gradiente analítico com o
numérico, `(L(w+ε) − L(w−ε)) / 2ε`, em pesos aleatórios. O erro relativo ficou entre
~1e-11 e ~1e-8, confirmando que as fórmulas do slide estão corretas.

## Referência

Fei-Fei Li, Yunzhu Li, Ruohan Gao. *CS231n – Lecture 4: Backpropagation and Neural Networks*, slides 30–34, 2023.
