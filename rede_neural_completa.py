"""
Reprodução da rede neural de 2 camadas dos slides do CS231n
(Fei-Fei Li, Yunzhu Li, Ruohan Gao - Lecture 4, slides 30-34).

Mesmo código do slide, com três acréscimos que NÃO alteram o algoritmo:
  - semente fixa (resultados reproduzíveis)
  - guarda a loss de cada iteração para plotar
  - verificação numérica do gradiente (gradient check)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.random import randn

np.random.seed(0)

# ---------- 1) Define the network ----------
N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)


def forward(x, w1, w2):
    h = 1 / (1 + np.exp(-x.dot(w1)))   # camada oculta com sigmoide
    y_pred = h.dot(w2)                 # camada de saída linear
    return h, y_pred


def loss_fn(y_pred, y):
    return np.square(y_pred - y).sum() # erro quadrático (L2)


# ---------- Gradient check (numérico vs. analítico) ----------
def grad_check(w1, w2, n_checks=5, eps=1e-5):
    h, y_pred = forward(x, w1, w2)
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))

    rng = np.random.RandomState(1)
    for nome, W, G in (("w1", w1, grad_w1), ("w2", w2, grad_w2)):
        for _ in range(n_checks):
            i, j = rng.randint(W.shape[0]), rng.randint(W.shape[1])
            antigo = W[i, j]
            W[i, j] = antigo + eps
            lp = loss_fn(forward(x, w1, w2)[1], y)
            W[i, j] = antigo - eps
            lm = loss_fn(forward(x, w1, w2)[1], y)
            W[i, j] = antigo
            num = (lp - lm) / (2 * eps)
            erro_rel = abs(num - G[i, j]) / max(1e-12, abs(num) + abs(G[i, j]))
            print(f"  {nome}[{i:4d},{j:3d}] analítico={G[i, j]:+.6e} "
                  f"numérico={num:+.6e} erro_rel={erro_rel:.2e}")


print("Gradient check:")
grad_check(w1, w2)

# ---------- Treino (slides 32-34) ----------
losses = []
for t in range(2000):
    # Forward pass
    h = 1 / (1 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    losses.append(loss)
    if t % 100 == 0 or t == 1999:
        print(t, loss)

    # Calculate the analytical gradients (backprop)
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)
    grad_w1 = x.T.dot(grad_h * h * (1 - h))

    # Gradient descent
    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2

# ---------- Gráfico ----------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(losses)
ax[0].set_title("Loss por iteração")
ax[0].set_xlabel("iteração t"); ax[0].set_ylabel("loss")
ax[1].semilogy(losses)
ax[1].set_title("Loss (escala log)")
ax[1].set_xlabel("iteração t")
plt.tight_layout()
plt.savefig("loss.png", dpi=150)
print("Gráfico salvo em loss.png")
