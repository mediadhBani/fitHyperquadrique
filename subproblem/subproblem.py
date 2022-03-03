# -*- coding: utf-8 -*-

# bibliotheques des graphes et fonctions mathematiques
import matplotlib.pyplot as plt
import numpy as np


# %%

# hyperquadratique
def hyperQuad(x, y, Lambda):
    som = 0
    for a, b, c, g in Lambda:
        som += abs(a*x + b*y + c)**g
    return som-1


# droites enveloppantes
def droiteEnv(hq, limy):
    """Tracé des droites enveloppantes de l'hyperquadratique `hq` dans le
    domaine limité en ordonnée par l'intervalle `limy` (couple)"""
    for a, b, c, _ in hq:
        if b:
            plt.axline((0, (c+1)/-b), slope=-a/b, lw=1.1)
            plt.axline((0, (c-1)/-b), slope=-a/b, lw=1.1)
        else:
            plt.vlines([-(c+1)/a, (1-c)/a], *limy, lw=1.1)


# Parametres de l'hyperquadratique
def hyperDescrip(hq):
    """Description des paramètres de l'hyperquadratique `hq`."""
    for (k, (a, b, c, g)) in enumerate(hq):
        print('Terme %2d : a = %6.3f ; b = %6.3f ; c = %6.3f ; gamma = %2d' %
              (k+1, a, b, c, g))


# %%
# Extraction de points
with open('Data_HQ_Ph1et2.csv', 'r') as f:
    x, y = [[float(u) for u in line.split(',')] for line in f.readlines()]


# probleme simplifie
def psi(x, y, a, b):
    return (a*x+b*y)**4 + (x+y)**4 - 1


# critere quadratique. Fonction objectif de minimisation de distance
def J(x, y, a, b):
    som = 0
    for u, v in zip(x, y):
        som += psi(u, v, a, b)**2
    return som


# %% Descente de gradient
def dPsi(x, y, a, b):
    return [4*x*(a*x+b*y)**3, 4*y*(a*x+b*y)**3]


def dJ(x, y, a, b):  # Derivation sur a et b
    som = [0, 0]
    for u, v in zip(x, y):
        etape = 8*psi(u, v, a, b)*(a*u+b*v)**3
        som[0] += u*etape
        som[1] += v*etape
    return som


def gradient(x, y, a0, b0, alpha, eps, nmax):
    dX = eps+1           # reel
    n = 0                # entier
    Xa, Xb = [a0], [b0]  # vecteurs de reels

    while dX > eps and n < nmax:
        D = dJ(x, y, Xa[-1], Xb[-1])
        Xa.append(Xa[-1] - alpha*D[0])
        Xb.append(Xb[-1] - alpha*D[1])
        dX = alpha*np.sqrt(D[0]**2 + D[1]**2)
        n += 1

    return Xa, Xb, n, (dX <= eps)


# %% Illustration
a = np.linspace(-1, 1, 100)
b = np.linspace(-1, 1, 100)
a2D, b2D = np.meshgrid(a, b)

Xa, Xb, nIter, converge = gradient(x, y, .1, -.1, .007, 1e-6, 50)

iso = 3*(np.logspace(0, 1)-1)

plt.figure(num='gradient')
plt.contour(a2D, b2D, J(x, y, a2D, b2D), levels=iso)
plt.plot(Xa, Xb, '-ro', lw=1.5, markersize=3)
plt.plot(Xa[-1], Xb[-1], 'ko', markersize=3)

plt.title("Méthode du gradient")
plt.xlabel('a')
plt.ylabel('b')
plt.axis('square')

plt.show()


# %% Methode de Newton
def HJ(x, y, a, b):
    som = [[0, 0], [0, 0]]
    for u, v in zip(x, y):
        etape1 = (a*u+b*v)
        etape2 = 8*etape1**2
        etape3 = etape2*(4*etape1**4 + 3*psi(u, v, a, b))
        som[0][0] += u*u*etape3
        som[0][1] += u*v*etape3
        som[1][0] += u*v*etape3
        som[1][1] += v*v*etape3
    return som


def newton(x, y, a0, b0, eps, nmax):
    dX = float('inf')    # reel
    n = 0                # entier
    Xa, Xb = [a0], [b0]  # vecteurs de reels

    while dX > eps and n < nmax:
        d1J = np.array(dJ(x, y, Xa[-1], Xb[-1]))
        d2J = np.array(HJ(x, y, Xa[-1], Xb[-1]))

        DeltaX = np.tensordot(-np.linalg.inv(d2J), d1J, 1)
        Xa.append(Xa[-1]+DeltaX[0])
        Xb.append(Xb[-1]+DeltaX[1])
        dX = np.linalg.norm(DeltaX)

        n += 1

    return Xa, Xb, n, (dX <= eps)


# %% Illustration de la méthode de Newton

iso = 3*(np.logspace(0, 1)-1)                         # valeurs des isovaleurs
plt.figure(num='newton')                              # nom de figure
plt.contour(a2D, b2D, J(x, y, a2D, b2D), levels=iso)  # tracé des isovaleurs
plt.colorbar()

departs = np.random.rand(5, 2)*2-1
for X in departs:
    Xa, Xb, nIter, converge = newton(x, y, X[0], X[1], 1e-6, 20)
    plt.plot(Xa, Xb, '-ro', lw=1.5, markersize=3)
    plt.plot(Xa[-1], Xb[-1], 'ko', markersize=3)
    if converge:  # on part du principe que converge est vraie au moins 1 fois
        a, b = Xa[-1], Xb[-1]

plt.title("Méthode de Newton")
plt.xlabel('a')
plt.ylabel('b')
plt.axis('square')

# %% Résultat sur le nuage de points
plt.figure('hyperquad')
plt.scatter(x, y)

borne = 1.5
u = np.linspace(-borne, borne)
v = np.linspace(-borne, borne)
u2D, v2D = np.meshgrid(u, v)
plt.contour(u2D, v2D, psi(u2D, v2D, a, b), levels=[0], colors='k')

plt.title('Hyperquadratique ajustée au nuage de points')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
