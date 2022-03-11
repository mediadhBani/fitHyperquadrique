import logging
import matplotlib.pyplot as plt
import numpy as np
import sys

# logging.basicConfig(level=logging.DEBUG, filename="simpleProblem.log",
#                     filemode='w')

def hyperquadrique(x: float, y: float, lsParam: list[list[float]]) -> float:
    """Calcule de la fonction hyperquadrique en un point donne.
    
    Entree:
    - x, y: coordonnees du point en argument de la fonction hyperquadrique
    - lsParam: liste des paramètres de l'hyperquadrique
    
    Sortie: valeur de l'hyperquadrique
    """
    som = 0
    for a, b, c, g in lsParam:
        som += abs(a*x + b*y + c)**g
    return som-1

def droite_enveloppante(lsParam: list[list[float]], ylim: tuple) -> None:
    """Trace les droites enveloppantes de l'hyperquadrique de parametres donnes.
    
    Entree:
    - lsParam: liste des parametres de l'hyperquadrique
    - ylim: intervalle d'affichage des droites enveloppantes
    """
    for a, b, c, _ in lsParam:
        if b:
            plt.axline((0, (c+1)/-b), slope=-a/b, lw=1.1)
            plt.axline((0, (c-1)/-b), slope=-a/b, lw=1.1)
        else:
            plt.vlines([-(c+1)/a, (1-c)/a], *ylim, lw=1.1)

def decrire_hyperquadrique(lsParam: list[list[float]]) -> None:
    """Description des paramètres d'une hyperquadrique.
    
    Entree:
    - lsParam: liste des parametres de l'hyperquadrique
    """
    for k, (a, b, c, g) in enumerate(lsParam):
        print(f"Terme {k: 2}: {a=:6.3f} {b=:6.3f} {c=:6.3f} gamma={g:6.3f}")

################################################################################

# probleme simplifie
def psi(x: float, y: float, a: float, b: float) -> float:
    """hyperquadrique du sous-probleme a optimiser
    
    Entree:
    - x, y: coordonnees du point en argument de l'hyperquadrique
    - a, b: parametres de l'hyperquadrique
    """
    return (a*x+b*y)**4 + (x+y)**4 - 1

def grad_psi(x: float, y: float, a: float, b: float) -> list[float]:
    """Gradient de l'hyperquadrique du sous-probleme a optimiser
    
    Entree:
    - x, y: coordonnees du point en argument de l'hyperquadrique
    - a, b: parametres de l'hyperquadrique
    """
    return [4*x*(a*x+b*y)**3, 4*y*(a*x+b*y)**3]

def fn_objectif(lsx: list[float], lsy: list[float], a: float, b: float) -> float:
    """Critere quadratique. Fonction objectif
    
    Entree:
    - x, y: coordonnees du point en argument de l'hyperquadrique
    - a, b: parametres de l'hyperquadrique
    """
    som = 0
    for u, v in zip(lsx, lsy):
        som += psi(u, v, a, b)**2
    return som

def grad_fn_objectif(x: float, y: float, a: float, b: float) -> list[float]:
    """Gradient du critere quadratique/fonction objectif.
    
    Entree:
    - x, y: coordonnees du point en argument de l'hyperquadrique
    - a, b: parametres de l'hyperquadrique
    """
    som = [0, 0]
    for u, v in zip(x, y):
        etape = 8*psi(u, v, a, b)*(a*u+b*v)**3
        som[0] += u*etape
        som[1] += v*etape
    return som

def hess_fn_objectif(x: float, y: float, a: float, b: float) -> list[list[float]]:
    """Hessienne du critere quadratique/fonction objectif.
    
    Entree:
    - x, y: coordonnees du point en argument de l'hyperquadrique
    - a, b: parametres de l'hyperquadrique

    Sortie:
    list[list[float]] hessienne du critere quadratique
    """
    som = [[0, 0], [0, 0]]
    for u, v in zip(x, y):
        etape1 = a*u + b*v
        etape2 = 8*etape1**2
        etape3 = etape2*(4*etape1**4 + 3*psi(u, v, a, b))
        som[0][0] += u*u*etape3
        som[0][1] += u*v*etape3
        som[1][0] += u*v*etape3
        som[1][1] += v*v*etape3
    return som

def descente_gradient(x: list[float], y: list[float], a0: float, b0: float,
                      alpha: float, nmax: int=50, eps: float=1e-6):
    """Descente de gradient
    
    Entree:
    - x, y: liste des coordonnees du nuage de points a fitter
    - a0, b0: parametres a optimiser avec la descente de gradient
    - alpha: pas d'apprentissage
    - nmax: nombre d'iterations de la descente de gradient
    - eps: condition d'arret sur la precision

    Sortie:
    - deux listes des parametres a0 puis b0 apres chaque iteration de descente
    - indice de derniere iteration (equivaut a nombre d'iterations - 1)
    - critere de convergence
    """
    convergence: bool=False
    lsA, lsB = np.full((nmax), np.nan), np.full((nmax), np.nan)

    lsA[0], lsB[0] = a0, b0
    for i in range(1, nmax):
        grd = grad_fn_objectif(x, y, lsA[i-1], lsB[i-1])

        lsA[i] = lsA[i-1] - alpha * grd[0]
        lsB[i] = lsB[i-1] - alpha * grd[1]

        if np.linalg.norm(grd) <= eps:
            convergence = True
            break

    return lsA, lsB, i, convergence

def newton(x: list[float], y: list[float], a0: float, b0: float,
           nmax: int=50, eps: float=1e-6):
    """Methode de Newton-Raphson
    
    Entree:
    - x, y: liste des coordonnees du nuage de points a fitter
    - a0, b0: parametres a optimiser avec la descente de gradient
    - nmax: nombre d'iterations de la descente de gradient
    - eps: condition d'arret sur la precision

    Sortie:
    - deux listes des parametres a0 puis b0 apres chaque iteration de descente
    - nombre d'iterations
    - critere de convergence
    """
    dX = float('inf')    # reel
    n = 0                # entier
    Xa, Xb = [a0], [b0]  # vecteurs de reels

    while dX > eps and n < nmax:
        d1J = np.array(grad_fn_objectif(x, y, Xa[-1], Xb[-1]))
        d2J = np.array(hess_fn_objectif(x, y, Xa[-1], Xb[-1]))

        DeltaX = np.tensordot(-np.linalg.inv(d2J), d1J, 1)
        Xa.append(Xa[-1]+DeltaX[0])
        Xb.append(Xb[-1]+DeltaX[1])
        dX = np.linalg.norm(DeltaX)

        n += 1

    return Xa, Xb, n, (dX <= eps)


if __name__ == '__main__':
    # Extraction de points
    with open('Data_HQ_Ph1et2.csv', 'r') as f:
        pts = [[float(u) for u in line.split(',')] for line in f.readlines()]

    # construction de la meshgrid
    a, b = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    a2D, b2D = np.meshgrid(a, b)

    ls_a, ls_b, nIter, converge = descente_gradient(*pts, .1, -.1, 0.004, 100)

    iso = 3*(np.logspace(0, 1)-1)

    plt.figure(num='gradient')
    plt.contour(a2D, b2D, fn_objectif(*pts, a2D, b2D), levels=iso)
    plt.plot(ls_a, ls_b, '-ro', lw=1.5, markersize=3)
    plt.plot(ls_a[nIter], ls_b[nIter], 'ko', markersize=3)

    plt.title("Méthode du gradient")
    plt.xlabel('a')
    plt.ylabel('b')
    plt.axis('square')

    iso = 3*(np.logspace(0, 1)-1)                         # valeurs des isovaleurs
    plt.figure(num='newton')                              # nom de figure
    plt.contour(a2D, b2D, fn_objectif(*pts, a2D, b2D), levels=iso)  # tracé des isovaleurs
    plt.colorbar()

    departs = np.random.rand(5, 2)*2-1
    for X in departs:
        ls_a, ls_b, nIter, converge = newton(*pts, X[0], X[1], 20)
        plt.plot(ls_a, ls_b, '-ro', lw=1.5, markersize=3)
        plt.plot(ls_a[-1], ls_b[-1], 'ko', markersize=3)
        if converge:  # on part du principe que converge est vraie au moins 1 fois
            a, b = ls_a[-1], ls_b[-1]

    plt.title("Méthode de Newton")
    plt.xlabel('a')
    plt.ylabel('b')
    plt.axis('square')

    plt.figure('hyperquad')
    plt.scatter(*pts)

    borne = 1.5
    u = np.linspace(-borne, borne)
    v = np.linspace(-borne, borne)
    u2D, v2D = np.meshgrid(u, v)
    plt.contour(u2D, v2D, psi(u2D, v2D, a, b), levels=[0], colors='k')

    plt.title('hyperquadrique ajustée au nuage de points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
