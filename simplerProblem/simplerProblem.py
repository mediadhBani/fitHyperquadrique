import logging
# from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import sys

# Create a custom logger
logger = logging.getLogger(__file__)
logging.basicConfig(format='%(asctime)s %(levelname)-8s -- %(message)s',
                    level=logging.INFO,
                    datefmt='%y%m%d-%H%M%S')

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

def droites_enveloppantes(lsParam: list[list[float]], ylim: tuple,
                          ax: plt.gca()) -> list:

    """Trace les droites enveloppantes de l'hyperquadrique de parametres donnes.

    Entree:
    - lsParam: liste des parametres de l'hyperquadrique
    - ylim: intervalle d'affichage des droites enveloppantes
    - ax: graphe cible dans lequel tracer les droites

    Sortie:
    - env: liste de droites enveloppantes sous forme de collections
        (classe ~matplotlib.collections.LineCollection)
    """

    for a, b, c, _ in lsParam:
        if b:
            lsLines = [ax.axline((0, (c+1)/-b), slope=-a/b, lw=1),
                       ax.axline((0, (c-1)/-b), slope=-a/b, lw=1)]
        else:
            lsLines = [ax.vlines([-(c+1)/a, (1-c)/a], *ylim, lw=1)]

    return lsLines

def decrire_hyperquadrique(lsParam: list[list[float]]) -> None:
    """Description des paramètres d'une hyperquadrique.

    Entree:
    - lsParam: liste des parametres de l'hyperquadrique
    """
    for k, (a, b, c, g) in enumerate(lsParam):
        print(f"Terme {k} : {a=:+8.2e} {b=:+8.2e} {c=:+8.2e} gamma={g:+8.2e}")

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

        if np.linalg.norm(grd) < eps:
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
    convergence: bool=False
    lsA, lsB = np.full((nmax), np.nan), np.full((nmax), np.nan)

    lsA[0], lsB[0] = a0, b0
    for i in range(1, nmax):
        grd = np.array(grad_fn_objectif(x, y, lsA[i-1], lsB[i-1]))
        hss = np.array(hess_fn_objectif(x, y, lsA[i-1], lsB[i-1]))

        delta = np.tensordot(-np.linalg.inv(hss), grd, 1)

        lsA[i] = lsA[i-1] + delta[0]
        lsB[i] = lsB[i-1] + delta[1]

        if np.linalg.norm(delta) < eps:
            convergence = True
            break

    return lsA, lsB, i, convergence

if __name__ == '__main__':
    # Extraction de points
    with open('Data_HQ_Ph1et2.csv', 'r') as f:
        pts = [[float(u) for u in line.split(',')] for line in f.readlines()]

    # construction des meshgrids des espaces (a, b) puis (x, y)
    meshAB = np.mgrid[-1:1:100j, -1:1:100j]
    meshXY = np.mgrid[-1.5:1.5:100j, -1.5:1.5:100j]

    # isovaleurs de l'espace (a, b)
    lsIso = 3*(np.logspace(0, 1)-1)

    # bornes du graphes dans l'espace (x, y)
    bornesHq = (-1.5, 1.5)

    # comparaison des deux methodes
    lsArg = [{'a0': 0.1, 'b0': -0.1, 'alpha': 4e-3, 'nmax': 100},
              {'a0': np.random.random(), 'b0': np.random.random()}]
    lsFct = [descente_gradient, newton]
    lsFname = ["descenteGradient", "methodeNewton"]
    lsTitle = ["Descente de gradient", "Méthode de Newton"]

################################################################################

    # Affichage etat final #

    # for fct, arg, ttl in zip(lsFct, lsArg, lsTitle):
    #     lsA, lsB, idx, cvg = fct(*pts, **arg)

    #     fig, ax = plt.subplots(1, 2, num=ttl + " fin")
    #     plt.suptitle(ttl)

    #     # representation de la methode courante dans l'espace (a, b)
    #     ax[0].contour(*meshAB, fn_objectif(*pts, *meshAB), levels=lsIso)
    #     ax[0].plot(lsA, lsB, '-ro', lw=1, markersize=2, label="Itérations")
    #     ax[0].plot(lsA[idx], lsB[idx], 'ko', markersize=5, label="Point final")

    #     ax[0].axis('square')
    #     ax[0].legend()
    #     ax[0].set_title("Espace des paramètres")
    #     ax[0].set_xlabel("a")
    #     ax[0].set_ylabel("b")

    #     # representation de la methode courante dans l'espace (x, y)
    #     cnt = ax[1].contour(*meshXY, psi(*meshXY, lsA[idx], lsB[idx]),
    #                         levels=[0], colors='r')
    #     ax[1].scatter(*pts, label="Données")

    #     ax[1].axis('square')
    #     ax[1].legend()
    #     ax[1].set_title("Espace des données")
    #     ax[1].set_xlabel("x")
    #     ax[1].set_ylabel("y")

################################################################################

    # Animation des iteration #
    
    # def animate(i):
    #     """Animation de la méthode de fit.

    #     Entree:
    #     - i: indice d'itération
    #     """
    #     global cnt, mth

    #     # rafraichissement de l'hyperquadrique
    #     for e in cnt.collections:
    #         e.remove()

    #     # trace des iteration de la methode courante
    #     mth = ax[0].plot(lsA[i-1:i+1], lsB[i-1:i+1], '-ro', lw=1, markersize=2,
    #                         label="Itérations")

    #     # trace de l'hyperquadrique
    #     cnt = ax[1].contour(*meshXY, psi(*meshXY, lsA[i], lsB[i]),
    #                         levels=[0], colors='r')

    # for fct, arg, ttl, fnm in zip(lsFct, lsArg, lsTitle, lsFname):
    #     logging.info(ttl + " en cours...")

    #     lsA, lsB, idx, cvg = fct(*pts, **arg)

    #     fig, ax = plt.subplots(1, 2, num=ttl)
    #     plt.suptitle(ttl)

    #     # representation de la methode courante dans l'espace (a, b)
    #     ax[0].contour(*meshAB, fn_objectif(*pts, *meshAB), levels=lsIso)
    #     mth = ax[0].plot(lsA[0], lsB[0], '-ro', lw=1, markersize=2,
    #                      label="Itérations")

    #     ax[0].axis('square')
    #     ax[0].legend()
    #     ax[0].set_title("Espace des paramètres")
    #     ax[0].set_xlabel("a")
    #     ax[0].set_ylabel("b")

    #     # representation de la methode courante dans l'espace (x, y)
    #     ax[1].scatter(*pts, label="Données")
    #     cnt = ax[1].contour(*meshXY, psi(*meshXY, lsA[idx], lsB[idx]),
    #                         levels=[0], colors='r')

    #     ax[1].autoscale(False)
    #     ax[1].axis('square')
    #     ax[1].legend()
    #     ax[1].set_title("Espace des données")
    #     ax[1].set_xlabel("x")
    #     ax[1].set_ylabel("y")

    #     fig.tight_layout()

    #     anim = animation.FuncAnimation(fig, animate, frames=idx)
    #     anim.save(fnm + ".gif", writer='imagemagick', fps=15)
