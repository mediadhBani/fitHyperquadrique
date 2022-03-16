"""
Ce module contient les definitions des fonctions du probleme generalise de fit
de nuage de points par hyperquadrique.
"""

import matplotlib.pyplot as plt
import numpy as np

# alias de type d'un ensemble de parametres d'hyperquadrique
PrmHQ = np.ndarray  # shape (N, 4)

def hyperquadrique(x: float, y: float, lsParam: PrmHQ) -> float:
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

def droites_enveloppantes(lsParam: PrmHQ, ylim: tuple,
                          ax: plt.Axes=plt.gca()) -> list:
    """Trace les droites enveloppantes de l'hyperquadrique de parametres donnes.

    Entree:
    - lsParam: parametres de l'hyperquadrique
    - ylim: intervalle d'affichage des droites enveloppantes
    - ax: graphe cible dans lequel tracer les droites

    Sortie:
    - env: liste de droites enveloppantes sous forme de collections
        (classe ~matplotlib.collections.LineCollection)
    """
    lsLines = []
    for a, b, c, _ in lsParam:
        if b:
            lsLines.extend([ax.axline((0, (c+1)/-b), slope=-a/b, lw=1),
                            ax.axline((0, (c-1)/-b), slope=-a/b, lw=1)])
        else:
            lsLines.append(ax.vlines([-(c+1)/a, (1-c)/a], *ylim, lw=1))

    return lsLines

def decrire_hyperquadrique(lsParam: PrmHQ) -> None:
    """Description des paramètres d'une hyperquadrique.

    Entree:
    - lsParam: liste des parametres de l'hyperquadrique
    """
    for k, (a, b, c, g) in enumerate(lsParam):
        print(f"Terme {k} : {a=:+8.2e} {b=:+8.2e} {c=:+8.2e} gamma={g:+8.2e}")

def traceHQ(x2D, y2D, hq, clr='k'):
    """Tracé de l'hyperquadratique `hq` d'isovaleurdans le plan."""
    plt.contour(x2D, y2D, hyperquadrique(x2D, y2D, hq), levels=[0], colors=clr)


def signe(x: float):
    """Fonction signe. Renvoie -1 si l'argument est strictement negatif,
    1 sinon
    """
    return -1 if x < 0 else 1

################################################################################

def extraire_nuage(file: str) -> np.ndarray:
    """extrait un ensemble de points a partir d'un fichier.
    
    Entree:
    - file: nom du fichier dont le contenu se present ainsi: 

    ```
        x0, x1, x2, x3, ..., xn
        y0, y1, y2, y3, ..., yn
    ```

    Sortie:
    - un tableau de shape (2, N)
    """
    with open(file, 'r') as f:
        pts = [[u for u in lin.split(',')] for lin in f.readlines()]
    
    return np.array(pts, dtype=np.float32)

def couronne_nuage(arr: np.ndarray) -> tuple[float, float]:
    """Determine les rayons d'une couronne englobant l'ensemble d'un nuage de
    point.

    Entree:
    - un nuage de points de shape (2, N)

    Sortie:
    - smin: rayon interieur de la couronne
    - smax: rayon exterieur de la couronne
    """
    moy = np.mean(arr, axis=1)                # centre du nuage de points
    dst = np.linalg.norm(arr - moy[:, None])  # distances au centre

    return dst.min(), dst.max()  # min

nuage = np.random.randint(0, 10, (2, 5))
