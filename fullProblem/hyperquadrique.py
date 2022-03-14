import matplotlib.pyplot as plt

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

def droites_enveloppantes(lsParam: list[list[float]], ylim: tuple) -> None:
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

def traceHQ(x2D, y2D, hq, clr='k'):
    """Tracé de l'hyperquadratique `hq` d'isovaleurdans le plan."""
    plt.contour(x2D, y2D, hyperquadrique(x2D, y2D, hq), levels=[0], colors=clr)
