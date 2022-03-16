import helper as h
from initialise_coefHQ import initialise_coefHQ
import logging
import matplotlib.pyplot as plt
import numpy as np

# Create a custom logger
logger = logging.getLogger(__file__)
logging.basicConfig(format='%(asctime)s %(levelname)-8s -- %(message)s',
                    level=logging.INFO,
                    datefmt='%y%m%d-%H%M%S')

# CONSTANTES
K1, K2 = 10, 10  # facteurs d'ajustement d'espace de recherche

def signe(x: float) -> int:
    """Fonction signe. Renvoie -1 si l'argument est strictement negatif,
    1 sinon.
    """
    return -1 if x < 0 else 1

# DERIVEES DU PARAMETRAGE DE L'HYPERQUADRIQUE #

def grad_hyperquadrique(x, y, hq: h.PrmHQ):
    """Dérivée du point image de l'hyperquadrique `hq` au point (`x`, `y`)"""
    grad = []
    for a, b, c, g in hq:
        sgn = signe(a*x + b*y + c)**g  # signe de la dérivée
        etape = sgn*g * abs(a*x + b*y + c)**(g-1)
        grad.extend([x*etape, y*etape, etape])

    return np.array(grad)

def hess_hyperquadrique(x, y, hq):
    """Hessienne du point image de l'hyperquadrique `hq` au point (`x`, `y`)
    """
    cH = len(hq)*3  # côté de la matrice hessienne
    H = np.zeros((cH, cH))

    # produit tensoriel de X = (x, y, 1) par lui-même
    XX = np.tensordot([x, y, 1], [x, y, 1], 0)

    for k, (a, b, c, g) in zip(range(0, cH, 3), hq):
        sgn = signe(a*x + b*y + c)  # sgn**n = sgn quand n impair
        etape = sgn*g*(g-1) * abs(a*x + b*y + c)**(g-2)

        H[k:k+3, k:k+3] = XX*etape

    return H  # matrice multi-diagonale

################################################################################

def fio(x, y, hq):
    """Fonction inside-outside au de l'hyperquadrique `hq` au point (`x`,`y`)
    """
    return np.sqrt(np.sqrt(h.hyperquadrique(x, y, hq)))

def grad_fio(x, y, hq):
    """Dérivée de la fonction inside-outside au de l'hyperquadrique `hq` au
    point (`x`,`y`)"""
    return grad_hyperquadrique(x, y, hq) / h.hyperquadrique(x, y, hq)**.75 / 4

def hess_fio(x, y, hq):
    """Hessienne de la fonction inside-outside de l'hyperquadrique `hq` au
    point (`x`,`y`)"""
    f1 = -3 / 16 / h.hyperquadrique(x, y, hq)**1.75
    f2 = h.hyperquadrique(x, y, hq)**-.75 / 4

    grd, hss = grad_hyperquadrique(x, y, hq), hess_hyperquadrique(x, y, hq)

    return f1*np.tensordot(grd, grd, 0) + f2*hss

def eof1(x, y, hq):
    """Fonction erreur de fit biaisée."""
    som = 0
    for u, v in zip(x, y):
        som += (1-fio(u, v, hq))**2
    return som/2

def grad_eof1(x, y, hq):
    """Gradient de l'erreur de fit biaisée."""
    grad = np.zeros(len(hq)*3)
    for u, v in zip(x, y):
        grad = grad + 2*(1-fio(u, v, hq))*grad_fio(u, v, hq)

    return grad

def hess_eof1(x, y, hq):
    """Hessienne de l'erreur de fit biaisée."""
    cH = len(hq)*3  # côté de la matrice hessienne
    hess = np.zeros((cH, cH))
    for u, v in zip(x, y):
        f, Df, Hf = fio(u, v, hq), grad_fio(u, v, hq), hess_fio(u, v, hq)
        hess = hess + 2*np.tensordot(Df, Df, 0) + 2*(1-f)*Hf

    return hess

################################################################################

# FONCTION PENALITE ET SES DERIVEES #

def penalite(hq):
    """Pénalité dans la fonction objectif si le domaine de l'hyperquadtraique
    définie par ses droites enveloppantes est trop large."""
    # profitons de la portée des variables mu1 et mu2 pour ne pas avoir à les
    # mettre en argument de la fonction pénalité.
    penalite = 0
    for a, b, _, _ in hq:
        etape = a**2 + b**2
        penalite += max(0, mu1-etape)**2 + max(0, etape-mu2)**2
    return penalite

def grad_penalite(hq: list):
    """Gradient de la pénalité dans la fonction objectif si le domaine de
    l'hyperquadtraique définie par ses droites enveloppantes est trop large."""
    # profitons de la portée des variables mu1 et mu2 pour ne pas avoir à les
    # mettre en argument de la fonction pénalité. Clarté.
    grad = []
    for a, b, *_ in hq:  # c n'intervient pas, on l'annule.
        # lambda squared, lambda étant la paramétrisation d'un terme de l'hq
        lsq = a**2 + b**2
        etape = 2*(max(0, lsq-mu2) - max(0, mu1-lsq))
        grad.extend([a*etape, b*etape, 0])

    return np.array(grad)

def hess_penalite(hq):
    """Hessienne de la pénalité dans la fonction objectif si le domaine de
    l'hyperquadtraique définie par ses droites enveloppantes est trop large."""
    cH = len(hq)*3  # côté de la matrice hessienne
    H = np.zeros((cH, cH))

    for k, (a, b, *_) in zip(range(0, cH, 3), hq):
        LL = np.tensordot([a, b, 0], [a, b, 0], 0)
        lsq = a**2 + b**2
        etape = 2*(max(0, lsq-mu2) - max(0, mu1-lsq))

        H[k:k+3, k:k+3] = np.diag([1, 1, 0])*etape
        H[k:k+3, k:k+3] = H[k:k+3, k:k+3] + LL*4*((lsq > mu2) + (mu1 > lsq))

    return H

################################################################################

def objectif(x, y, hq, nu=1e-8):
    """Fonction objectif pénalisée à minimiser."""
    return eof1(x, y, hq) + nu*penalite(hq)

def levenberg_marquardt(x, y, lam0, nmax: int=50, eps: float=1e-6, nu: float=1e-8):
    """Algorithme de Levenberg-Marquardt. Renvoie une hyperquadrique
    optimisée et un signal d'erreur."""
    lng = len(lam0)  # nombre de termes dans l'hyperquadrique
    cH = lng*3       # côté de la hessienne

    bet, n, dLam = .01, 0, np.inf    # réels
    lam = np.array(lam0.copy())      # copie de l'hyperquadrique, matrice

    # for i in range(nmax):
    #     pass

    while dLam > eps and n < nmax:
        grd = grad_eof1(x, y, lam) + nu*grad_penalite(lam)
        hss = hess_eof1(x, y, lam) + nu*hess_penalite(lam)

        while True:
            inv = np.linalg.inv(hss + bet * np.eye(cH))

            # Parfois l'inversion de la matrice échoue, surement a cause d'un
            # mauvais conditionnement, mais il reste intéressant de récupérer la
            # dernière itération de lam
            if np.any(np.isnan(inv)):
                logging.warning("NaN détecté => Matrice mal conditionnée.\n\
                    dernier paramétrage hyperquadrique renvoyé.")
                return lam, True

            # DLam = np.tensordot(inv, -DJ, 1)
            DLam = inv.dot(-grd)

            # On réorganise `DLam` pour pouvoir directement l'ajouter à `lam`
            temp = np.zeros((lng, 4))
            temp[:lng, :3] = DLam.reshape((lng, 3))
            temp = temp + lam

            if objectif(x, y, temp) >= objectif(x, y, lam):
                bet *= 10
            else:
                break

        bet /= 10
        lam = temp.copy()
        dLam = np.linalg.norm(DLam)
        n += 1

    return lam, False

def schema_general(x, y, lam0, nmax=5, eps=1e-6):
    lam = lam0.copy()
    n = 0
    err = False  # signal d'erreur

    while eof1(x, y, lam) > eps and n < 10 and not err:
        # h.traceHQ(x2D, y2D, lam)
        lam, err = levenberg_marquardt(x, y, lam, nmax)
        # print(n, lam, sep='\n')
        n += 1

    return lam, n, (eof1(x, y, lam) < eps)

if __name__ == '__main__':
    # NOM DE FICHIER ICI
    
    nuage = h.extraire_nuage('hq1b1.csv')  # extraction nuage de points
    smin, smax = h.couronne_nuage(*nuage)  # rayons couronne nuage de points

    # variables mu intervenant dans la contrainte du domaine de recherche
    mu1, mu2 = 4/(K1*smax)**2, 4/(K2*smin)**2

    # nombre de termes arbitraire (>2)
    parametresHQ: h.PrmHQ=initialise_coefHQ(*nuage, 3)

    # affichage des paramètres de l'hyperquadrique
    h.decrire_hyperquadrique(parametresHQ)

    exit()
    mesh = np.meshgrid(x, y)
    neo, n, cvrg = schema_general(*nuage, parametresHQ, 3)

    plt.scatter(nuage)
    h.traceHQ(*mesh, parametresHQ)
    h.traceHQ(*mesh, neo, 'r')

    plt.title(cvrg)
    plt.axis('square')
    h.droites_enveloppantes(neo, plt.ylim())
    plt.show()
