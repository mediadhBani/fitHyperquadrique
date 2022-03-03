# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:07:26 2021

@author: Mohamed-iadh BANI
"""

# modules
import hyperquadratique as h  # contient les fonctions des phases 1 et 2
from initialise_coefHQ import initialise_coefHQ as initHQ
import matplotlib.pyplot as plt
import numpy as np

# CONSTANTES
EPS = 1e-6       # Seuil de précsion du fit
K1, K2 = 10, 10  # facteurs d'ajustement d'espace de recherche
NU = 1e8         # facteur de pénalité


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# D'abord les commandes simples.

def couronneNuage(x, y):
    """Couronne de rayons intérieure et extérieure englobant strictement le
    nuage des points d'abscisses `x` et `y`"""
    xc, yc = np.mean(x), np.mean(y)  # coordonnées du centre du nuage
    smin, smax = np.inf, 0           # rayons minimal et maximal

    for u, v in zip(x, y):
        dist = np.sqrt((u-xc)**2 + (v-yc)**2)  # distance par rapport au centre
        smin = min(smin, dist)
        smax = max(smax, dist)

    return smin, smax


def signe(x):
    """Fonction signe de `x`. Renvoie -1 si `x` strictement négatif, 1 sinon"""
    return -1 if x < 0 else 1


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# DERIVEES DU PARAMETRAGE DE L'HYPERQUADRATIQUE #

def Dphi(x, y, hq):
    """Dérivée du point image de l'hyperquadratique `hq` au point (`x`, `y`)"""
    grad = []
    for a, b, c, g in hq:
        sgn = signe(a*x + b*y + c)**g  # signe de la dérivée
        etape = sgn*g * abs(a*x + b*y + c)**(g-1)
        grad.extend([x*etape, y*etape, etape])

    return np.array(grad)


def Hphi(x, y, hq):
    """Hessienne du point image de l'hyperquadratique `hq` au point (`x`, `y`)
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# FONCTION Fio ET SES DERIVEES #

def fio(x, y, hq):
    """Fonction inside-outside au de l'hyperquadratique `hq` au point (`x`,`y`)
    """
    return np.sqrt(np.sqrt(h.imageHQ(x, y, hq)))


def Dfio(x, y, hq):
    """Dérivée de la fonction inside-outside au de l'hyperquadratique `hq` au
    point (`x`,`y`)"""
    return Dphi(x, y, hq) / h.imageHQ(x, y, hq)**.75 / 4


def Hfio(x, y, hq):
    """Hessienne de la fonction inside-outside de l'hyperquadratique `hq` au
    point (`x`,`y`)"""
    f1 = -3 / 16 / h.imageHQ(x, y, hq)**1.75
    f2 = h.imageHQ(x, y, hq)**-.75 / 4

    Dp, Hp = Dphi(x, y, hq), Hphi(x, y, hq)

    return f1*np.tensordot(Dp, Dp, 0) + f2*Hp


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# FONCTION EOF1 ET SES DERIVEES #

def eof1(x, y, hq):
    """Fonction erreur de fit biaisée."""
    som = 0
    for u, v in zip(x, y):
        som += (1-fio(u, v, hq))**2
    return som/2


def Deof(x, y, hq):
    """Gradient de l'erreur de fit biaisée."""
    grad = np.zeros(len(hq)*3)
    for u, v in zip(x, y):
        grad = grad + 2*(1-fio(u, v, hq))*Dfio(u, v, hq)

    return grad


def Heof(x, y, hq):
    """Hessienne de l'erreur de fit biaisée."""
    cH = len(hq)*3  # côté de la matrice hessienne
    hess = np.zeros((cH, cH))
    for u, v in zip(x, y):
        f, Df, Hf = fio(u, v, hq), Dfio(u, v, hq), Hfio(u, v, hq)
        hess = hess + 2*np.tensordot(Df, Df, 0) + 2*(1-f)*Hf

    return hess


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# FONCTION PENALITE ET SES DERIVEES #

def pen(hq):
    """Pénalité dans la fonction objectif si le domaine de l'hyperquadtraique
    définie par ses droites enveloppantes est trop large."""
    # profitons de la portée des variables mu1 et mu2 pour ne pas avoir à les
    # mettre en argument de la fonction pénalité.
    penalite = 0
    for a, b, _, _ in hq:
        etape = a**2 + b**2
        penalite += max(0, mu1-etape)**2 + max(0, etape-mu2)**2
    return penalite


def Dpen(hq):
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


def Hpen(hq):
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

def J(x, y, hq, nu=NU):
    """Fonction objectif pénalisée à minimiser."""
    return eof1(x, y, hq) + nu*pen(hq)


# ALGORITHME DE LEVENBERG-MARQUARDT #

def levmar(x, y, lam0, nmax, eps=EPS, nu=NU):
    """Algorithme de Levenberg-Marquardt. Renvoie une hyperquadratique
    optimisée et un signal d'erreur."""
    lng = len(lam0)  # nombre de termes dans l'hyperquadratique
    cH = lng*3       # côté de la hessienne

    bet, n, dLam = .01, 0, np.inf     # réels
    lam = np.array(lam0.copy())      # copie de l'hyperquadratique, matrice

    while dLam > eps and n < nmax:
        DJ = Deof(x, y, lam) + nu*Dpen(lam)
        HJ = Heof(x, y, lam) + nu*Hpen(lam)

        while True:
            inv = np.linalg.inv(HJ + bet*np.eye(cH))

            # Parfois l'inversion de la matrice échoue, sûrement pour des
            # raisons de singularité, mais il reste intéressant de récupérer la
            # dernière itération de lam
            if np.any(np.isnan(inv)):
                print('######################################################')
                print('Erreur : dernier paramétrage hyperquadratique renvoyé.')
                print('######################################################')
                return lam, True

            # DLam = np.tensordot(inv, -DJ, 1)
            DLam = inv.dot(-DJ)

            # On réorganise `DLam` pour pouvoir directement l'ajouter à `lam`
            temp = np.zeros((lng, 4))
            temp[:lng, :3] = DLam.reshape((lng, 3))
            temp = temp + lam

            if J(x, y, temp) >= J(x, y, lam):
                bet *= 10
            else:
                break

        bet /= 10
        lam = temp.copy()
        dLam = np.linalg.norm(DLam)
        n += 1

    return lam, False


def schemaGeneral(x, y, lam0, nmax=5, eps=EPS):
    lam = lam0.copy()
    n = 0
    err = False  # signal d'erreur

    while eof1(x, y, lam) > eps and n < 10 and not err:
        # h.traceHQ(x2D, y2D, lam)
        lam, err = levmar(x, y, lam, nmax)
        # print(n, lam, sep='\n')
        n += 1

    return lam, n, (eof1(x, y, lam) < eps)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# NOM DE FICHIER ICI
# extraction du nuage de points
with open('truc.txt', 'r') as f:
    nuage = [[float(u) for u in ligne.split(',')] for ligne in f.readlines()]

# tensorisation numpy
xn, yn = np.array(nuage)

# rayons de la couronne du nuage de points
smin, smax = couronneNuage(xn, yn)

# variables mu intervenant dans la contrainte du domaine de recherche
mu1, mu2 = 4/(K1*smax)**2, 4/(K2*smin)**2

# NOMBRE DE TERMES ICI
# initialisation du fit
parametresHQ = initHQ(xn, yn, 3)  # nombre de termes arbitraire (>2)

# affichage des paramètres de l'hyperquadratique
h.lireHQ(parametresHQ)
plt.close('all')
x = np.linspace(min(xn)-10, max(xn)+10)
y = np.linspace(min(yn)-10, max(yn)+10)
x2D, y2D = np.meshgrid(x, y)
neo, n, cvrg = schemaGeneral(xn, yn, parametresHQ, 3)
plt.scatter(xn, yn)

h.traceHQ(x2D, y2D, parametresHQ)
h.traceHQ(x2D, y2D, neo, 'r')

plt.title(cvrg)
plt.axis('square')
h.droiteEnvHQ(neo, plt.ylim())
plt.show()
