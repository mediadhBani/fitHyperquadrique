# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 22:17:22 2021

@author: Mohamed-iadh BANI
"""

import matplotlib.pyplot as plt


def droiteEnvHQ(hq, limy):
    """Tracé des droites enveloppantes de l'hyperquadratique `hq` dans le
    domaine limité en ordonnée par l'intervalle `limy` (couple)"""
    for a, b, c, _ in hq:
        if b:
            plt.axline((0, (c+1)/-b), slope=-a/b, lw=1.1)
            plt.axline((0, (c-1)/-b), slope=-a/b, lw=1.1)
        else:
            plt.vlines([-(c+1)/a, (1-c)/a], *limy, lw=1.1)


def imageHQ(x, y, hq):
    """Calcule le point image de l'hyperquadratique `hq` au point (`x`, `y`)"""
    som = 0
    for a, b, c, g in hq:
        som += abs(a*x + b*y + c)**g
    return som


def lireHQ(hq):
    """Description des coefficients de l'hyperquadratique `hq`"""
    for (k, (a, b, c, g)) in enumerate(hq):
        print('Terme %2d : a = %6.3f ; b = %6.3f ; c = %6.3f ; gamma = %2d' %
              (k+1, a, b, c, g))


def traceHQ(x2D, y2D, hq, clr='k'):
    """Tracé de l'hyperquadratique `hq` d'isovaleurdans le plan."""
    plt.contour(x2D, y2D, imageHQ(x2D, y2D, hq), levels=[1], colors=clr)
