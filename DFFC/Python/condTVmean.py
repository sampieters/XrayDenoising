import imresize
import numpy as np
from scipy import optimize

def fun(projections, meanFF, FF, DF, x):
    FF_eff = np.zeros((FF.shape[1], FF.shape[2]))
    for i in range(0, FF.shape[0]):
        FF_eff = FF_eff + x[i] * FF[:][:][i]

    logCorProj = np.divide((projections - DF), (meanFF + FF_eff)) * np.mean(meanFF[:] + FF_eff[:])
    Gy, Gx = np.gradient(logCorProj)
    mag = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))
    cost = np.sum(mag[:])
    return cost

def condTVmean(projections, meanFF, FF, DF, x, DS):
    projections = tryout.imresize(projections, 1/DS)
    meanFF = tryout.imresize(meanFF, 1/DS)
    FF2 = np.zeros((FF.shape[0], meanFF.shape[0], meanFF.shape[1]))
    for i in range(0, FF.shape[0]):
        h = tryout.imresize(FF[:][:][i], 1/DS)
        FF2[:][:][i] = tryout.imresize(FF[:][:][i], 1/DS)
    FF = FF2
    DF = tryout.imresize(DF, 1/DS)
    func = lambda X: fun(projections, meanFF, FF, DF, X)
    xNew = optimize.minimize(func, x)
    return xNew


