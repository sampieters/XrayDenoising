import numpy as np
import scipy as sp


def parallelAnalysis(flatFields, repetitions):
    stdEFF = np.std(flatFields, 1, ddof=1)
    keepTrack = np.zeros((repetitions, flatFields.shape[1]))
    stdMatrix = np.transpose([stdEFF] * flatFields.shape[1])
    for i in range(1, repetitions+1):
        print(f"Parallel Analysis: repetition {str(i)}")
        tmp = np.random.randn(flatFields.shape[0], flatFields.shape[1])
        sample = np.multiply(stdMatrix, tmp)
        cov = np.cov(sample, rowvar=False)
        # TODO: np.linalg.eig() returns different matrix of eigenvalues but this is actually the same??
        D1, dm = np.linalg.eigh(cov)
        keepTrack[:][i-1] = D1
    keepTrack = keepTrack.transpose()

    mean_flat_fields_EFF = np.mean(flatFields, 1)
    F = flatFields - np.transpose([mean_flat_fields_EFF] * flatFields.shape[1])
    cov2 = np.cov(F, rowvar=False)
    #D1, V1 = np.linalg.eigh(cov2)
    D1, V1 = sp.linalg.eigh(cov2)

    selection = np.zeros((1, flatFields.shape[1]))
    # selection(D1>(tmp3)) = 1
    # --------------------------------------------------------------
    tmp3 = np.mean(keepTrack, 1) + 2 * np.std(keepTrack, 1, ddof=1)
    for i in range(0, D1.shape[0]):
        if D1[i] > tmp3[i]:
            selection[0][i] = 1
    # --------------------------------------------------------------
    numberPC = np.sum(selection).astype(int)
    return V1, D1, numberPC
