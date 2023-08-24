import numpy as np

def parallelAnalysis(flatFields, repetitions):
    stdEFF = np.std(flatFields, 1)
    keepTrack = np.zeros((repetitions, flatFields.shape[1]))
    stdMatrix = np.transpose([stdEFF] * flatFields.shape[1])

    for i in range(1, repetitions+1):
        print(f"Parallel Analysis: repetition {str(i)}")
        tmp = np.random.randn(flatFields.shape[0], flatFields.shape[1])
        sample = stdMatrix * tmp
        cov = np.cov(sample, bias=True, rowvar=False)
        D1, _ = np.linalg.eigh(cov)
        keepTrack[:][i-1] = D1

    mean_flat_fields_EFF = np.mean(flatFields, 1)
    F = flatFields - np.transpose([mean_flat_fields_EFF] * flatFields.shape[1])
    cov2 = np.cov(F, rowvar=False)
    D1, V1 = np.linalg.eigh(cov2)

    mean_keepTrack = np.mean(keepTrack, axis=0)
    std_keepTrack = np.std(keepTrack, axis=0, ddof=1)
    selection = D1 > mean_keepTrack + 2 * std_keepTrack
    numberPC = np.sum(selection)
    return V1, D1, numberPC
