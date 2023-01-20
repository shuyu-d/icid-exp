## ---- Standardization -----
def standardize(B, X, varN=None):
    # B:        Weighted adjacency matrix of the causal structure
    # varN:     Diagonal of noise variances
    d = B.shape[0]
    if varN == None:
        varN = np.diag(np.eye(d)) # EV case
    elif varN == 'const':
        varN = abs(np.random.normal(scale=1.2))*np.diag(np.eye(d))
    else:
        varN = abs(np.random.normal(scale=1.1, size=[d]))
    CovX = ( (np.linalg.inv(np.eye(d)-B)).T @ np.diag(varN) ) @ np.linalg.inv(np.eye(d)-B)
    ThetaX = ((np.eye(d)-B) @ np.diag(1/varN)) @ (np.eye(d)-B).T
    D = np.sqrt(np.diag(CovX))  # standard deviations of Xi's
    # Standardization of X is equivalent to the following transformation on InvCov:
    ThetaX_st =  (np.diag(D)@ ThetaX) @ np.diag(D)
    # Standardization of X
    #   made sure that X is of size nxd
    X_st = X @ np.diag(1/D)
    return ThetaX_st, X_st, ThetaX

