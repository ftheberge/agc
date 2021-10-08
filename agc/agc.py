import numpy as np
import scipy.stats as ss

## faster version w/o sample weights
## called from main function if needed
def agc_fast(y_true, y_score, pos_label=1, truncate=1, normalized=True):
    
    n = len(y_score)
    pos = sum(y_true)
    rk = n+1-ss.rankdata(y_score)
    if truncate <=1:
        T = int(np.round(truncate*n))
    else:
        T = int(truncate)

    x = np.where(y_true==pos_label)[0]
    r = [rk[i] for i in x if rk[i]<=T]

    A = sum([T-i+0.5 for i in list(r)])
    if T <= pos:
        M = T*T/2
    else:
        M = pos*pos/2 + (T-pos)*pos
    R = T*T*pos/(2*n)

    if normalized:
        return (A-R)/(M-R)
    else:
        return (A/M)


## Area under Gain Curve
def agc_score(y_true, y_score, pos_label=1, sample_weight=None, truncate=1, normalized=True):
    """
    Compute the area under the gain curve (agc) for binary labelled data

    Parameters
    ----------

    y_true: List or 1-dim array of binary labels

    y_score: List or 1-dim array of non-negative scores, same order and format as y_true

    pos_label: Value of the positive label in y_true, default=1, any other value is considered a negative label

    sample_weight: (optional) List or 1-dim array of weights for data points, same order and format as y_true

    truncate: float in range (0,1] or integer > 1
      if in range (0,1]: proportion of top-scoring points to consider
      if integer > 1: number of top-scoring points to consider
      if several points have the same score as the cutoff point, all such points are also considered

    normalized: boolean, default=True
      return normalized AGC
    
    Returns
    -------

    agc: float; area under the gain curve

    Notes
    -----

    Normalized AGC is given by: (AGC-R(AGC))/(M(AGC)-R(AGC)) where R(AGC) is the value under random scores, and M(AGC) is the maximum possible value.

    Example
    -------

    >>> labels = np.concatenate((np.repeat(1,100),np.repeat(0,900)))
    >>> scores = np.concatenate((np.random.uniform(.4,.8,100),np.random.uniform(.2,.6,900)))
    >>> agc_score(labels, scores)

    Reference
    ---------

    TBD
    """
    if sample_weight is None:
        return agc_fast(y_true=y_true, y_score=y_score, pos_label=pos_label, truncate=truncate, normalized=normalized)
    
    ## number of data points
    N = len(y_score)
    if len(y_true) != N:
        print("different number of scores and labels!")
        return None
           
    ## sort unique probabilities, decreasing
    U = -np.sort(-np.unique(y_score))

    ## build dictionaries, key = probability
    pos = dict()
    neg = dict()
    for i in U:
        pos[i]=0
        neg[i]=0

    ## first count each point with weight 1 
    for i in range(N):
        if y_true[i]==pos_label:
            pos[y_score[i]]+=1 
        else: 
            neg[y_score[i]]+=1
            
    ## total points for each proba and cdf
    u = [pos[i]+neg[i] for i in U]
    u_hat = np.cumsum(u)

    ## where to truncate to cover proportion 'truncate' or more of points
    if truncate > 1:
        truncate /= N
    trunc = np.argmax(u_hat >= u_hat[-1]*truncate)
    
    ## weighted - need to recompute cdf
    if sample_weight is not None:        
        for i in U:
            pos[i]=0
            neg[i]=0
        for i in range(N):
            if y_true[i]==pos_label:
                pos[y_score[i]]+=sample_weight[i] # w
            else: 
                neg[y_score[i]]+=sample_weight[i] # w
        u = [pos[i]+neg[i] for i in U]
        u_hat = np.cumsum(u)    

    ## total positives for each proba and cdf
    p = [pos[i] for i in U]
    p_hat = np.cumsum(p)

    ## compute A, R, M
    A = 0
    for i in range(trunc+1):
        A += u[i]*(p_hat[i]-p[i]/2)
    R = (u_hat[trunc]**2) * p_hat[-1]/(u_hat[-1]*2)
    if p_hat[-1] <= u_hat[trunc]:
        M = (p_hat[-1]**2)/2 + ((u_hat[trunc]-p_hat[-1])*p_hat[-1])
    else:
        M = (u_hat[trunc]**2)/2

    ## return normalized AGC or simple AGC
    if normalized:
        return (A-R)/(M-R)
    else:
        return A/M

    
## Area under Gain Curve
def gain_curve(y_true, y_score, pos_label=1, sample_weight=None, truncate=1):
    """
    Compute the proportion of data points and true positive rate for all thresholds, for plotting

    Parameters
    ----------

    y_true: List or 1-dim array of binary labels

    y_score: List or 1-dim array of non-negative scores, same order and format as y_true

    pos_label: Value of the positive label in y_true, default=1, any other value is considered a negative label

    sample_weight: (optional) List or 1-dim array of weights for data points, same order and format as y_true

    truncate: float in range (0,1] or integer > 1
      if in range (0,1]: proportion of top-scoring points to consider
      if integer > 1: number of top-scoring points to consider
      if several points have the same score as the cutoff point, all such points are also considered
    
    Returns
    -------

    top: proportion of (top scoring) points
    tpr: true positive rate for top scoring points
    thresh: corresponding score thresholds

    Example
    -------

    >>> labels = np.concatenate((np.repeat(1,100),np.repeat(0,900)))
    >>> scores = np.concatenate((np.random.uniform(.4,.8,100),np.random.uniform(.2,.6,900)))
    >>> top, tpr, _ = gain_curve(labels, scores)

    Reference
    ---------

    TBD
    """
    ## number of data points
    N = len(y_score)
    if len(y_true) != N:
        print("different number of scores and labels!")
        return None
           
    ## sort unique probabilities, decreasing
    U = -np.sort(-np.unique(y_score))

    ## build dictionaries, key = probability
    pos = dict()
    neg = dict()
    for i in U:
        pos[i]=0
        neg[i]=0

    ## first count each point with weight 1 
    for i in range(N):
        if y_true[i]==pos_label:
            pos[y_score[i]]+=1 
        else: 
            neg[y_score[i]]+=1
            
    ## total points for each proba and cdf
    u = [pos[i]+neg[i] for i in U]
    u_hat = np.cumsum(u)

    ## where to truncate to cover proportion 'truncate' or more of points
    if truncate > 1:
        truncate /= N
    trunc = np.argmax(u_hat >= u_hat[-1]*truncate)
    
    ## weighted - need to recompute cdf
    if sample_weight is not None:        
        for i in U:
            pos[i]=0
            neg[i]=0
        for i in range(N):
            if y_true[i]==pos_label:
                pos[y_score[i]]+=sample_weight[i] # w
            else: 
                neg[y_score[i]]+=sample_weight[i] # w
        u = [pos[i]+neg[i] for i in U]
        u_hat = np.cumsum(u)    

    ## total positives for each proba and cdf
    p = [pos[i] for i in U]
    p_hat = np.cumsum(p)

    return u_hat[:trunc]/max(u_hat), p_hat[:trunc]/max(p_hat), U

    
## Approximate AGC via sampling
def agc_approximate(y_true, y_score, pos_label=1, sample_weight=None, truncate=1, normalized=True, sample=1, quantiles=100, interpolation='linear'):

    """
    Approximate the area under the gain curve for binary labelled data via sampling

    Parameters
    ----------

    y_true: List or 1-dim array of binary labels

    y_score: List or 1-dim array of non-negative scores, same order and format as y_true

    pos_label: Value of the positive label in y_true, default=1, any other value is considered a negative label

    sample_weight: (optional) List or 1-dim array of weights for data points, same order and format as y_true

    truncate: float in range (0,1] or integer > 1
      if in range (0,1]: proportion of top-scoring points to consider
      if integer > 1: number of top-scoring points to consider
      if several points have the same score as the cutoff point, all such points are also considered

    normalized: boolean, default=True
      return normalized AGC

    sample: float in range (0,1]
      proportion of points to sample

    quantiles: int
      number of quantiles (bins) to use for the approximation

    interpolation: passed to the numpy.quantile() function
      some speed-up can be obtained with interpolation='nearest'

   
    Returns
    -------

    agc: float; approximated area under the gain curve

    Notes
    -----

    Normalized AGC is given by: (AGC-R(AGC))/(M(AGC)-R(AGC)) where R(AGC) is the value under random scores, and M(AGC) is the maximum possible value.

    For huge datasets, faster results (but coarser approximation) can be obtained by reducing both 'sample' and 'quantiles'

    Example
    -------

    >>> labels = np.concatenate((np.repeat(1,100),np.repeat(0,900)))
    >>> scores = np.concatenate((np.random.uniform(.4,.8,100),np.random.uniform(.2,.6,900)))
    >>> print('AGC:',agc_score(labels, scores),'approximation:',agc_approximate(labels,scores,sample=.5))

    Reference
    ---------

    TBD
    """

    ## number of data points
    N = len(y_score)
    if len(y_true) != N:
        print("different number of scores and labels!")
        return None

    ## sample and get weights
    ss = np.int(sample*N)
    S = np.random.choice(N,ss,replace=False)

    ## truncated quantiles
    if truncate > 1:
        truncate /= N
    n_q = quantiles
    eps = truncate/n_q
    q = np.arange(1-eps,1-truncate-eps/2,-eps)
    q[-1] = 1-truncate
        
    ## this is the bottleneck with large number of quantiles;
    ## 'nearest' is a bit faster than the default
    s = np.quantile([y_score[i] for i in S],q=q,interpolation=interpolation)    
    
    ## should have unique values, but in some cases, quantiles could be equal
    s = -np.unique(-s)
    n_q = len(s)
    ## rare but not impossible case:
    if n_q == 1:
        s = np.append(s,s[0]-.000001)
        n_q = 2    

    ## positive counts (p_i's and p_i^hat's)
    p = np.zeros(n_q+1) ## needed if we truncate
    u = np.zeros(n_q+1)
    x = np.digitize([y_score[i] for i in S],s)
    if sample_weight is not None:
        for i in range(len(S)):
            if y_true[S[i]] == pos_label:
                p[x[i]] += sample_weight[S[i]]
            u[x[i]] += sample_weight[S[i]]
    else:
        for i in range(len(S)):
            if y_true[S[i]] == pos_label:
                p[x[i]] += 1
            u[x[i]] += 1           
    p_hat = np.cumsum(p)

    ## sum of weights
    n_pos = sum(p)
    n_all = sum(u)
    
    ## try this
    truncate = sum([u[i] for i in range(n_q)])/n_all
        
    ## A, R, M
    A = 0
    for i in range(n_q):
        A+=(u[i]*(p_hat[i]-0.5*p[i]))

    if (truncate*n_all) <= n_pos:
        M = (truncate*n_all)**2/2
    else:
        M = n_pos**2/2 + (truncate*n_all-n_pos)*n_pos
    R = truncate**2*n_all*n_pos/2
        
    ## approximate AGC
    if normalized:
        return (A-R)/(M-R)
    else:
        return A/M
