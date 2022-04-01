import numpy as np
import pandas as pd
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

## fast version with weights
def agc_w(y_true, y_score, sample_weight=None, pos_label=1, truncate=1, normalized=True):
    
    n = len(y_score)
    rk = n+1-ss.rankdata(y_score)
    if truncate <=1:
        T = int(np.round(truncate*n))
    else:
        T = int(truncate)
        
    D = pd.DataFrame(rk,columns=['rank'])
    D['w'] = sample_weight
    D['label'] = y_true
    
    # bottleneck here
    pd.set_option('mode.chained_assignment',None)
    D['Wp'] = 0
    D['Wp'][D['label'] == 1] = D['w'][D['label'] == 1]
    D['Wn'] = 0
    D['Wn'][D['label'] != 1] = D['w'][D['label'] != 1]

    Wp_tot = np.sum(D['Wp'])
    Wn_tot = np.sum(D['Wn'])

    D = D.loc[D['rank'] <= T]

    D2 = pd.DataFrame({'Wp': D.groupby(['rank'])['Wp'].sum(), 
                   'Wn': D.groupby(['rank'])['Wn'].sum()})
    D2['W'] = np.cumsum(D2['Wp']+D2['Wn'])
    W_tot = D2.iloc[-1,-1]
    D2 = D2.loc[D2['Wp'] > 0]

    ## area
    A = np.sum(D2['Wp']*( (D2['Wp']+D2['Wn'])/2 + (W_tot - D2['W']) ))
    
    ## max
    if W_tot <= Wp_tot:
        M = W_tot*W_tot/2
    else:
        M = Wp_tot*Wp_tot/2 + Wp_tot*(W_tot-Wp_tot)    

    ## random
    R = W_tot * W_tot * Wp_tot / (2*(Wp_tot+Wn_tot))

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
      if several points have the same score, average rank is used

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
    else:
        return agc_w(y_true=y_true, y_score=y_score, sample_weight=sample_weight, pos_label=pos_label, truncate=truncate, normalized=normalized)

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
      if several points have the same score, average rank is used
    
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

    return u_hat[:trunc+1]/max(u_hat), p_hat[:trunc+1]/max(p_hat), U

