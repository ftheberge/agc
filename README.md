# AGC - Area under Gain Curves
Python code to compute and plot (truncated, weighted) area under gain curves (**AGC**)

For binary classification, gain curves are a nice alternative to ROC curves in that they can naturally be **truncated** to focus on the top scoring points only.
Moreover, the data points can have **weights**. In this code, we provide three functions:

* `agc_score`: Compute the area under the gain curve (AGC) for binary labelled data
* `agc_approximate`: Approximate the area under the gain curve (AGC) for binary labelled data via sampling
* `gain_curve`: Compute the proportion of data points and true positive rate for all thresholds, for plotting

The first two functions return the **normalized area** by default (improvement over random, so this could be negative).
The functions can be imported from the supplied `agc.py` file, or installed via `pip install agc`.

## A simple example

```
## create toy binary labels and scores
labels = np.concatenate((np.repeat(1,100),np.repeat(0,900)))
scores = np.concatenate((np.random.uniform(.4,.8,100),np.random.uniform(.2,.6,900)))

## compute (normalized) area under the gain curve
print(agc_score(labels, scores))

## compute (un-normalized) area under the gain curve
print(agc_score(labels, scores, normalized=False))

## now the area for the top scoring 10% of the points
print(agc_score(labels, scores, truncate=0.1))
```

