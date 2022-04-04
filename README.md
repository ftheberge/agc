# AGC - Area under Gain Curves
Python code to compute and plot (truncated, weighted) area under gain curves (**AGC**)

For binary classification, gain curves are a nice alternative to ROC curves in that they can naturally be **truncated** to focus on the top scoring points only.
Moreover, the data points can have **weights**. In this code, we provide two functions:

* `agc_score`: Compute the area under the gain curve (AGC) for binary labelled data
* `gain_curve`: Compute the proportion of data points and true positive rate for all thresholds, for plotting

The first function returns the **normalized area** by default (improvement over random, so this could be negative).
The functions can be imported from the supplied `agc.py` file, or installed via `pip install agc`.

## A simple example

```
## create toy binary labels and scores for illustration
labels = np.concatenate((np.repeat(1,100),np.repeat(0,900)))
scores = np.concatenate((np.random.uniform(.4,.8,100),np.random.uniform(.2,.6,900)))

## compute (normalized) area under the gain curve
print(agc_score(labels, scores))

## compute (un-normalized) area under the gain curve
print(agc_score(labels, scores, normalized=False))

## now the area for the top scoring 10% of the points
print(agc_score(labels, scores, truncate=0.1))

## or top scoring 100 points
print(agc_score(labels, scores, truncate=100))
```

## More details in Notebooks:

For a quick introduction, see the following notebook:
https://github.com/ftheberge/agc/blob/main/agc/agc_intro.ipynb
also available in markup format:
https://github.com/ftheberge/agc/blob/main/intro/agc_intro.md

For more details, see the notebook:
https://github.com/ftheberge/agc/blob/main/agc/agc.ipynb
also available in markup format:
https://github.com/ftheberge/agc/blob/main/example/agc.md
