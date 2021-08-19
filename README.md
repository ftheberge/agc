# AGC - Area under Gain Curves
Python code to compute and plot (truncated, weighted) area under gain curves (**AGC**)

For binary classification, gain curves are a nice alternative to ROC curves in that they can naturally be truncated to focus on the top scoring points only.
In this code, we provide three functions:

* `agc_score`: Compute the area under the gain curve (AGC) for binary labelled data
* `agc_approximate`: Approximate the area under the gain curve (AGC) for binary labelled data via sampling
* `gain_curve`: Compute the proportion of data points and true positive rate for all thresholds, for plotting




