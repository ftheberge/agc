# gain-curves
Python code to compute and plot (truncated, weighted) area under gain curves

For binary classification, gain curves are a nice alternative to ROC curves in that they can naturally be truncated to focus on the top scoring points only.
In this code, we provide three functions:

* `gain_agc_score`: Compute the area under the gain curve (agc) for binary labelled data
* `gain_agc_approximate`: Approximate the area under the gain curve for binary labelled data via sampling
* `gain_curve`: Compute the proportion of data points and true positive rate for all thresholds, for plotting




