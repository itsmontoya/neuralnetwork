# Classification Metrics

The `metric` package reports model behavior without affecting optimization.
Classification metrics expect prediction and target matrices with matching
shapes, where each row is one sample.

## Binary Metrics

Binary metrics use `[batchSize, 1]` predictions and targets. Targets must be
encoded as `0` or `1`.

The zero value of each binary metric uses a threshold of `0.5`. Constructors such
as `NewBinaryAccuracy`, `NewBinaryPrecision`, `NewBinaryRecall`, and
`NewBinaryF1` accept a custom threshold.

`NewBinaryConfusionMatrix` also uses the default threshold of `0.5`.
`NewBinaryConfusionMatrixWithThreshold` accepts a custom threshold.

Thresholds must be finite. Any finite `float32` threshold is valid, including
values outside `[0, 1]`. The threshold is a numeric decision boundary rather than
a probability-range validator.

Predictions greater than or equal to the configured threshold map to class `1`.
Predictions below the threshold map to class `0`.

`BinaryPrecision`, `BinaryRecall`, and `BinaryF1` report the positive class
(`1`).

## Categorical Metrics

Categorical metrics use one-hot targets. Each target row must contain exactly one
`1`, and all other values in the row must be `0`.

Predicted classes are selected with the argmax of each prediction row. When a row
contains tied maximum prediction values, the first maximum column is selected.

Macro categorical metrics compute the unweighted mean across classes.

## Confusion Matrices

`ConfusionMatrix` counts use target classes as rows and predicted classes as
columns.

For example, `At(targetClass, predictedClass)` returns the count for samples
whose true class is `targetClass` and whose predicted class is `predictedClass`.
`Counts` returns a copy using the same target-row by predicted-column layout.
