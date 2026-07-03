package model

// FitCallback receives metrics after each completed training epoch.
type FitCallback func(metrics EpochMetrics) (err error)
