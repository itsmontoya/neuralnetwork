package model

// EpochMetrics reports model metrics after one completed Fit epoch.
type EpochMetrics struct {
	// Epoch is the one-based epoch number.
	Epoch int
	// Loss is the training loss after the epoch completes.
	Loss float64
	// ValidationLoss is set when validation data is configured.
	ValidationLoss float64
	// HasValidationLoss reports whether ValidationLoss is populated.
	HasValidationLoss bool
	// Accuracy is set when an accuracy callback is configured.
	Accuracy float64
	// HasAccuracy reports whether Accuracy is populated.
	HasAccuracy bool
	// ValidationAccuracy is set when validation data and accuracy are configured.
	ValidationAccuracy float64
	// HasValidationAccuracy reports whether ValidationAccuracy is populated.
	HasValidationAccuracy bool
}
