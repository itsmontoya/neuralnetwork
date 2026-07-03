package model

// TrainingHistory contains metrics recorded by Fit for each completed epoch.
type TrainingHistory struct {
	// Epochs contains one entry for each completed epoch in chronological order.
	Epochs []EpochMetrics
}

func (h *TrainingHistory) record(metrics EpochMetrics) {
	h.Epochs = append(h.Epochs, metrics)
}
