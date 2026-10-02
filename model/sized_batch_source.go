package model

// SizedBatchSource optionally reports the expected size of each complete pass.
// A false known result permits sources with dynamic or unknown counts.
type SizedBatchSource interface {
	BatchSource
	SampleCount() (count int, known bool)
	BatchCount() (count int, known bool)
}
