package data

// SequenceDatasetSplitViewFunc consumes temporary aligned train and test views
// synchronously. Neither view nor its matrices or lengths may be retained,
// mutated, or used concurrently.
type SequenceDatasetSplitViewFunc func(
	train,
	test *SequenceDatasetView,
) (err error)
