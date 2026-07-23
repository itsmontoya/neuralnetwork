package device

// ExecutionSnapshot captures private command-batching diagnostics.
type ExecutionSnapshot struct {
	BufferCreations    uint64
	InputUploads       uint64
	ResultDownloads    uint64
	KernelEncodes      uint64
	CommandSubmissions uint64
	Waits              uint64
	Barriers           uint64
	FallbackBarriers   uint64
	Publications       uint64
	DiscardedWrites    uint64
	BoundValues        uint64
	PeakBoundValues    uint64
	TransientBytes     uint64
	PeakTransientBytes uint64
}
