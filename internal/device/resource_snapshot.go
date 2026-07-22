package device

// ResourceSnapshot captures private backend resource activity.
type ResourceSnapshot struct {
	LiveBuffers       uint64
	LiveBufferBytes   uint64
	PeakBuffers       uint64
	PeakBufferBytes   uint64
	LiveScopes        uint64
	PeakScopes        uint64
	CreatedBuffers    uint64
	ReleasedBuffers   uint64
	CreatedScopes     uint64
	ReleasedScopes    uint64
	SubmittedCommands uint64
	CompletedCommands uint64
}
