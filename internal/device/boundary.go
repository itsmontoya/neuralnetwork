package device

// Boundary identifies why pending device commands must complete.
type Boundary uint8

const (
	// BoundaryHostObservation completes producers before a host read.
	BoundaryHostObservation Boundary = iota
	// BoundaryHostMutation completes users before a host overwrite.
	BoundaryHostMutation
	// BoundaryCPUFallback completes producers before CPU work.
	BoundaryCPUFallback
	// BoundaryTopLevel completes work before a public model return.
	BoundaryTopLevel
	// BoundaryCommandLimit rotates a full command buffer.
	BoundaryCommandLimit
)
