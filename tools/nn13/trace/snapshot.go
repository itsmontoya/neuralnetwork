// Package trace is serial benchmark instrumentation, never a production kernel
// dependency. Instrumented copies use it only outside latency/allocation timing.
package trace

// Snapshot reports executed payload copies and function-scoped live storage.
// Numeric payload bytes exclude Go objects, shape metadata, wire bytes and RSS.
type Snapshot struct {
	CopiedBytes    int64            `json:"copied_bytes"`
	PeakBytes      int64            `json:"peak_payload_bytes"`
	AllocatedBytes int64            `json:"allocated_payload_bytes"`
	Copies         map[string]int64 `json:"copies_by_site"`
	Peak           map[string]int64 `json:"peak_by_site"`
}
