package device

// ResidencySnapshot captures private matrix-coherence state for repository tests.
type ResidencySnapshot struct {
	State                  string
	LogicalRevision        uint64
	HostRevision           uint64
	DeviceRevision         uint64
	PendingRevision        uint64
	HostRevisionAdvances   uint64
	DeviceRevisionAdvances uint64
	ProposedRevisions      uint64
	Publications           uint64
	DiscardedPublications  uint64
	Uploads                uint64
	Downloads              uint64
	AvoidedUploads         uint64
	DeviceCopies           uint64
	HasBuffer              bool
	HasPendingBuffer       bool
	LastError              string
}
