package device

import "errors"

var (
	// ErrUnavailable reports that this build or machine has no usable device backend.
	ErrUnavailable = errors.New("device: backend unavailable")
	// ErrReleased reports an operation on a released private resource.
	ErrReleased = errors.New("device: resource is released")
	// ErrInvalidState reports an invalid command-scope transition.
	ErrInvalidState = errors.New("device: invalid command scope state")
)
