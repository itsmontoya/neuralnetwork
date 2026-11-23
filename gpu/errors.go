package gpu

import "errors"

var (
	// ErrUnavailable indicates that this build or machine cannot open a Metal device.
	ErrUnavailable = errors.New("gpu: Metal device unavailable")
	// ErrInvalid indicates an invalid argument or command-scope transition.
	ErrInvalid = errors.New("gpu: invalid argument or scope state")
	// ErrClosed indicates that the handle is closed or uninitialized.
	ErrClosed = errors.New("gpu: handle is closed")
	// ErrBusy indicates a conflicting tensor access or a device with live children.
	ErrBusy = errors.New("gpu: resource is busy")
	// ErrUninitialized indicates that the tensor needs a successful full write before it can be read.
	ErrUninitialized = errors.New("gpu: tensor has no readable values")
)
