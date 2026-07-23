package device

import "fmt"

// CategoricalTargetError preserves categorical one-hot validation diagnostics.
type CategoricalTargetError struct {
	Row       uint32
	Column    uint32
	Ones      uint32
	Value     float32
	NonBinary bool
}

// Error returns the existing categorical target diagnostic.
func (e CategoricalTargetError) Error() (message string) {
	if e.NonBinary {
		message = fmt.Sprintf(
			"loss: categorical target at row %d column %d must be 0 or 1: value=%g",
			e.Row,
			e.Column,
			e.Value,
		)
		return message
	}

	message = fmt.Sprintf(
		"loss: categorical target row %d must contain exactly one class: ones=%d",
		e.Row,
		e.Ones,
	)
	return message
}
