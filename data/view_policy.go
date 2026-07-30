package data

import "fmt"

// ViewPolicy controls whether a non-contiguous view operation may copy its
// selected values. Its zero value is the strict ViewOnly policy.
type ViewPolicy uint8

const (
	// ViewOnly rejects selections that cannot share contiguous owner storage.
	ViewOnly ViewPolicy = iota
	// ViewOrCopy permits an explicit copied fallback for non-contiguous rows.
	ViewOrCopy
)

func validateViewPolicy(prefix string, policy ViewPolicy) (err error) {
	if policy != ViewOnly && policy != ViewOrCopy {
		err = fmt.Errorf("%s policy is invalid: policy=%d", prefix, policy)
		return err
	}

	return nil
}
