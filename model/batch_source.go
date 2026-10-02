package model

import (
	"context"
	"math/rand"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// BatchSource lazily supplies one supervised batch at a time.
//
// Reset starts a deterministic pass for the one-based epoch. The random source
// is supplied for training passes and is nil for metric evaluation passes.
// Next returns done with nil matrices at the end of a pass. The source retains
// ownership of returned matrices and may reuse them after the next call to
// Next or Reset; fitting completes all work with a batch before advancing.
type BatchSource interface {
	Reset(epoch int, random *rand.Rand) (err error)
	Next(ctx context.Context) (inputs, targets *matrix.Matrix, done bool, err error)
}
