package gpu

import "testing"

func newFFNScratch(t testing.TB, device *Device, rows, hidden, intermediate int64) (out *ffnScratch) {
	t.Helper()
	var scratch ffnScratch
	scratch.input = requireTensor(t, device, rows, hidden)
	scratch.gate = requireTensor(t, device, rows, intermediate)
	scratch.up = requireTensor(t, device, rows, intermediate)
	scratch.activated = requireTensor(t, device, rows, intermediate)
	scratch.output = requireTensor(t, device, rows, hidden)
	return &scratch
}

type ffnScratch struct {
	input, gate, up, activated, output *Tensor
}

// encode composes a full FFN from public GPU operations. Weights are stored
// row-major as [outputs, inputs] and can be shared across independent scopes.
func (s *ffnScratch) encode(scope *Scope, weights [3]*Tensor) (err error) {
	if err = scope.MatMulRightTranspose(s.input, weights[0], s.gate); err != nil {
		return err
	}
	if err = scope.MatMulRightTranspose(s.input, weights[1], s.up); err != nil {
		return err
	}
	if err = scope.SwiGLU(s.gate, s.up, s.activated); err != nil {
		return err
	}
	return scope.MatMulRightTranspose(s.activated, weights[2], s.output)
}

func newFFNWeights(t testing.TB, device *Device, hidden, intermediate int64) (weights [3]*Tensor) {
	t.Helper()
	return [3]*Tensor{
		requireTensor(t, device, intermediate, hidden),
		requireTensor(t, device, intermediate, hidden),
		requireTensor(t, device, hidden, intermediate),
	}
}
