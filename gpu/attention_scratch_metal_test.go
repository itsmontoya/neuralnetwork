//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"math"
	"slices"
	"testing"
)

func newAttentionScratch(t testing.TB, device *Device, f *attentionFixture, rows int64) (scratch *attentionScratch) {
	t.Helper()
	var s attentionScratch
	s.input = requireTensor(t, device, rows, f.hidden)
	s.normalized = requireTensor(t, device, rows, f.hidden)
	s.q = requireTensor(t, device, rows, f.hidden)
	s.k = requireTensor(t, device, rows, f.kvWidth)
	s.v = requireTensor(t, device, rows, f.kvWidth)
	s.rotatedQ = requireTensor(t, device, rows, f.hidden)
	s.rotatedK = requireTensor(t, device, rows, f.kvWidth)
	s.attended = requireTensor(t, device, rows, f.hidden)
	s.projected = requireTensor(t, device, rows, f.hidden)
	s.output = requireTensor(t, device, rows, f.hidden)
	return &s
}

type attentionScratch struct {
	input, normalized, q, k, v, rotatedQ, rotatedK, attended, projected, output *Tensor
}

func (s *attentionScratch) encode(scope *Scope, cache *KVCache, f *attentionFixture, position int64) (update *KVAppend, err error) {
	if err = scope.RMSNorm(s.input, f.norm, s.normalized, attentionEpsilon); err != nil {
		return nil, err
	}
	if err = scope.MatMulRightTranspose(s.normalized, f.q, s.q); err != nil {
		return nil, err
	}
	if err = scope.MatMulRightTranspose(s.normalized, f.k, s.k); err != nil {
		return nil, err
	}
	if err = scope.MatMulRightTranspose(s.normalized, f.v, s.v); err != nil {
		return nil, err
	}
	if err = scope.RotatePairs(s.q, f.cosine, f.sine, s.rotatedQ, f.width, position, 0); err != nil {
		return nil, err
	}
	if err = scope.RotatePairs(s.k, f.cosine, f.sine, s.rotatedK, f.width, position, 0); err != nil {
		return nil, err
	}
	if update, err = scope.AppendKV(cache, s.rotatedK, s.v, position); err != nil {
		return nil, err
	}
	if err = scope.CausalAttention(s.rotatedQ, cache, s.attended, position); err != nil {
		return nil, err
	}
	if err = scope.MatMulRightTranspose(s.attended, f.o, s.projected); err != nil {
		return nil, err
	}
	if err = scope.Add(s.projected, s.input, s.output); err != nil {
		return nil, err
	}
	return update, nil
}

func completeAttentionScope(t testing.TB, scope *Scope) {
	t.Helper()
	var ctx = context.Background()
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)
}

func compareAttention(t testing.TB, got []float32, want []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(want) || len(got) == 0 {
		t.Fatalf("output length %d, reference length %d", len(got), len(want))
	}
	var (
		errors = make([]float64, len(got))
		sum    float64
	)
	for index, value := range got {
		var delta = math.Abs(float64(value) - want[index])
		if math.IsNaN(delta) || delta > tolerance {
			t.Fatalf("attention[%d] = %g, want %g, absolute error %.8g (limit %.8g)", index, value, want[index], delta, tolerance)
		}
		errors[index] = delta
		sum += delta * delta
	}
	slices.Sort(errors)
	t.Logf("n=%d abs_p50=%.8g abs_p95=%.8g abs_max=%.8g rmse=%.8g abs_limit=%.8g", len(got),
		errors[len(errors)/2], errors[(len(errors)-1)*95/100], errors[len(errors)-1], math.Sqrt(sum/float64(len(errors))), tolerance)
}
