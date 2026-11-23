//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func Test_MetalTensorElementwise(t *testing.T) {
	var (
		gpu       = requireMetalDevice(t)
		ctx       = context.Background()
		native    = gpu.state.backend.(metalBackend).runtime
		initial   = native.ResourceSnapshot()
		inf       = float32(math.Inf(1))
		nan       = float32(math.NaN())
		zero      = float32(math.Copysign(0, -1))
		leftEdges = []float32{0, zero, 1e-30, -1e-30, math.MaxFloat32 / 4, -math.MaxFloat32, -80, -87, -100, 80,
			inf, -inf, inf, 0, nan, 2, -2, math.MaxFloat32, -math.MaxFloat32, 5}
		rightEdges = []float32{0, 1, 1e-6, -1e-6, 2, 0.25, 10, 0.5, 1, -2,
			-inf, -1, 0, inf, 3, nan, inf, math.MaxFloat32, 2, 1e-30}
	)
	var operations = []struct {
		name      string
		encode    func(*Scope, *Tensor, *Tensor, *Tensor) error
		reference func(float32, float32) float32
		tolerance float64
	}{
		{name: "add", encode: (*Scope).Add, reference: func(l, r float32) float32 { return l + r }},
		{name: "mul", encode: (*Scope).Mul, reference: func(l, r float32) float32 { return l * r }},
		{
			name: "silu",
			encode: func(s *Scope, l, _ *Tensor, out *Tensor) error {
				return s.SiLU(l, out)
			},
			reference: func(l, _ float32) float32 { return cpuSiLU(l) },
			tolerance: 2e-6,
		},
		{
			name: "swiglu", encode: (*Scope).SwiGLU,
			reference: func(l, r float32) float32 { return cpuSiLU(l) * r },
			tolerance: 2e-6,
		},
	}
	for _, shape := range [][]int64{{1}, {3, 5, 17}, {1, 257}, {1025}, {4099}} {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			var (
				left   = requireTensor(t, gpu, shape...)
				right  = requireTensor(t, gpu, shape...)
				count  = int(left.state.count)
				random = rand.New(rand.NewPCG(0x4e4e04, 0x2048192))
				l, r   = make([]float32, count), make([]float32, count)
			)
			copy(l, leftEdges)
			copy(r, rightEdges)
			for i := len(leftEdges); i < count; i++ {
				l[i] = 96*random.Float32() - 48
				r[i] = 8*random.Float32() - 4
			}
			requireError(t, left.Upload(l), nil)
			requireError(t, right.Upload(r), nil)
			for _, operation := range operations {
				t.Run(operation.name, func(t *testing.T) {
					var (
						destination = requireTensor(t, gpu, shape...)
						scope       = requireScope(t, gpu)
						got, want   = make([]float32, count), make([]float32, count)
					)
					requireError(t, operation.encode(scope, left, right, destination), nil)
					requireError(t, scope.Submit(ctx), nil)
					requireError(t, scope.Wait(ctx), nil)
					requireError(t, scope.Close(), nil)
					requireError(t, destination.DownloadInto(got), nil)
					for i := range want {
						want[i] = operation.reference(l[i], r[i])
					}
					checkElementwise(t, got, want, operation.tolerance)
				})
			}
		})
	}
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("elementwise operations leaked native resources: before %+v; after %+v", initial, after)
	}
}

func Test_MetalTensorElementwiseIndependentScopes(t *testing.T) {
	var (
		gpu       = requireMetalDevice(t)
		ctx       = context.Background()
		weight    = requireTensor(t, gpu, 257)
		scratch   = requireTensor(t, gpu, 257)
		firstOut  = requireTensor(t, gpu, 257)
		secondOut = requireTensor(t, gpu, 257)
		first     = requireScope(t, gpu)
		second    = requireScope(t, gpu)
		conflict  = requireScope(t, gpu)
		shared    [257]float32
		results   = make(chan error, 2)
	)
	for i := range shared {
		shared[i] = float32(i%17 - 8)
	}
	requireError(t, weight.Upload(shared[:]), nil)
	requireError(t, first.Fill(scratch, 2), nil)
	requireError(t, first.Add(weight, scratch, firstOut), nil)
	requireError(t, second.Mul(weight, weight, secondOut), nil)
	requireError(t, conflict.SiLU(weight, firstOut), ErrBusy)
	requireError(t, conflict.SiLU(scratch, secondOut), ErrBusy)
	requireError(t, weight.Upload(shared[:]), ErrBusy)
	requireError(t, firstOut.DownloadInto(shared[:]), ErrBusy)
	requireError(t, scratch.Close(), nil)
	requireError(t, first.Submit(ctx), nil)
	requireError(t, second.Submit(ctx), nil)
	go func() { results <- first.Wait(ctx) }()
	go func() { results <- second.Wait(ctx) }()
	requireError(t, <-results, nil)
	requireError(t, <-results, nil)

	var a, b [257]float32
	requireError(t, firstOut.DownloadInto(a[:]), nil)
	requireError(t, secondOut.DownloadInto(b[:]), nil)
	for i, value := range shared {
		if a[i] != value+2 || b[i] != value*value {
			t.Fatalf("element %d: got %g and %g from input %g", i, a[i], b[i], value)
		}
	}
}

func cpuSiLU(input float32) (output float32) {
	if input >= 0 {
		var exponent = float32(math.Exp(float64(-input)))
		return input * (1 / (1 + exponent))
	}
	var exponent = float32(math.Exp(float64(input)))
	return input * (exponent / (1 + exponent))
}

func checkElementwise(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	const smallestNormal = 0x1p-126
	var (
		deltas      = make([]float64, 0, len(got))
		maxRelative float64
	)
	for i, value := range got {
		var actual, expected = float64(value), float64(want[i])
		if math.IsNaN(expected) || math.IsInf(expected, 0) {
			if actual == expected || math.IsNaN(expected) && math.IsNaN(actual) {
				continue
			}
			t.Fatalf("element %d: got %g, CPU reference %g", i, actual, expected)
		}
		var (
			delta = math.Abs(actual - expected)
			limit = tolerance * math.Abs(expected)
		)
		if expected != 0 && math.Abs(expected) < smallestNormal {
			limit = smallestNormal
		}
		if math.IsNaN(delta) || delta > limit {
			t.Fatalf("element %d: got %.9g, CPU reference %.9g, absolute error %.3g exceeds %.3g", i, actual, expected, delta, limit)
		}
		deltas = append(deltas, delta)
		if math.Abs(expected) >= smallestNormal {
			maxRelative = max(maxRelative, delta/math.Abs(expected))
		}
	}
	slices.Sort(deltas)
	var last = len(deltas) - 1
	t.Logf("absolute error vs FP32 for %d finite outputs: p50 %.3g, p95 %.3g, p99 %.3g, max %.3g, maximum normal relative error %.3g",
		len(deltas), deltas[last*50/100], deltas[last*95/100], deltas[last*99/100], deltas[last], maxRelative)
}
