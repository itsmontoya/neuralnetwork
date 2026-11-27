//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"
)

func geluReferenceFP64(x float64) (out float64) {
	if math.IsNaN(x) {
		return x
	}
	if math.IsInf(x, 1) {
		return x
	}
	if math.IsInf(x, -1) {
		return 0
	}
	return 0.5 * x * (1.0 + math.Erf(x/math.Sqrt2))
}

func geluTanhReferenceFP64(x float64) (out float64) {
	if math.IsNaN(x) {
		return x
	}
	if math.IsInf(x, 1) {
		return x
	}
	if math.IsInf(x, -1) {
		return 0
	}
	var (
		coeff = math.Sqrt(2.0 / math.Pi)
		inner = coeff * (x + 0.044715*x*x*x)
	)
	if inner >= 0 {
		return x / (1.0 + math.Exp(-2.0*inner))
	}
	var e = math.Exp(2.0 * inner)
	return x * e / (1.0 + e)
}

func checkGELUAgainstFP64(t *testing.T, label string, got []float32, want []float64, absTol, relTol float64) {
	t.Helper()
	var (
		deltas = make([]float64, 0, len(got))
		maxRel float64
	)
	for i, value := range got {
		var (
			actual   = float64(value)
			expected = want[i]
		)
		if math.IsNaN(expected) || math.IsInf(expected, 0) {
			if actual == expected || (math.IsNaN(expected) && math.IsNaN(actual)) {
				continue
			}
			t.Fatalf("%s element %d: got %g, FP64 reference %g", label, i, actual, expected)
		}
		var (
			delta = math.Abs(actual - expected)
			limit = absTol + relTol*math.Abs(expected)
		)
		if math.IsNaN(delta) || delta > limit {
			t.Fatalf("%s element %d: got %.9g, FP64 reference %.9g, abs error %.3g exceeds %.3g",
				label, i, actual, expected, delta, limit)
		}
		deltas = append(deltas, delta)
		if math.Abs(expected) >= 1e-6 {
			maxRel = max(maxRel, delta/math.Abs(expected))
		}
	}
	if len(deltas) == 0 {
		return
	}
	slices.Sort(deltas)
	var last = len(deltas) - 1
	t.Logf("%s FP64 abs error (%d finite): p50=%.3g p95=%.3g p99=%.3g max=%.3g maxRel=%.3g",
		label, len(deltas), deltas[last*50/100], deltas[last*95/100], deltas[last*99/100], deltas[last], maxRel)
}

func Test_MetalGELUAgainstFP64(t *testing.T) {
	var (
		gpu     = requireMetalDevice(t)
		ctx     = context.Background()
		native  = gpu.state.backend.(metalBackend).runtime
		initial = native.ResourceSnapshot()
		inf     = float32(math.Inf(1))
		nan     = float32(math.NaN())
		negZero = float32(math.Copysign(0, -1))
		edges   = []float32{
			0, negZero, 1e-40, -1e-40, math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
			1e-30, -1e-30, 1e-6, -1e-6, 0.1, -0.1, 0.5, -0.5, 1, -1, 2, -2, 3, -3, 5, -5,
			9.5, -9.5, 9.99, -9.99, 10, -10, 10.01, -10.01, 20, -20,
			80, -80, 100, -100, 1000, -1000,
			math.MaxFloat32, -math.MaxFloat32, inf, -inf, nan,
		}
		operations = []struct {
			name      string
			encode    func(*Scope, *Tensor, *Tensor) error
			reference func(float64) float64
			tolerance float64
		}{
			{
				name:      "gelu",
				encode:    (*Scope).GELU,
				reference: geluReferenceFP64,
				tolerance: 1e-6,
			},
			{
				name:      "gelu_tanh",
				encode:    (*Scope).GELUTanh,
				reference: geluTanhReferenceFP64,
				tolerance: 1e-6,
			},
		}
		shapes = [][]int64{
			{1},
			{3},
			{17},
			{31},
			{32},
			{33},
			{2, 64},
			{3, 5, 17},
			{1, 257},
			{2, 2, 4, 33},
			{4096},
		}
	)

	for _, shape := range shapes {
		t.Run(fmt.Sprint(shape), func(t *testing.T) {
			var (
				input  = requireTensor(t, gpu, shape...)
				count  = int(input.state.storage.count)
				random = rand.New(rand.NewPCG(0x4e4e26, uint64(count)))
				values = make([]float32, count)
			)
			copy(values, edges)
			for i := len(edges); i < count; i++ {
				values[i] = 24*random.Float32() - 12
			}
			requireError(t, input.Upload(values), nil)

			for _, op := range operations {
				t.Run(op.name, func(t *testing.T) {
					var (
						destination = requireTensor(t, gpu, shape...)
						scope       = requireScope(t, gpu)
						got         = make([]float32, count)
						want        = make([]float64, count)
					)
					requireError(t, op.encode(scope, input, destination), nil)
					requireError(t, scope.Submit(ctx), nil)
					requireError(t, scope.Wait(ctx), nil)
					requireError(t, scope.Close(), nil)
					requireError(t, destination.DownloadInto(got), nil)
					for i, v := range values {
						want[i] = op.reference(float64(v))
					}
					checkGELUAgainstFP64(t, op.name, got, want, op.tolerance, op.tolerance)
				})
			}
		})
	}

	t.Run("dense_sweep_error_distribution", func(t *testing.T) {
		var (
			count  = 20001
			input  = requireTensor(t, gpu, int64(count))
			values = make([]float32, count)
		)
		for i := range count {
			values[i] = -12.0 + 24.0*float32(i)/float32(count-1)
		}
		requireError(t, input.Upload(values), nil)

		for _, op := range operations {
			var (
				destination = requireTensor(t, gpu, int64(count))
				scope       = requireScope(t, gpu)
				got         = make([]float32, count)
				maxAbs      float64
				maxRel      float64
			)
			requireError(t, op.encode(scope, input, destination), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)
			requireError(t, destination.DownloadInto(got), nil)
			for i, v := range values {
				var (
					want = op.reference(float64(v))
					abs  = math.Abs(float64(got[i]) - want)
				)
				if abs > maxAbs {
					maxAbs = abs
				}
				if denom := math.Abs(want); denom > 1e-6 {
					var rel = abs / denom
					if rel > maxRel {
						maxRel = rel
					}
				}
			}
			t.Logf("%s dense sweep [-12, 12] (%d pts): maxAbs=%.3e maxRel=%.3e", op.name, count, maxAbs, maxRel)
			if maxAbs >= op.tolerance {
				t.Fatalf("%s maxAbs=%.3e >= %.3e", op.name, maxAbs, op.tolerance)
			}
		}
	})

	t.Run("view_alias_rejected", func(t *testing.T) {
		var (
			input    = requireTensor(t, gpu, 4, 8)
			alias, e = input.View([]int64{4, 8})
		)
		requireError(t, e, nil)
		defer alias.Close()
		requireError(t, input.Upload(make([]float32, 32)), nil)
		for _, op := range operations {
			var scope = requireScope(t, gpu)
			requireError(t, op.encode(scope, input, alias), ErrInvalid)
			requireError(t, scope.Close(), nil)
		}
	})

	t.Run("concurrent_scopes", func(t *testing.T) {
		const (
			workers    = 8
			iterations = 12
			count      = 1024
		)
		var (
			input  = requireTensor(t, gpu, 4, 256)
			values = make([]float32, count)
			wantG  = make([]float64, count)
			wantT  = make([]float64, count)
			random = rand.New(rand.NewPCG(0x4e4e26c, 1))
			wg     sync.WaitGroup
			errCh  = make(chan error, workers)
		)
		for i := range values {
			values[i] = 20*random.Float32() - 10
			wantG[i] = geluReferenceFP64(float64(values[i]))
			wantT[i] = geluTanhReferenceFP64(float64(values[i]))
		}
		requireError(t, input.Upload(values), nil)

		for worker := range workers {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()
				var (
					dstG = requireTensor(t, gpu, 4, 256)
					dstT = requireTensor(t, gpu, 4, 256)
					gotG = make([]float32, count)
					gotT = make([]float32, count)
				)
				for range iterations {
					var scope, err = gpu.NewScope()
					if err != nil {
						errCh <- fmt.Errorf("worker %d NewScope: %w", id, err)
						return
					}
					if err = scope.GELU(input, dstG); err != nil {
						_ = scope.Close()
						errCh <- fmt.Errorf("worker %d GELU: %w", id, err)
						return
					}
					if err = scope.GELUTanh(input, dstT); err != nil {
						_ = scope.Close()
						errCh <- fmt.Errorf("worker %d GELUTanh: %w", id, err)
						return
					}
					if err = scope.Submit(ctx); err != nil {
						_ = scope.Close()
						errCh <- fmt.Errorf("worker %d Submit: %w", id, err)
						return
					}
					if err = scope.Wait(ctx); err != nil {
						_ = scope.Close()
						errCh <- fmt.Errorf("worker %d Wait: %w", id, err)
						return
					}
					if err = scope.Close(); err != nil {
						errCh <- fmt.Errorf("worker %d Close: %w", id, err)
						return
					}
					if err = dstG.DownloadInto(gotG); err != nil {
						errCh <- fmt.Errorf("worker %d Download G: %w", id, err)
						return
					}
					if err = dstT.DownloadInto(gotT); err != nil {
						errCh <- fmt.Errorf("worker %d Download T: %w", id, err)
						return
					}
					for i := range count {
						if math.Abs(float64(gotG[i])-wantG[i]) > 1e-6 {
							errCh <- fmt.Errorf("worker %d GELU[%d]=%g want %g", id, i, gotG[i], wantG[i])
							return
						}
						if math.Abs(float64(gotT[i])-wantT[i]) > 1e-6 {
							errCh <- fmt.Errorf("worker %d GELUTanh[%d]=%g want %g", id, i, gotT[i], wantT[i])
							return
						}
					}
				}
			}(worker)
		}
		wg.Wait()
		close(errCh)
		for err := range errCh {
			t.Fatal(err)
		}
	})

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != initial.LiveBuffers || after.LiveBufferBytes != initial.LiveBufferBytes || after.LiveScopes != initial.LiveScopes {
		t.Fatalf("GELU tests leaked native resources: before %+v; after %+v", initial, after)
	}
}
