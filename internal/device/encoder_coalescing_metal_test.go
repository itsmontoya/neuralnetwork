//go:build darwin && cgo && metal && !purego

package device

import (
	"fmt"
	"math"
	"sync"
	"testing"
)

func Test_MetalScopeEncoderCoalescingInterleavedComputeAndBlit(t *testing.T) {
	const count = 257

	var (
		runtime  *Runtime
		baseline ResourceSnapshot
		after    ResourceSnapshot
		a        *Buffer
		b        *Buffer
		c        *Buffer
		d        *Buffer
		e        *Buffer
		f        *Buffer
		scope    *Scope
		aHost    []float32
		bHost    []float32
		gotC     []float32
		gotD     []float32
		gotF     []float32
		index    int
		err      error
	)

	runtime = requireMetalRuntime(t)
	baseline = runtime.ResourceSnapshot()

	if a, err = runtime.NewBuffer(count); err != nil {
		t.Fatalf("NewBuffer a returned error: %v", err)
	}
	defer a.Release()
	if b, err = runtime.NewBuffer(count); err != nil {
		t.Fatalf("NewBuffer b returned error: %v", err)
	}
	defer b.Release()
	if c, err = runtime.NewBuffer(count); err != nil {
		t.Fatalf("NewBuffer c returned error: %v", err)
	}
	defer c.Release()
	if d, err = runtime.NewBuffer(count); err != nil {
		t.Fatalf("NewBuffer d returned error: %v", err)
	}
	defer d.Release()
	if e, err = runtime.NewBuffer(count); err != nil {
		t.Fatalf("NewBuffer e returned error: %v", err)
	}
	defer e.Release()
	if f, err = runtime.NewBuffer(count); err != nil {
		t.Fatalf("NewBuffer f returned error: %v", err)
	}
	defer f.Release()

	aHost = make([]float32, count)
	bHost = make([]float32, count)
	for index = 0; index < count; index++ {
		aHost[index] = float32(index%19-9) * 0.5
		bHost[index] = float32(index%7-3) * 0.25
	}
	if err = a.Upload(aHost); err != nil {
		t.Fatalf("Upload a returned error: %v", err)
	}
	if err = b.Upload(bHost); err != nil {
		t.Fatalf("Upload b returned error: %v", err)
	}

	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}
	// Compute -> Compute -> Blit -> Compute -> Compute -> Blit -> Compute in one scope.
	if err = scope.EncodeElementwise(a, b, c, ElementwiseAdd); err != nil {
		scope.Release()
		t.Fatalf("EncodeElementwise Add returned error: %v", err)
	}
	if err = scope.EncodeTensorScale(c, e, 2.0); err != nil {
		scope.Release()
		t.Fatalf("EncodeTensorScale returned error: %v", err)
	}
	if err = scope.EncodeCopy(e, d); err != nil {
		scope.Release()
		t.Fatalf("first EncodeCopy returned error: %v", err)
	}
	if err = scope.EncodeElementwise(d, nil, c, ElementwiseReLU); err != nil {
		scope.Release()
		t.Fatalf("EncodeElementwise ReLU returned error: %v", err)
	}
	if err = scope.EncodeElementwise(c, b, e, ElementwiseMul); err != nil {
		scope.Release()
		t.Fatalf("EncodeElementwise Mul returned error: %v", err)
	}
	if err = scope.EncodeCopy(e, d); err != nil {
		scope.Release()
		t.Fatalf("second EncodeCopy returned error: %v", err)
	}
	if err = scope.EncodeElementwise(d, a, f, ElementwiseSub); err != nil {
		scope.Release()
		t.Fatalf("EncodeElementwise Sub returned error: %v", err)
	}
	if err = scope.Commit(); err != nil {
		scope.Release()
		t.Fatalf("Commit returned error: %v", err)
	}
	if err = scope.Wait(); err != nil {
		scope.Release()
		t.Fatalf("Wait returned error: %v", err)
	}
	if err = scope.Release(); err != nil {
		t.Fatalf("Release returned error: %v", err)
	}

	gotC = make([]float32, count)
	gotD = make([]float32, count)
	gotF = make([]float32, count)
	if err = c.Download(gotC); err != nil {
		t.Fatalf("Download c returned error: %v", err)
	}
	if err = d.Download(gotD); err != nil {
		t.Fatalf("Download d returned error: %v", err)
	}
	if err = f.Download(gotF); err != nil {
		t.Fatalf("Download f returned error: %v", err)
	}

	for index = 0; index < count; index++ {
		var (
			sum    = aHost[index] + bHost[index]
			scaled = sum * 2.0
			relu   = scaled
		)
		if relu < 0 {
			relu = 0
		}
		var (
			prod = relu * bHost[index]
			sub  = prod - aHost[index]
		)
		if math.Float32bits(gotC[index]) != math.Float32bits(relu) {
			t.Fatalf("c[%d] = %g (0x%08x), want %g (0x%08x)", index, gotC[index], math.Float32bits(gotC[index]), relu, math.Float32bits(relu))
		}
		if math.Float32bits(gotD[index]) != math.Float32bits(prod) {
			t.Fatalf("d[%d] = %g (0x%08x), want %g (0x%08x)", index, gotD[index], math.Float32bits(gotD[index]), prod, math.Float32bits(prod))
		}
		if math.Float32bits(gotF[index]) != math.Float32bits(sub) {
			t.Fatalf("f[%d] = %g (0x%08x), want %g (0x%08x)", index, gotF[index], math.Float32bits(gotF[index]), sub, math.Float32bits(sub))
		}
	}

	after = runtime.ResourceSnapshot()
	if after.LiveScopes != baseline.LiveScopes {
		t.Fatalf("live scopes changed from %d to %d", baseline.LiveScopes, after.LiveScopes)
	}
	if after.SubmittedCommands != baseline.SubmittedCommands+1 ||
		after.CompletedCommands != baseline.CompletedCommands+1 {
		t.Fatalf("want 1 submitted and completed command, before=%+v after=%+v", baseline, after)
	}
}

func Test_MetalScopeEncoderCoalescingDiscardOpenEncoder(t *testing.T) {
	var (
		runtime  *Runtime
		baseline ResourceSnapshot
		after    ResourceSnapshot
		bufA     *Buffer
		bufB     *Buffer
		scope    *Scope
		got      []float32
		err      error
	)

	runtime = requireMetalRuntime(t)
	baseline = runtime.ResourceSnapshot()

	if bufA, err = runtime.NewBuffer(64); err != nil {
		t.Fatalf("NewBuffer bufA returned error: %v", err)
	}
	defer bufA.Release()
	if bufB, err = runtime.NewBuffer(64); err != nil {
		t.Fatalf("NewBuffer bufB returned error: %v", err)
	}
	defer bufB.Release()

	// Encode compute ops (leaving computeEncoder open) and release without committing.
	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}
	if err = scope.EncodeFill(bufA, 7.5); err != nil {
		scope.Release()
		t.Fatalf("EncodeFill returned error: %v", err)
	}
	if err = scope.EncodeTensorScale(bufA, bufB, 3.0); err != nil {
		scope.Release()
		t.Fatalf("EncodeTensorScale returned error: %v", err)
	}
	if err = scope.Release(); err != nil {
		t.Fatalf("Release uncommitted scope with open compute encoder returned error: %v", err)
	}

	after = runtime.ResourceSnapshot()
	if after.LiveScopes != baseline.LiveScopes {
		t.Fatalf("live scopes after discard = %d, want %d", after.LiveScopes, baseline.LiveScopes)
	}
	if after.SubmittedCommands != baseline.SubmittedCommands {
		t.Fatalf("discarded scope submitted command: before=%+v after=%+v", baseline, after)
	}

	// Verify runtime remains healthy on a follow-up committed scope.
	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("follow-up NewScope returned error: %v", err)
	}
	if err = scope.EncodeFill(bufA, 2.25); err != nil {
		scope.Release()
		t.Fatalf("follow-up EncodeFill returned error: %v", err)
	}
	if err = scope.Commit(); err != nil {
		scope.Release()
		t.Fatalf("follow-up Commit returned error: %v", err)
	}
	if err = scope.Wait(); err != nil {
		scope.Release()
		t.Fatalf("follow-up Wait returned error: %v", err)
	}
	if err = scope.Release(); err != nil {
		t.Fatalf("follow-up Release returned error: %v", err)
	}

	got = make([]float32, 64)
	if err = bufA.Download(got); err != nil {
		t.Fatalf("Download returned error: %v", err)
	}
	requireAllFloat32(t, got, 2.25)
}

func Test_MetalScopeEncoderCoalescingMidScopeEncodeFailure(t *testing.T) {
	var (
		runtime  *Runtime
		baseline ResourceSnapshot
		after    ResourceSnapshot
		small    *Buffer
		large    *Buffer
		scope    *Scope
		err      error
	)

	runtime = requireMetalRuntime(t)
	baseline = runtime.ResourceSnapshot()

	if small, err = runtime.NewBuffer(16); err != nil {
		t.Fatalf("NewBuffer small returned error: %v", err)
	}
	defer small.Release()
	if large, err = runtime.NewBuffer(32); err != nil {
		t.Fatalf("NewBuffer large returned error: %v", err)
	}
	defer large.Release()

	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}
	if err = scope.EncodeFill(small, 4.0); err != nil {
		scope.Release()
		t.Fatalf("EncodeFill returned error: %v", err)
	}
	// Trigger a mid-scope encode error while computeEncoder is open.
	if err = scope.EncodeTensorScale(small, large, 2.0); err == nil {
		scope.Release()
		t.Fatal("EncodeTensorScale with mismatched buffer lengths returned nil error")
	}
	// Also test blit encode error while computeEncoder is open.
	if err = scope.EncodeCopy(small, large); err == nil {
		scope.Release()
		t.Fatal("EncodeCopy with mismatched buffer lengths returned nil error")
	}
	if err = scope.Release(); err != nil {
		t.Fatalf("Release after mid-scope encode failure returned error: %v", err)
	}

	after = runtime.ResourceSnapshot()
	if after.LiveScopes != baseline.LiveScopes {
		t.Fatalf("live scopes after failed encode = %d, want %d", after.LiveScopes, baseline.LiveScopes)
	}
	if after.SubmittedCommands != baseline.SubmittedCommands {
		t.Fatalf("failed uncommitted scope submitted command: before=%+v after=%+v", baseline, after)
	}
}

func Test_MetalScopeEncoderCoalescingConcurrentStress(t *testing.T) {
	const (
		workers    = 8
		iterations = 16
		count      = 128
	)

	var (
		runtime   *Runtime
		baseline  ResourceSnapshot
		after     ResourceSnapshot
		waitGroup sync.WaitGroup
		errorsOut chan error
		worker    int
		err       error
	)

	runtime = requireMetalRuntime(t)
	baseline = runtime.ResourceSnapshot()
	errorsOut = make(chan error, workers)

	for worker = 0; worker < workers; worker++ {
		waitGroup.Add(1)
		go func(workerID int) {
			defer waitGroup.Done()
			var (
				a         *Buffer
				b         *Buffer
				c         *Buffer
				d         *Buffer
				scope     *Scope
				discarded *Scope
				got       []float32
				iter      int
				idx       int
				werr      error
			)
			if a, werr = runtime.NewBuffer(count); werr != nil {
				errorsOut <- werr
				return
			}
			defer a.Release()
			if b, werr = runtime.NewBuffer(count); werr != nil {
				errorsOut <- werr
				return
			}
			defer b.Release()
			if c, werr = runtime.NewBuffer(count); werr != nil {
				errorsOut <- werr
				return
			}
			defer c.Release()
			if d, werr = runtime.NewBuffer(count); werr != nil {
				errorsOut <- werr
				return
			}
			defer d.Release()

			got = make([]float32, count)
			for iter = 0; iter < iterations; iter++ {
				var base = float32(workerID*iterations + iter + 1)

				// Periodically discard a scope with an open compute encoder.
				if iter%4 == 0 {
					if discarded, werr = runtime.NewScope(); werr != nil {
						errorsOut <- werr
						return
					}
					if werr = discarded.EncodeFill(a, -999); werr != nil {
						discarded.Release()
						errorsOut <- werr
						return
					}
					if werr = discarded.Release(); werr != nil {
						errorsOut <- werr
						return
					}
				}

				if scope, werr = runtime.NewScope(); werr != nil {
					errorsOut <- werr
					return
				}
				if werr = scope.EncodeFill(a, base); werr != nil {
					scope.Release()
					errorsOut <- werr
					return
				}
				if werr = scope.EncodeTensorScale(a, b, 2.0); werr != nil {
					scope.Release()
					errorsOut <- werr
					return
				}
				if werr = scope.EncodeCopy(b, c); werr != nil {
					scope.Release()
					errorsOut <- werr
					return
				}
				if werr = scope.EncodeElementwise(c, a, d, ElementwiseAdd); werr != nil {
					scope.Release()
					errorsOut <- werr
					return
				}
				if werr = scope.Commit(); werr != nil {
					scope.Release()
					errorsOut <- werr
					return
				}
				if werr = scope.Wait(); werr != nil {
					scope.Release()
					errorsOut <- werr
					return
				}
				if werr = scope.Release(); werr != nil {
					errorsOut <- werr
					return
				}
				if werr = d.Download(got); werr != nil {
					errorsOut <- werr
					return
				}
				var want = base*2.0 + base
				for idx = 0; idx < count; idx++ {
					if math.Float32bits(got[idx]) != math.Float32bits(want) {
						errorsOut <- fmt.Errorf("worker %d iter %d idx %d = %g, want %g", workerID, iter, idx, got[idx], want)
						return
					}
				}
			}
			errorsOut <- nil
		}(worker)
	}

	waitGroup.Wait()
	close(errorsOut)
	for err = range errorsOut {
		if err != nil {
			t.Fatalf("concurrent encoder coalescing worker failed: %v", err)
		}
	}

	after = runtime.ResourceSnapshot()
	if after.LiveBuffers != baseline.LiveBuffers ||
		after.LiveBufferBytes != baseline.LiveBufferBytes ||
		after.LiveScopes != baseline.LiveScopes {
		t.Fatalf("native live resources leaked: before=%+v after=%+v", baseline, after)
	}
}
