//go:build darwin && cgo && metal && !purego

package device

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"
)

func Test_MetalBufferRoundTripAndFill(t *testing.T) {
	var (
		runtime *Runtime
		buffer  *Buffer
		scope   *Scope
		got     []float32
		err     error
	)

	runtime = requireMetalRuntime(t)
	if buffer, err = runtime.NewBuffer(4); err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	defer buffer.Release()
	if err = buffer.Upload([]float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Upload returned error: %v", err)
	}
	got = make([]float32, 4)
	if err = buffer.Download(got); err != nil {
		t.Fatalf("Download returned error: %v", err)
	}
	requireFloat32Values(t, got, []float32{1, 2, 3, 4})

	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}
	if err = scope.EncodeFill(buffer, -2.5); err != nil {
		t.Fatalf("EncodeFill returned error: %v", err)
	}
	if err = scope.Commit(); err != nil {
		t.Fatalf("Commit returned error: %v", err)
	}
	if err = scope.Wait(); err != nil {
		t.Fatalf("Wait returned error: %v", err)
	}
	if err = scope.Release(); err != nil {
		t.Fatalf("Release returned error: %v", err)
	}
	if err = buffer.Download(got); err != nil {
		t.Fatalf("Download after fill returned error: %v", err)
	}
	requireFloat32Values(t, got, []float32{-2.5, -2.5, -2.5, -2.5})

	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope for zero returned error: %v", err)
	}
	if err = scope.EncodeFill(buffer, 0); err != nil {
		t.Fatalf("EncodeFill zero returned error: %v", err)
	}
	if err = scope.Commit(); err != nil {
		t.Fatalf("Commit zero returned error: %v", err)
	}
	if err = scope.Wait(); err != nil {
		t.Fatalf("Wait zero returned error: %v", err)
	}
	if err = scope.Release(); err != nil {
		t.Fatalf("Release zero returned error: %v", err)
	}
	if err = buffer.Download(got); err != nil {
		t.Fatalf("Download after zero returned error: %v", err)
	}
	requireFloat32Values(t, got, []float32{0, 0, 0, 0})
}

func Test_MetalScopeCommandOrdering(t *testing.T) {
	var (
		runtime *Runtime
		source  *Buffer
		result  *Buffer
		scope   *Scope
		got     []float32
		err     error
	)

	runtime = requireMetalRuntime(t)
	if source, err = runtime.NewBuffer(32); err != nil {
		t.Fatalf("NewBuffer source returned error: %v", err)
	}
	defer source.Release()
	if result, err = runtime.NewBuffer(32); err != nil {
		t.Fatalf("NewBuffer result returned error: %v", err)
	}
	defer result.Release()
	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}
	defer scope.Release()

	if err = scope.EncodeFill(source, 3); err != nil {
		t.Fatalf("EncodeFill source returned error: %v", err)
	}
	if err = scope.EncodeCopy(source, result); err != nil {
		t.Fatalf("EncodeCopy returned error: %v", err)
	}
	if err = scope.EncodeFill(source, 9); err != nil {
		t.Fatalf("second EncodeFill source returned error: %v", err)
	}
	if err = scope.Commit(); err != nil {
		t.Fatalf("Commit returned error: %v", err)
	}
	if err = scope.Wait(); err != nil {
		t.Fatalf("Wait returned error: %v", err)
	}

	got = make([]float32, 32)
	if err = result.Download(got); err != nil {
		t.Fatalf("Download result returned error: %v", err)
	}
	requireAllFloat32(t, got, 3)
	if err = source.Download(got); err != nil {
		t.Fatalf("Download source returned error: %v", err)
	}
	requireAllFloat32(t, got, 9)
}

func Test_MetalDenseForwardKernels(t *testing.T) {
	var (
		runtime   *Runtime
		values    *Buffer
		rowVector *Buffer
		relu      *Buffer
		softmax   *Buffer
		scope     *Scope
		got       []float32
		err       error
	)

	runtime = requireMetalRuntime(t)
	if values, err = runtime.NewBuffer(6); err != nil {
		t.Fatalf("NewBuffer values returned error: %v", err)
	}
	defer values.Release()
	if rowVector, err = runtime.NewBuffer(3); err != nil {
		t.Fatalf("NewBuffer row vector returned error: %v", err)
	}
	defer rowVector.Release()
	if relu, err = runtime.NewBuffer(6); err != nil {
		t.Fatalf("NewBuffer ReLU returned error: %v", err)
	}
	defer relu.Release()
	if softmax, err = runtime.NewBuffer(6); err != nil {
		t.Fatalf("NewBuffer Softmax returned error: %v", err)
	}
	defer softmax.Release()

	if err = values.Upload([]float32{-3, -1, 1, 2, 4, 6}); err != nil {
		t.Fatalf("Upload values returned error: %v", err)
	}
	if err = rowVector.Upload([]float32{1, -2, 3}); err != nil {
		t.Fatalf("Upload row vector returned error: %v", err)
	}
	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}
	defer scope.Release()

	if err = scope.EncodeAddRowVector(values, rowVector, 2, 3); err != nil {
		t.Fatalf("EncodeAddRowVector returned error: %v", err)
	}
	if err = scope.EncodeReLU(values, relu); err != nil {
		t.Fatalf("EncodeReLU returned error: %v", err)
	}
	if err = scope.EncodeSoftmaxRows(relu, softmax, 2, 3); err != nil {
		t.Fatalf("EncodeSoftmaxRows returned error: %v", err)
	}
	if err = scope.Commit(); err != nil {
		t.Fatalf("Commit returned error: %v", err)
	}
	if err = scope.Wait(); err != nil {
		t.Fatalf("Wait returned error: %v", err)
	}

	got = make([]float32, 6)
	if err = values.Download(got); err != nil {
		t.Fatalf("Download biased values returned error: %v", err)
	}
	requireMetalValuesAlmostEqual(t, got, []float32{-2, -3, 4, 3, 2, 9}, 0)
	if err = relu.Download(got); err != nil {
		t.Fatalf("Download ReLU returned error: %v", err)
	}
	requireMetalValuesAlmostEqual(t, got, []float32{0, 0, 4, 3, 2, 9}, 0)
	if err = softmax.Download(got); err != nil {
		t.Fatalf("Download Softmax returned error: %v", err)
	}
	requireMetalValuesAlmostEqual(
		t,
		got,
		[]float32{0.017668422, 0.017668422, 0.96466315, 0.002472318, 0.000909011, 0.9966187},
		2e-5,
	)
}

func Test_MetalDenseBackwardKernels(t *testing.T) {
	var (
		runtime        *Runtime
		input          *Buffer
		outputGradient *Buffer
		relu           *Buffer
		softmax        *Buffer
		columnSums     *Buffer
		accumulated    *Buffer
		added          *Buffer
		scope          *Scope
		got            []float32
		err            error
	)

	runtime = requireMetalRuntime(t)
	if input, err = runtime.NewBuffer(6); err != nil {
		t.Fatalf("NewBuffer input returned error: %v", err)
	}
	defer input.Release()
	if outputGradient, err = runtime.NewBuffer(6); err != nil {
		t.Fatalf("NewBuffer output gradient returned error: %v", err)
	}
	defer outputGradient.Release()
	if relu, err = runtime.NewBuffer(6); err != nil {
		t.Fatalf("NewBuffer ReLU returned error: %v", err)
	}
	defer relu.Release()
	if softmax, err = runtime.NewBuffer(6); err != nil {
		t.Fatalf("NewBuffer Softmax returned error: %v", err)
	}
	defer softmax.Release()
	if columnSums, err = runtime.NewBuffer(3); err != nil {
		t.Fatalf("NewBuffer column sums returned error: %v", err)
	}
	defer columnSums.Release()
	if accumulated, err = runtime.NewBuffer(3); err != nil {
		t.Fatalf("NewBuffer accumulated column sums returned error: %v", err)
	}
	defer accumulated.Release()
	if added, err = runtime.NewBuffer(6); err != nil {
		t.Fatalf("NewBuffer addition returned error: %v", err)
	}
	defer added.Release()

	if err = input.Upload([]float32{-1, 0, 2, 1, -2, 3}); err != nil {
		t.Fatalf("Upload input returned error: %v", err)
	}
	if err = outputGradient.Upload([]float32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatalf("Upload output gradient returned error: %v", err)
	}
	if err = accumulated.Upload([]float32{10, 20, 30}); err != nil {
		t.Fatalf("Upload accumulated column sums returned error: %v", err)
	}
	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}
	defer scope.Release()

	if err = scope.EncodeReLUBackward(input, outputGradient, relu); err != nil {
		t.Fatalf("EncodeReLUBackward returned error: %v", err)
	}
	if err = scope.EncodeSoftmaxRowsBackward(input, outputGradient, softmax, 2, 3); err != nil {
		t.Fatalf("EncodeSoftmaxRowsBackward returned error: %v", err)
	}
	if err = scope.EncodeColumnSums(outputGradient, columnSums, 2, 3, false); err != nil {
		t.Fatalf("EncodeColumnSums returned error: %v", err)
	}
	if err = scope.EncodeColumnSums(outputGradient, accumulated, 2, 3, true); err != nil {
		t.Fatalf("EncodeColumnSums accumulated returned error: %v", err)
	}
	if err = scope.EncodeAddScaled(input, outputGradient, added, 0.5); err != nil {
		t.Fatalf("EncodeAddScaled returned error: %v", err)
	}
	if err = scope.Commit(); err != nil {
		t.Fatalf("Commit returned error: %v", err)
	}
	if err = scope.Wait(); err != nil {
		t.Fatalf("Wait returned error: %v", err)
	}

	got = make([]float32, 6)
	if err = relu.Download(got); err != nil {
		t.Fatalf("Download ReLU backward returned error: %v", err)
	}
	requireMetalValuesAlmostEqual(t, got, []float32{0, 0, 3, 4, 0, 6}, 0)
	if err = softmax.Download(got); err != nil {
		t.Fatalf("Download Softmax backward returned error: %v", err)
	}
	requireMetalValuesAlmostEqual(
		t,
		got,
		[]float32{-0.075693, -0.09156, 0.167253, -0.208216, -0.004467, 0.212683},
		2e-5,
	)
	if err = added.Download(got); err != nil {
		t.Fatalf("Download scaled addition returned error: %v", err)
	}
	requireMetalValuesAlmostEqual(t, got, []float32{-0.5, 1, 3.5, 3, 0.5, 6}, 0)

	got = make([]float32, 3)
	if err = columnSums.Download(got); err != nil {
		t.Fatalf("Download column sums returned error: %v", err)
	}
	requireMetalValuesAlmostEqual(t, got, []float32{5, 7, 9}, 0)
	if err = accumulated.Download(got); err != nil {
		t.Fatalf("Download accumulated column sums returned error: %v", err)
	}
	requireMetalValuesAlmostEqual(t, got, []float32{15, 27, 39}, 0)
}

func Test_MetalScopeRetainsReleasedBuffer(t *testing.T) {
	var (
		runtime *Runtime
		buffer  *Buffer
		scope   *Scope
		err     error
	)

	runtime = requireMetalRuntime(t)
	if buffer, err = runtime.NewBuffer(32); err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}
	if err = scope.EncodeFill(buffer, 4); err != nil {
		t.Fatalf("EncodeFill returned error: %v", err)
	}
	buffer.Release()
	if err = scope.Commit(); err != nil {
		t.Fatalf("Commit after buffer release returned error: %v", err)
	}
	if err = scope.Wait(); err != nil {
		t.Fatalf("Wait after buffer release returned error: %v", err)
	}
	if err = scope.Release(); err != nil {
		t.Fatalf("Release returned error: %v", err)
	}
}

func Test_MetalRuntimeRepeatedReuse(t *testing.T) {
	const repetitions = 256

	var (
		runtime    *Runtime
		buffer     *Buffer
		scope      *Scope
		got        []float32
		repetition int
		err        error
	)

	runtime = requireMetalRuntime(t)
	if buffer, err = runtime.NewBuffer(64); err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	defer buffer.Release()
	for repetition = 0; repetition < repetitions; repetition++ {
		if scope, err = runtime.NewScope(); err != nil {
			t.Fatalf("NewScope repetition %d returned error: %v", repetition, err)
		}
		if err = scope.EncodeFill(buffer, float32(repetition)); err != nil {
			t.Fatalf("EncodeFill repetition %d returned error: %v", repetition, err)
		}
		if err = scope.Commit(); err != nil {
			t.Fatalf("Commit repetition %d returned error: %v", repetition, err)
		}
		if err = scope.Wait(); err != nil {
			t.Fatalf("Wait repetition %d returned error: %v", repetition, err)
		}
		if err = scope.Release(); err != nil {
			t.Fatalf("Release repetition %d returned error: %v", repetition, err)
		}
	}

	got = make([]float32, 64)
	if err = buffer.Download(got); err != nil {
		t.Fatalf("Download returned error: %v", err)
	}
	requireAllFloat32(t, got, repetitions-1)
}

func Test_MetalRuntimeIndependentScopes(t *testing.T) {
	const workers = 8

	var (
		runtime   *Runtime
		waitGroup sync.WaitGroup
		errorsOut chan error
		worker    int
		err       error
	)

	runtime = requireMetalRuntime(t)
	errorsOut = make(chan error, workers)
	for worker = 0; worker < workers; worker++ {
		waitGroup.Add(1)
		go runMetalScopeWorker(runtime, worker, &waitGroup, errorsOut)
	}
	waitGroup.Wait()
	close(errorsOut)
	for err = range errorsOut {
		if err != nil {
			t.Fatalf("independent scope returned error: %v", err)
		}
	}
}

func Test_MetalRuntimeFaults(t *testing.T) {
	var (
		runtime      *Runtime
		backendValue *metalBackend
		buffer       *Buffer
		scope        *Scope
		ok           bool
		err          error
	)

	runtime = requireMetalRuntime(t)
	if backendValue, ok = runtime.backend.(*metalBackend); !ok {
		t.Fatalf("backend type = %T, want *metalBackend", runtime.backend)
	}

	if _, err = backendValue.bufferHandle(nil); err == nil {
		t.Fatal("bufferHandle(nil) returned nil error")
	}
	if _, err = backendValue.scopeHandle(nil); err == nil {
		t.Fatal("scopeHandle(nil) returned nil error")
	}
	if err = backendValue.testMissingKernel("nn_deliberately_missing_kernel"); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("missing-kernel error = %v, want not found", err)
	}
	if err = backendValue.testCompileSource("kernel void broken("); err == nil || !strings.Contains(err.Error(), "compile") {
		t.Fatalf("compilation error = %v, want compile failure", err)
	}

	if buffer, err = runtime.NewBuffer(4); err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	defer buffer.Release()
	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}
	if err = scope.EncodeFill(buffer, 1); err != nil {
		t.Fatalf("EncodeFill returned error: %v", err)
	}
	if err = backendValue.testFailScope(scope.handle); err != nil {
		t.Fatalf("testFailScope returned error: %v", err)
	}
	if err = scope.Commit(); err != nil {
		t.Fatalf("Commit returned error: %v", err)
	}
	if err = scope.Wait(); err == nil || !strings.Contains(err.Error(), "injected failure") {
		t.Fatalf("Wait error = %v, want injected failure", err)
	}
	if err = scope.Release(); err != nil {
		t.Fatalf("Release after reported failure returned error: %v", err)
	}
}

func Test_MetalRuntimeResourcesRemainBounded(t *testing.T) {
	const repetitions = 512

	var (
		runtime    *Runtime
		buffer     *Buffer
		scope      *Scope
		snapshot   ResourceSnapshot
		repetition int
		err        error
	)

	runtime = requireMetalRuntime(t)
	if err = runtime.ResetResourcePeaks(); err != nil {
		t.Fatalf("ResetResourcePeaks returned error: %v", err)
	}
	for repetition = 0; repetition < repetitions; repetition++ {
		if buffer, err = runtime.NewBuffer(16); err != nil {
			t.Fatalf("NewBuffer repetition %d returned error: %v", repetition, err)
		}
		if scope, err = runtime.NewScope(); err != nil {
			buffer.Release()
			t.Fatalf("NewScope repetition %d returned error: %v", repetition, err)
		}
		if err = scope.EncodeFill(buffer, float32(repetition)); err != nil {
			buffer.Release()
			scope.Release()
			t.Fatalf("EncodeFill repetition %d returned error: %v", repetition, err)
		}
		if err = scope.Commit(); err != nil {
			buffer.Release()
			scope.Release()
			t.Fatalf("Commit repetition %d returned error: %v", repetition, err)
		}
		if err = scope.Wait(); err != nil {
			buffer.Release()
			scope.Release()
			t.Fatalf("Wait repetition %d returned error: %v", repetition, err)
		}
		if err = scope.Release(); err != nil {
			buffer.Release()
			t.Fatalf("Release repetition %d returned error: %v", repetition, err)
		}
		buffer.Release()
	}

	snapshot = runtime.ResourceSnapshot()
	if snapshot.LiveBuffers != 0 || snapshot.LiveBufferBytes != 0 || snapshot.LiveScopes != 0 {
		t.Fatalf("live resources after stress = %+v", snapshot)
	}
	if snapshot.PeakBuffers != 1 || snapshot.PeakScopes != 1 || snapshot.PeakBufferBytes != 64 {
		t.Fatalf("peak resources after stress = %+v, want one buffer, one scope, and 64 bytes", snapshot)
	}
	if snapshot.CreatedBuffers != repetitions || snapshot.ReleasedBuffers != repetitions {
		t.Fatalf("buffer resources after stress = %+v, want %d balanced buffers", snapshot, repetitions)
	}
	if snapshot.CreatedScopes != repetitions || snapshot.ReleasedScopes != repetitions {
		t.Fatalf("scope resources after stress = %+v, want %d balanced scopes", snapshot, repetitions)
	}
	if snapshot.SubmittedCommands != repetitions || snapshot.CompletedCommands != repetitions {
		t.Fatalf("command resources after stress = %+v, want %d completed commands", snapshot, repetitions)
	}
}

func Benchmark_MetalRuntime(b *testing.B) {
	var (
		runtime *Runtime
		buffer  *Buffer
		scope   *Scope
		values  []float32
		index   int
		err     error
	)

	runtime = requireMetalRuntime(b)
	values = make([]float32, 1024)

	b.Run("ColdBufferAndScope", func(b *testing.B) {
		b.ReportAllocs()
		for index = 0; index < b.N; index++ {
			if buffer, err = runtime.NewBuffer(1024); err != nil {
				b.Fatalf("NewBuffer returned error: %v", err)
			}
			if err = buffer.Upload(values); err != nil {
				b.Fatalf("Upload returned error: %v", err)
			}
			if scope, err = runtime.NewScope(); err != nil {
				b.Fatalf("NewScope returned error: %v", err)
			}
			if err = scope.EncodeFill(buffer, 1); err != nil {
				b.Fatalf("EncodeFill returned error: %v", err)
			}
			if err = scope.Commit(); err != nil {
				b.Fatalf("Commit returned error: %v", err)
			}
			if err = scope.Wait(); err != nil {
				b.Fatalf("Wait returned error: %v", err)
			}
			if err = buffer.Download(values); err != nil {
				b.Fatalf("Download returned error: %v", err)
			}
			if err = scope.Release(); err != nil {
				b.Fatalf("Release scope returned error: %v", err)
			}
			buffer.Release()
		}
	})

	b.Run("WarmBufferReuse", func(b *testing.B) {
		if buffer, err = runtime.NewBuffer(1024); err != nil {
			b.Fatalf("NewBuffer returned error: %v", err)
		}
		defer buffer.Release()
		if err = buffer.Upload(values); err != nil {
			b.Fatalf("Upload returned error: %v", err)
		}

		b.ReportAllocs()
		b.ResetTimer()
		for index = 0; index < b.N; index++ {
			if scope, err = runtime.NewScope(); err != nil {
				b.Fatalf("NewScope returned error: %v", err)
			}
			if err = scope.EncodeFill(buffer, float32(index)); err != nil {
				b.Fatalf("EncodeFill returned error: %v", err)
			}
			if err = scope.Commit(); err != nil {
				b.Fatalf("Commit returned error: %v", err)
			}
			if err = scope.Wait(); err != nil {
				b.Fatalf("Wait returned error: %v", err)
			}
			if err = scope.Release(); err != nil {
				b.Fatalf("Release scope returned error: %v", err)
			}
		}
	})
}

func requireMetalRuntime(tb testing.TB) (runtime *Runtime) {
	tb.Helper()

	var (
		available bool
		err       error
	)

	runtime, available, err = SharedRuntime()
	if err != nil {
		tb.Fatalf("SharedRuntime returned error: %v", err)
	}
	if !available {
		tb.Skip("Metal device unavailable")
	}

	return runtime
}

func requireAllFloat32(tb testing.TB, values []float32, want float32) {
	tb.Helper()

	var (
		index int
		value float32
	)

	for index, value = range values {
		if value != want {
			tb.Fatalf("value %d = %g, want %g", index, value, want)
		}
	}
}

func requireMetalValuesAlmostEqual(tb testing.TB, got, want []float32, epsilon float32) {
	tb.Helper()

	var index int
	if len(got) != len(want) {
		tb.Fatalf("value length = %d, want %d", len(got), len(want))
	}
	for index = range want {
		if float32(math.Abs(float64(got[index]-want[index]))) > epsilon {
			tb.Fatalf("value %d = %g, want %g", index, got[index], want[index])
		}
	}
}

func runMetalScopeWorker(runtime *Runtime, worker int, waitGroup *sync.WaitGroup, errorsOut chan<- error) {
	defer waitGroup.Done()

	var (
		buffer *Buffer
		scope  *Scope
		values []float32
		err    error
	)

	if buffer, err = runtime.NewBuffer(128); err != nil {
		errorsOut <- err
		return
	}
	defer buffer.Release()
	if scope, err = runtime.NewScope(); err != nil {
		errorsOut <- err
		return
	}
	defer scope.Release()
	if err = scope.EncodeFill(buffer, float32(worker)); err != nil {
		errorsOut <- err
		return
	}
	if err = scope.Commit(); err != nil {
		errorsOut <- err
		return
	}
	if err = scope.Wait(); err != nil {
		errorsOut <- err
		return
	}
	values = make([]float32, 128)
	if err = buffer.Download(values); err != nil {
		errorsOut <- err
		return
	}
	var value float32
	for _, value = range values {
		if value != float32(worker) {
			errorsOut <- fmt.Errorf("Metal worker %d value = %g, want %g", worker, value, float32(worker))
			return
		}
	}
	errorsOut <- nil
}
