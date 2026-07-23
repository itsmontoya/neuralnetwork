package device

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func Test_Float32Bytes(t *testing.T) {
	type testcase struct {
		name      string
		count     uint64
		want      uint64
		wantError string
	}

	tests := []testcase{
		{name: "one", count: 1, want: 4},
		{name: "largest valid", count: math.MaxUint64 / 4, want: math.MaxUint64 - 3},
		{name: "zero", count: 0, wantError: "must be positive"},
		{name: "overflow", count: math.MaxUint64/4 + 1, wantError: "overflow"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				got uint64
				err error
			)

			got, err = float32Bytes(tt.count)
			if tt.wantError == "" {
				if err != nil {
					t.Fatalf("float32Bytes returned error: %v", err)
				}
				if got != tt.want {
					t.Fatalf("bytes = %d, want %d", got, tt.want)
				}
				return
			}

			if err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("error = %v, want containing %q", err, tt.wantError)
			}
		})
	}
}

func Test_RuntimeAvailability(t *testing.T) {
	type testcase struct {
		name       string
		available  bool
		backendErr error
		wantError  bool
	}

	tests := []testcase{
		{name: "available", available: true},
		{name: "missing device", available: false},
		{name: "initialization failure", backendErr: errors.New("injected initialization failure"), wantError: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				backendValue *testBackend
				runtime      *Runtime
				available    bool
				err          error
			)

			backendValue = newTestBackend()
			backendValue.availableFlag = tt.available
			backendValue.availableErr = tt.backendErr
			runtime = newRuntime(backendValue)
			available, err = runtime.Available()
			if tt.wantError {
				if err == nil {
					t.Fatal("Available returned nil error")
				}
				return
			}
			if err != nil {
				t.Fatalf("Available returned error: %v", err)
			}
			if available != tt.available {
				t.Fatalf("available = %t, want %t", available, tt.available)
			}
		})
	}
}

func Test_BufferRoundTripAndRelease(t *testing.T) {
	var (
		backendValue *testBackend
		runtime      *Runtime
		buffer       *Buffer
		got          []float32
		snapshot     ResourceSnapshot
		err          error
	)

	backendValue = newTestBackend()
	runtime = newRuntime(backendValue)
	buffer, err = runtime.NewBuffer(4)
	if err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	if err = buffer.Upload([]float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("Upload returned error: %v", err)
	}
	got = make([]float32, 4)
	if err = buffer.Download(got); err != nil {
		t.Fatalf("Download returned error: %v", err)
	}
	requireFloat32Values(t, got, []float32{1, 2, 3, 4})

	buffer.Release()
	buffer.Release()
	if err = buffer.Upload([]float32{1, 2, 3, 4}); !errors.Is(err, ErrReleased) {
		t.Fatalf("Upload error = %v, want ErrReleased", err)
	}

	snapshot = runtime.ResourceSnapshot()
	if snapshot.LiveBuffers != 0 || snapshot.CreatedBuffers != 1 || snapshot.ReleasedBuffers != 1 {
		t.Fatalf("resource snapshot = %+v, want one balanced buffer", snapshot)
	}
}

func Test_ScopeOrdersCommands(t *testing.T) {
	var (
		runtime *Runtime
		source  *Buffer
		result  *Buffer
		scope   *Scope
		got     []float32
		err     error
	)

	runtime = newRuntime(newTestBackend())
	if source, err = runtime.NewBuffer(3); err != nil {
		t.Fatalf("NewBuffer source returned error: %v", err)
	}
	defer source.Release()
	if result, err = runtime.NewBuffer(3); err != nil {
		t.Fatalf("NewBuffer result returned error: %v", err)
	}
	defer result.Release()
	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}
	defer scope.Release()

	if err = scope.EncodeFill(source, 2); err != nil {
		t.Fatalf("EncodeFill returned error: %v", err)
	}
	if err = scope.EncodeCopy(source, result); err != nil {
		t.Fatalf("EncodeCopy returned error: %v", err)
	}
	if err = scope.EncodeFill(source, 7); err != nil {
		t.Fatalf("second EncodeFill returned error: %v", err)
	}
	if err = scope.Commit(); err != nil {
		t.Fatalf("Commit returned error: %v", err)
	}
	if err = scope.Wait(); err != nil {
		t.Fatalf("Wait returned error: %v", err)
	}

	got = make([]float32, 3)
	if err = result.Download(got); err != nil {
		t.Fatalf("Download result returned error: %v", err)
	}
	requireFloat32Values(t, got, []float32{2, 2, 2})
	if err = source.Download(got); err != nil {
		t.Fatalf("Download source returned error: %v", err)
	}
	requireFloat32Values(t, got, []float32{7, 7, 7})
}

func Test_ScopeDenseForwardOperations(t *testing.T) {
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

	runtime = newRuntime(newTestBackend())
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
	requireFloat32Values(t, got, []float32{-2, -3, 4, 3, 2, 9})
	if err = relu.Download(got); err != nil {
		t.Fatalf("Download ReLU returned error: %v", err)
	}
	requireFloat32Values(t, got, []float32{0, 0, 4, 3, 2, 9})
	if err = softmax.Download(got); err != nil {
		t.Fatalf("Download Softmax returned error: %v", err)
	}
	requireFloat32ValuesAlmostEqual(
		t,
		got,
		[]float32{0.017668422, 0.017668422, 0.96466315, 0.002472318, 0.000909011, 0.9966187},
		2e-5,
	)
}

func Test_ScopeInvalidTransitions(t *testing.T) {
	var (
		runtime *Runtime
		buffer  *Buffer
		scope   *Scope
		err     error
	)

	runtime = newRuntime(newTestBackend())
	if buffer, err = runtime.NewBuffer(1); err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	defer buffer.Release()
	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}

	if err = scope.Wait(); !errors.Is(err, ErrInvalidState) {
		t.Fatalf("Wait before commit error = %v, want ErrInvalidState", err)
	}
	if err = scope.EncodeFill(nil, 0); err == nil || !strings.Contains(err.Error(), "nil") {
		t.Fatalf("EncodeFill nil error = %v, want nil buffer error", err)
	}
	if err = scope.EncodeFill(buffer, 1); err != nil {
		t.Fatalf("EncodeFill returned error: %v", err)
	}
	if err = scope.Commit(); err != nil {
		t.Fatalf("Commit returned error: %v", err)
	}
	if err = scope.EncodeFill(buffer, 2); !errors.Is(err, ErrInvalidState) {
		t.Fatalf("EncodeFill after commit error = %v, want ErrInvalidState", err)
	}
	if err = scope.Commit(); !errors.Is(err, ErrInvalidState) {
		t.Fatalf("second Commit error = %v, want ErrInvalidState", err)
	}
	if err = scope.Wait(); err != nil {
		t.Fatalf("Wait returned error: %v", err)
	}
	if err = scope.Release(); err != nil {
		t.Fatalf("Release returned error: %v", err)
	}
	if err = scope.Release(); err != nil {
		t.Fatalf("second Release returned error: %v", err)
	}
	if _, err = scope.Completed(); !errors.Is(err, ErrInvalidState) {
		t.Fatalf("Completed after release error = %v, want ErrInvalidState", err)
	}
}

func Test_RuntimeConstructionFailuresCleanUp(t *testing.T) {
	type testcase struct {
		name        string
		bufferError error
		scopeError  error
		construct   func(*Runtime) error
	}

	tests := []testcase{
		{
			name:        "buffer allocation",
			bufferError: errors.New("injected allocation failure"),
			construct: func(runtime *Runtime) (err error) {
				_, err = runtime.NewBuffer(1)
				return err
			},
		},
		{
			name:       "scope creation",
			scopeError: errors.New("injected scope failure"),
			construct: func(runtime *Runtime) (err error) {
				_, err = runtime.NewScope()
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				backendValue *testBackend
				runtime      *Runtime
				snapshot     ResourceSnapshot
				err          error
			)

			backendValue = newTestBackend()
			backendValue.newBufferErr = tt.bufferError
			backendValue.newScopeErr = tt.scopeError
			runtime = newRuntime(backendValue)
			if err = tt.construct(runtime); err == nil {
				t.Fatal("construction returned nil error")
			}
			snapshot = runtime.ResourceSnapshot()
			if snapshot.LiveBuffers != 0 || snapshot.LiveScopes != 0 {
				t.Fatalf("resource snapshot after failed construction = %+v", snapshot)
			}
		})
	}
}

func Test_ScopeCommandFailureReleasesResources(t *testing.T) {
	var (
		backendValue *testBackend
		runtime      *Runtime
		buffer       *Buffer
		scope        *Scope
		snapshot     ResourceSnapshot
		err          error
	)

	backendValue = newTestBackend()
	backendValue.waitErr = errors.New("injected command failure")
	runtime = newRuntime(backendValue)
	if buffer, err = runtime.NewBuffer(1); err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}
	if err = scope.EncodeFill(buffer, 1); err != nil {
		t.Fatalf("EncodeFill returned error: %v", err)
	}
	if err = scope.Commit(); err != nil {
		t.Fatalf("Commit returned error: %v", err)
	}
	if err = scope.Wait(); err == nil || !strings.Contains(err.Error(), "injected command failure") {
		t.Fatalf("Wait error = %v, want injected command failure", err)
	}
	if err = scope.Release(); err != nil {
		t.Fatalf("Release returned error after reported failure: %v", err)
	}
	buffer.Release()

	snapshot = runtime.ResourceSnapshot()
	if snapshot.LiveBuffers != 0 || snapshot.LiveScopes != 0 {
		t.Fatalf("resource snapshot after failure cleanup = %+v", snapshot)
	}
}

func requireFloat32Values(tb testing.TB, got, want []float32) {
	tb.Helper()

	var index int
	if len(got) != len(want) {
		tb.Fatalf("value length = %d, want %d", len(got), len(want))
	}
	for index = range want {
		if got[index] != want[index] {
			tb.Fatalf("value %d = %g, want %g", index, got[index], want[index])
		}
	}
}

func requireFloat32ValuesAlmostEqual(tb testing.TB, got, want []float32, epsilon float32) {
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
