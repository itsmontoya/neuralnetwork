package device

import (
	"errors"
	"strings"
	"testing"
)

func Test_ExecutionBatchesOrderedCommands(t *testing.T) {
	var (
		backendValue *testBackend
		runtimeValue *Runtime
		execution    *Execution
		source       *Buffer
		destination  *Buffer
		snapshot     ExecutionSnapshot
		resources    ResourceSnapshot
		values       []float32
		publications int
		err          error
	)

	backendValue = newTestBackend()
	runtimeValue = newRuntime(backendValue)
	execution = NewExecution(runtimeValue)
	if source, err = runtimeValue.NewBuffer(2); err != nil {
		t.Fatalf("NewBuffer source returned error: %v", err)
	}
	defer source.Release()
	if destination, err = runtimeValue.NewBuffer(2); err != nil {
		t.Fatalf("NewBuffer destination returned error: %v", err)
	}
	defer destination.Release()

	if err = execution.EncodeFill(source, 2, 0, countingPublication(&publications)); err != nil {
		t.Fatalf("EncodeFill source returned error: %v", err)
	}
	if err = execution.EncodeCopy(source, destination, 0, countingPublication(&publications)); err != nil {
		t.Fatalf("EncodeCopy returned error: %v", err)
	}
	if err = execution.EncodeFill(source, 7, 0, countingPublication(&publications)); err != nil {
		t.Fatalf("EncodeFill source replacement returned error: %v", err)
	}
	if err = execution.Finish(); err != nil {
		t.Fatalf("Finish returned error: %v", err)
	}

	values = make([]float32, 2)
	if err = destination.Download(values); err != nil {
		t.Fatalf("Download destination returned error: %v", err)
	}
	requireDeviceValues(t, values, []float32{2, 2})
	if err = source.Download(values); err != nil {
		t.Fatalf("Download source returned error: %v", err)
	}
	requireDeviceValues(t, values, []float32{7, 7})
	if publications != 3 {
		t.Fatalf("published writes = %d, want 3", publications)
	}

	snapshot = execution.Snapshot()
	if snapshot.KernelEncodes != 3 || snapshot.CommandSubmissions != 1 ||
		snapshot.Waits != 1 || snapshot.Publications != 3 {
		t.Fatalf("execution snapshot = %+v, want three encodes and one submission/wait", snapshot)
	}
	resources = runtimeValue.ResourceSnapshot()
	if resources.LiveScopes != 0 || resources.CreatedScopes != resources.ReleasedScopes {
		t.Fatalf("scope resources after Finish = %+v, want balanced scopes", resources)
	}
}

func Test_ExecutionBarrierStartsNewCommandScope(t *testing.T) {
	var (
		runtimeValue *Runtime
		execution    *Execution
		buffer       *Buffer
		snapshot     ExecutionSnapshot
		published    int
		err          error
	)

	runtimeValue = newRuntime(newTestBackend())
	execution = NewExecution(runtimeValue)
	if buffer, err = runtimeValue.NewBuffer(1); err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	defer buffer.Release()

	if err = execution.EncodeFill(buffer, 3, 0, countingPublication(&published)); err != nil {
		t.Fatalf("first EncodeFill returned error: %v", err)
	}
	if err = execution.Barrier(BoundaryCPUFallback); err != nil {
		t.Fatalf("Barrier returned error: %v", err)
	}
	if err = execution.EncodeFill(buffer, 5, 0, countingPublication(&published)); err != nil {
		t.Fatalf("second EncodeFill returned error: %v", err)
	}
	if err = execution.Finish(); err != nil {
		t.Fatalf("Finish returned error: %v", err)
	}

	snapshot = execution.Snapshot()
	if snapshot.CommandSubmissions != 2 || snapshot.Waits != 2 ||
		snapshot.Barriers != 2 || snapshot.FallbackBarriers != 1 {
		t.Fatalf("execution snapshot = %+v, want two scopes and one fallback barrier", snapshot)
	}
}

func Test_ExecutionRotatesAtKernelLimit(t *testing.T) {
	var (
		runtimeValue *Runtime
		execution    *Execution
		buffer       *Buffer
		snapshot     ExecutionSnapshot
		published    int
		index        int
		err          error
	)

	runtimeValue = newRuntime(newTestBackend())
	execution = NewExecution(runtimeValue)
	if buffer, err = runtimeValue.NewBuffer(1); err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	defer buffer.Release()

	for index = 0; index < executionKernelLimit+1; index++ {
		if err = execution.EncodeFill(buffer, float32(index), 0, countingPublication(&published)); err != nil {
			t.Fatalf("EncodeFill %d returned error: %v", index, err)
		}
	}
	if err = execution.Finish(); err != nil {
		t.Fatalf("Finish returned error: %v", err)
	}

	snapshot = execution.Snapshot()
	if snapshot.KernelEncodes != executionKernelLimit+1 ||
		snapshot.CommandSubmissions != 2 || snapshot.Waits != 2 {
		t.Fatalf("bounded execution snapshot = %+v, want 65 encodes in two scopes", snapshot)
	}
	if published != executionKernelLimit+1 {
		t.Fatalf("published writes = %d, want %d", published, executionKernelLimit+1)
	}
}

func Test_ExecutionFailureDiscardsPendingWrites(t *testing.T) {
	var (
		backendValue *testBackend
		runtimeValue *Runtime
		execution    *Execution
		buffer       *Buffer
		snapshot     ExecutionSnapshot
		resources    ResourceSnapshot
		published    int
		discarded    int
		err          error
	)

	backendValue = newTestBackend()
	backendValue.waitErr = errors.New("injected command failure")
	runtimeValue = newRuntime(backendValue)
	execution = NewExecution(runtimeValue)
	if buffer, err = runtimeValue.NewBuffer(1); err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	defer buffer.Release()

	if err = execution.EncodeFill(buffer, 9, 0, trackedPublication(&published, &discarded)); err != nil {
		t.Fatalf("EncodeFill returned error: %v", err)
	}
	if err = execution.Finish(); err == nil || !strings.Contains(err.Error(), "injected command failure") {
		t.Fatalf("Finish error = %v, want injected command failure", err)
	}
	if published != 0 || discarded != 1 {
		t.Fatalf("publication counts = published %d discarded %d, want 0/1", published, discarded)
	}

	snapshot = execution.Snapshot()
	if snapshot.Publications != 0 || snapshot.DiscardedWrites != 1 || snapshot.Waits != 1 {
		t.Fatalf("failed execution snapshot = %+v", snapshot)
	}
	resources = runtimeValue.ResourceSnapshot()
	if resources.LiveScopes != 0 || resources.CreatedScopes != resources.ReleasedScopes {
		t.Fatalf("scope resources after failure = %+v, want balanced scopes", resources)
	}
}

func Test_ExecutionCleanupFailureDiscardsAtomicUpdate(t *testing.T) {
	var (
		backendValue *testBackend
		runtimeValue *Runtime
		execution    *Execution
		buffer       *Buffer
		snapshot     ExecutionSnapshot
		published    int
		discarded    int
		err          error
	)

	backendValue = newTestBackend()
	runtimeValue = newRuntime(backendValue)
	execution = NewExecution(runtimeValue)
	if buffer, err = runtimeValue.NewBuffer(1); err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	defer buffer.Release()
	if err = execution.EncodeFill(
		buffer,
		9,
		0,
		trackedPublication(&published, &discarded),
	); err != nil {
		t.Fatalf("EncodeFill returned error: %v", err)
	}
	backendValue.releaseErr = errors.New("injected cleanup failure")
	if err = execution.Finish(); err == nil ||
		!strings.Contains(err.Error(), "injected cleanup failure") {
		t.Fatalf("Finish error = %v, want injected cleanup failure", err)
	}
	if published != 0 || discarded != 1 {
		t.Fatalf(
			"publication counts = published %d discarded %d, want 0/1",
			published,
			discarded,
		)
	}

	snapshot = execution.Snapshot()
	if snapshot.Publications != 0 || snapshot.DiscardedWrites != 1 {
		t.Fatalf("failed cleanup execution snapshot = %+v", snapshot)
	}
}

func Test_ExecutionTrainingPhaseFailuresRespectUpdateBoundary(t *testing.T) {
	type testcase struct {
		name              string
		run               func(*testBackend, *Execution, *Buffer, Publication) error
		wantLossPublished int
		wantDiscarded     int
	}

	injected := errors.New("injected training phase failure")
	tests := []testcase{
		{
			name: "loss synchronization",
			run: func(
				backendValue *testBackend,
				execution *Execution,
				buffer *Buffer,
				loss Publication,
			) (err error) {
				if err = execution.EncodeFill(buffer, 1, 0, loss); err != nil {
					return err
				}
				backendValue.waitErr = injected
				err = execution.Barrier(BoundaryHostObservation)
				return err
			},
			wantDiscarded: 1,
		},
		{
			name: "loss download",
			run: func(
				backendValue *testBackend,
				execution *Execution,
				buffer *Buffer,
				loss Publication,
			) (err error) {
				if err = execution.EncodeFill(buffer, 1, 0, loss); err != nil {
					return err
				}
				if err = execution.Barrier(BoundaryHostObservation); err != nil {
					return err
				}
				backendValue.downloadErr = injected
				err = buffer.Download(make([]float32, 1))
				return err
			},
			wantLossPublished: 1,
		},
		{
			name: "update encoding",
			run: func(
				backendValue *testBackend,
				execution *Execution,
				buffer *Buffer,
				loss Publication,
			) (err error) {
				if err = execution.EncodeFill(buffer, 1, 0, loss); err != nil {
					return err
				}
				if err = execution.Barrier(BoundaryHostObservation); err != nil {
					return err
				}
				backendValue.encodeErr = injected
				err = execution.EncodeFill(buffer, 2, 0, loss)
				return err
			},
			wantLossPublished: 1,
			wantDiscarded:     1,
		},
		{
			name: "update execution",
			run: func(
				backendValue *testBackend,
				execution *Execution,
				buffer *Buffer,
				loss Publication,
			) (err error) {
				if err = execution.EncodeFill(buffer, 1, 0, loss); err != nil {
					return err
				}
				if err = execution.Barrier(BoundaryHostObservation); err != nil {
					return err
				}
				if err = execution.EncodeFill(buffer, 2, 0, loss); err != nil {
					return err
				}
				backendValue.waitErr = injected
				err = execution.Finish()
				return err
			},
			wantLossPublished: 1,
			wantDiscarded:     1,
		},
		{
			name: "update cleanup",
			run: func(
				backendValue *testBackend,
				execution *Execution,
				buffer *Buffer,
				loss Publication,
			) (err error) {
				if err = execution.EncodeFill(buffer, 1, 0, loss); err != nil {
					return err
				}
				if err = execution.Barrier(BoundaryHostObservation); err != nil {
					return err
				}
				if err = execution.EncodeFill(buffer, 2, 0, loss); err != nil {
					return err
				}
				backendValue.releaseErr = injected
				err = execution.Finish()
				return err
			},
			wantLossPublished: 1,
			wantDiscarded:     1,
		},
	}

	var test testcase
	for _, test = range tests {
		t.Run(test.name, func(t *testing.T) {
			var (
				backendValue *testBackend
				runtimeValue *Runtime
				execution    *Execution
				buffer       *Buffer
				publication  Publication
				published    int
				discarded    int
				abortErr     error
				err          error
			)

			backendValue = newTestBackend()
			runtimeValue = newRuntime(backendValue)
			execution = NewExecution(runtimeValue)
			if buffer, err = runtimeValue.NewBuffer(1); err != nil {
				t.Fatalf("NewBuffer returned error: %v", err)
			}
			defer buffer.Release()
			publication = trackedPublication(&published, &discarded)
			err = test.run(backendValue, execution, buffer, publication)
			if err == nil || !strings.Contains(err.Error(), injected.Error()) {
				t.Fatalf("%s error = %v, want injected failure", test.name, err)
			}
			if execution.Active() {
				if abortErr = execution.Abort(err); abortErr != nil {
					t.Fatalf("%s Abort returned error: %v", test.name, abortErr)
				}
			}
			if published != test.wantLossPublished ||
				discarded != test.wantDiscarded {
				t.Fatalf(
					"%s publications = %d discarded = %d, want %d/%d",
					test.name,
					published,
					discarded,
					test.wantLossPublished,
					test.wantDiscarded,
				)
			}
		})
	}
}

func Test_ExecutionAbortDiscardsUncommittedWrites(t *testing.T) {
	var (
		runtimeValue *Runtime
		execution    *Execution
		buffer       *Buffer
		resources    ResourceSnapshot
		published    int
		discarded    int
		err          error
	)

	runtimeValue = newRuntime(newTestBackend())
	execution = NewExecution(runtimeValue)
	if buffer, err = runtimeValue.NewBuffer(1); err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	defer buffer.Release()
	if err = execution.EncodeFill(buffer, 4, 0, trackedPublication(&published, &discarded)); err != nil {
		t.Fatalf("EncodeFill returned error: %v", err)
	}
	if err = execution.Abort(errors.New("injected early error")); err != nil {
		t.Fatalf("Abort returned cleanup error: %v", err)
	}
	if published != 0 || discarded != 1 {
		t.Fatalf("publication counts = published %d discarded %d, want 0/1", published, discarded)
	}
	resources = runtimeValue.ResourceSnapshot()
	if resources.SubmittedCommands != 0 || resources.LiveScopes != 0 {
		t.Fatalf("resources after Abort = %+v, want no submission or live scope", resources)
	}
}

func Test_ExecutionResetClearsCompletedCallState(t *testing.T) {
	var (
		runtimeValue *Runtime
		execution    *Execution
		buffer       *Buffer
		snapshot     ExecutionSnapshot
		published    int
		err          error
	)

	runtimeValue = newRuntime(newTestBackend())
	execution = NewExecution(runtimeValue)
	if buffer, err = runtimeValue.NewBuffer(1); err != nil {
		t.Fatalf("NewBuffer returned error: %v", err)
	}
	defer buffer.Release()
	if err = execution.EncodeFill(buffer, 1, 0, countingPublication(&published)); err != nil {
		t.Fatalf("first EncodeFill returned error: %v", err)
	}
	if err = execution.Finish(); err != nil {
		t.Fatalf("first Finish returned error: %v", err)
	}
	if err = execution.Reset(runtimeValue); err != nil {
		t.Fatalf("Reset returned error: %v", err)
	}
	if err = execution.EncodeFill(buffer, 2, 0, countingPublication(&published)); err != nil {
		t.Fatalf("second EncodeFill returned error: %v", err)
	}
	if err = execution.Finish(); err != nil {
		t.Fatalf("second Finish returned error: %v", err)
	}

	snapshot = execution.Snapshot()
	if snapshot.KernelEncodes != 1 || snapshot.CommandSubmissions != 1 || snapshot.Waits != 1 {
		t.Fatalf("reset execution snapshot = %+v, want only the second call", snapshot)
	}
	if published != 2 {
		t.Fatalf("published writes = %d, want 2 across both calls", published)
	}
}

func countingPublication(published *int) (publication Publication) {
	var discarded int
	publication = trackedPublication(published, &discarded)
	return publication
}

func trackedPublication(published, discarded *int) (publication Publication) {
	publication.Publish = func() (err error) {
		*published++
		return nil
	}
	publication.Discard = func(error) (err error) {
		*discarded++
		return nil
	}
	return publication
}

func requireDeviceValues(tb testing.TB, got, want []float32) {
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
