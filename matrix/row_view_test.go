package matrix_test

import (
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_MatrixWithRowViewReturnsContiguousRows(t *testing.T) {
	type testcase struct {
		name       string
		start      int
		end        int
		wantRows   int
		wantValues []float32
	}

	var (
		source *matrix.Matrix
		tests  []testcase
	)

	source = mustMatrix(t, 4, 3, []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
	})
	tests = []testcase{
		{
			name:       "single row",
			start:      2,
			end:        3,
			wantRows:   1,
			wantValues: []float32{7, 8, 9},
		},
		{
			name:       "full window",
			start:      0,
			end:        4,
			wantRows:   4,
			wantValues: []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		},
		{
			name:       "partial final window",
			start:      3,
			end:        4,
			wantRows:   1,
			wantValues: []float32{10, 11, 12},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				retained *matrix.Matrix
				err      error
			)

			err = source.WithRowView(tt.start, tt.end, func(view *matrix.Matrix) (err error) {
				retained = view
				if view.Rows() != tt.wantRows || view.Cols() != 3 {
					t.Fatalf("view shape = %dx%d, want %dx3", view.Rows(), view.Cols(), tt.wantRows)
				}
				requireMatrixValues(t, view, tt.wantValues)
				return nil
			})
			if err != nil {
				t.Fatalf("WithRowView returned error: %v", err)
			}
			if retained == nil {
				t.Fatal("WithRowView did not publish a view")
			}
			if err = retained.Validate(); err == nil {
				t.Fatal("retained view Validate error = nil, want expired view error")
			}
			if retained.Rows() != 0 || retained.Cols() != 0 {
				t.Fatalf("expired view shape = %dx%d, want 0x0", retained.Rows(), retained.Cols())
			}
		})
	}
}

func Test_MatrixWithRowViewSupportsNestedReadOnlyWindows(t *testing.T) {
	var (
		source        *matrix.Matrix
		retainedOuter *matrix.Matrix
		retainedInner *matrix.Matrix
		err           error
	)

	source = mustMatrix(t, 4, 2, []float32{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	})
	err = source.WithRowView(1, 4, func(outer *matrix.Matrix) (err error) {
		retainedOuter = outer
		requireMatrixValues(t, outer, []float32{3, 4, 5, 6, 7, 8})

		err = outer.WithRowView(1, 3, func(inner *matrix.Matrix) (err error) {
			retainedInner = inner
			requireMatrixValues(t, inner, []float32{5, 6, 7, 8})
			return nil
		})
		if err != nil {
			return err
		}
		if err = retainedInner.Validate(); err == nil {
			t.Fatal("nested retained view Validate error = nil, want expired view error")
		}
		requireMatrixValues(t, outer, []float32{3, 4, 5, 6, 7, 8})
		return nil
	})
	if err != nil {
		t.Fatalf("outer WithRowView returned error: %v", err)
	}
	if err = retainedOuter.Validate(); err == nil {
		t.Fatal("outer retained view Validate error = nil, want expired view error")
	}
}

func Test_MatrixWithRowViewExpiresAfterCallbackErrorAndPanic(t *testing.T) {
	var (
		callbackErr error
		source      *matrix.Matrix
		retained    *matrix.Matrix
		err         error
	)

	callbackErr = errors.New("injected callback failure")
	source = mustMatrix(t, 2, 1, []float32{1, 2})
	err = source.WithRowView(0, 1, func(view *matrix.Matrix) (err error) {
		retained = view
		return callbackErr
	})
	if !errors.Is(err, callbackErr) {
		t.Fatalf("WithRowView error = %v, want %v", err, callbackErr)
	}
	if err = retained.Validate(); err == nil {
		t.Fatal("error-retained view Validate error = nil, want expired view error")
	}

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("WithRowView panic = nil, want injected panic")
			}
		}()

		err = source.WithRowView(1, 2, func(view *matrix.Matrix) (err error) {
			retained = view
			panic("injected panic")
		})
	}()
	if err = retained.Validate(); err == nil {
		t.Fatal("panic-retained view Validate error = nil, want expired view error")
	}
}

func Test_MatrixWithRowViewRejectsInvalidCallsBeforeCallback(t *testing.T) {
	type testcase struct {
		name   string
		source *matrix.Matrix
		start  int
		end    int
		useNil bool
	}

	var (
		source *matrix.Matrix
		tests  []testcase
	)

	source = mustMatrix(t, 2, 2, []float32{1, 2, 3, 4})
	tests = []testcase{
		{name: "nil receiver", source: nil, start: 0, end: 1},
		{name: "zero receiver", source: &matrix.Matrix{}, start: 0, end: 1},
		{name: "negative start", source: source, start: -1, end: 1},
		{name: "empty window", source: source, start: 1, end: 1},
		{name: "reversed window", source: source, start: 1, end: 0},
		{name: "end out of range", source: source, start: 0, end: 3},
		{name: "nil callback", source: source, start: 0, end: 1, useNil: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				called bool
				use    matrix.RowViewFunc
				err    error
			)

			if !tt.useNil {
				use = func(*matrix.Matrix) (err error) {
					called = true
					return nil
				}
			}
			err = tt.source.WithRowView(tt.start, tt.end, use)
			if err == nil {
				t.Fatal("WithRowView error = nil, want error")
			}
			if !strings.HasPrefix(err.Error(), "matrix: row view") {
				t.Fatalf("WithRowView error = %q, want matrix row-view context", err)
			}
			if called {
				t.Fatal("WithRowView invoked callback for invalid call")
			}
		})
	}
}

func Test_MatrixWithRowViewCloneRemainsIndependent(t *testing.T) {
	var (
		source *matrix.Matrix
		clone  *matrix.Matrix
		err    error
	)

	source = mustMatrix(t, 3, 1, []float32{1, 2, 3})
	err = source.WithRowView(1, 3, func(view *matrix.Matrix) (viewErr error) {
		clone, viewErr = view.Clone()
		return viewErr
	})
	if err != nil {
		t.Fatalf("WithRowView returned error: %v", err)
	}
	if err = source.Set(1, 0, 99); err != nil {
		t.Fatalf("source Set returned error: %v", err)
	}
	requireMatrixValues(t, clone, []float32{2, 3})
}

func Test_MatrixWithRowViewSupportsDistinctMatrixConcurrency(t *testing.T) {
	var (
		first     *matrix.Matrix
		second    *matrix.Matrix
		waitGroup sync.WaitGroup
		errorsOut chan error
		run       func(*matrix.Matrix)
		err       error
	)

	first = mustMatrix(t, 2, 1, []float32{1, 2})
	second = mustMatrix(t, 2, 1, []float32{3, 4})
	errorsOut = make(chan error, 2)
	run = func(source *matrix.Matrix) {
		defer waitGroup.Done()
		errorsOut <- source.WithRowView(0, 2, func(view *matrix.Matrix) (viewErr error) {
			_, viewErr = view.At(1, 0)
			return viewErr
		})
	}

	waitGroup.Add(2)
	go run(first)
	go run(second)
	waitGroup.Wait()
	close(errorsOut)
	for err = range errorsOut {
		if err != nil {
			t.Fatalf("concurrent distinct WithRowView returned error: %v", err)
		}
	}
}
