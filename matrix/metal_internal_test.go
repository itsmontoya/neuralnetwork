//go:build darwin && cgo && metal && !purego

package matrix

import (
	"math"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
)

const (
	metalMatMulTestEpsilon = 1e-3
)

func Test_MetalMatMulKernels(t *testing.T) {
	requireMetalAvailable(t)

	t.Run("standard", func(t *testing.T) {
		var (
			left  *Matrix
			right *Matrix
			got   *Matrix
			want  *Matrix
			err   error
		)

		left = metalTestMatrix(t, 128, 256, 0.25)
		right = metalTestMatrix(t, 256, 128, -0.75)
		got, err = New(128, 128)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		want, err = New(128, 128)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}

		if err = metalRunMatMul(left, right, got, metalMatMulStandard); err != nil {
			t.Fatalf("metalRunMatMul returned error: %v", err)
		}
		matMulIntoPure(left, right, want)
		requireMetalMatrixValues(t, got, want, metalMatMulTestEpsilon)
	})

	t.Run("left transpose", func(t *testing.T) {
		var (
			left  *Matrix
			right *Matrix
			got   *Matrix
			want  *Matrix
			err   error
		)

		left = metalTestMatrix(t, 256, 128, 0.125)
		right = metalTestMatrix(t, 256, 128, -0.5)
		got, err = New(128, 128)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		want, err = New(128, 128)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}

		if err = metalRunMatMul(left, right, got, metalMatMulLeftTranspose); err != nil {
			t.Fatalf("metalRunMatMul returned error: %v", err)
		}
		matMulLeftTransposeIntoPure(left, right, want)
		requireMetalMatrixValues(t, got, want, metalMatMulTestEpsilon)
	})

	t.Run("right transpose", func(t *testing.T) {
		var (
			left  *Matrix
			right *Matrix
			got   *Matrix
			want  *Matrix
			err   error
		)

		left = metalTestMatrix(t, 128, 256, 0.375)
		right = metalTestMatrix(t, 128, 256, -0.25)
		got, err = New(128, 128)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}
		want, err = New(128, 128)
		if err != nil {
			t.Fatalf("New returned error: %v", err)
		}

		if err = metalRunMatMul(left, right, got, metalMatMulRightTranspose); err != nil {
			t.Fatalf("metalRunMatMul returned error: %v", err)
		}
		matMulRightTransposeIntoPure(left, right, want)
		requireMetalMatrixValues(t, got, want, metalMatMulTestEpsilon)
	})
}

func Test_MetalMatrixResidencyCoherence(t *testing.T) {
	var (
		left        *Matrix
		right       *Matrix
		result      *Matrix
		cpuResult   *Matrix
		clone       *Matrix
		counters    metaltest.Counters
		value       float32
		cloneValue  float32
		resultValue float32
		err         error
	)

	requireMetalAvailable(t)
	left = metalTestMatrix(t, 64, 128, 0.25)
	right = metalTestMatrix(t, 128, 128, -0.5)
	if result, err = New(64, 128); err != nil {
		t.Fatalf("New result returned error: %v", err)
	}
	if cpuResult, err = New(64, 128); err != nil {
		t.Fatalf("New CPU result returned error: %v", err)
	}

	metaltest.Enable()
	defer metaltest.Disable()
	if err = left.MatMulInto(right, result); err != nil {
		t.Fatalf("first MatMulInto returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, 3, 2, 0, 1, 1)

	if result.Rows() != 64 || result.Cols() != 128 {
		t.Fatalf("result shape = %dx%d, want 64x128", result.Rows(), result.Cols())
	}
	if err = result.Validate(); err != nil {
		t.Fatalf("device-newer Validate returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, 3, 2, 0, 1, 1)

	if err = left.MatMulInto(right, result); err != nil {
		t.Fatalf("warmed MatMulInto returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, 4, 2, 0, 2, 2)
	if _, err = result.Values(); err != nil {
		t.Fatalf("device-newer Values returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, 4, 2, 1, 2, 2)

	if err = left.Set(0, 0, 3.5); err != nil {
		t.Fatalf("Set uploaded input returned error: %v", err)
	}
	if err = left.MatMulInto(right, result); err != nil {
		t.Fatalf("MatMulInto after host mutation returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, 5, 3, 1, 3, 3)
	if value, err = result.At(0, 0); err != nil {
		t.Fatalf("At device-newer result returned error: %v", err)
	}
	if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
		t.Fatalf("At device-newer result = %g, want finite", value)
	}
	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, 5, 3, 2, 3, 3)

	if err = left.MatMulInto(right, result); err != nil {
		t.Fatalf("MatMulInto before CPU fallback returned error: %v", err)
	}
	if err = result.AddScalarInto(1, cpuResult); err != nil {
		t.Fatalf("AddScalarInto fallback returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, 6, 3, 3, 4, 4)

	if err = left.MatMulInto(right, result); err != nil {
		t.Fatalf("MatMulInto before Clone returned error: %v", err)
	}
	metaltest.Reset()
	if clone, err = result.Clone(); err != nil {
		t.Fatalf("device-newer Clone returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, 1, 0, 0, 1, 1)
	if cloneValue, err = clone.At(0, 0); err != nil {
		t.Fatalf("clone At returned error: %v", err)
	}
	if resultValue, err = result.At(0, 0); err != nil {
		t.Fatalf("result At returned error: %v", err)
	}
	requireMetalFloat(t, cloneValue, resultValue, 0)
	if err = clone.Set(0, 0, cloneValue+10); err != nil {
		t.Fatalf("clone Set returned error: %v", err)
	}
	if resultValue, err = result.At(0, 0); err != nil {
		t.Fatalf("result At after clone mutation returned error: %v", err)
	}
	requireMetalFloat(t, resultValue, cloneValue, 0)
}

func Test_MetalExecutionBatchesDependentMultiplications(t *testing.T) {
	var (
		execution    *device.Execution
		left         *Matrix
		right        *Matrix
		intermediate *Matrix
		result       *Matrix
		wantFirst    *Matrix
		want         *Matrix
		counters     metaltest.Counters
		available    bool
		err          error
	)

	requireMetalAvailable(t)
	left = metalTestMatrix(t, 64, 128, 0.25)
	right = metalTestMatrix(t, 128, 128, -0.5)
	if intermediate, err = New(64, 128); err != nil {
		t.Fatalf("New intermediate returned error: %v", err)
	}
	if result, err = New(64, 128); err != nil {
		t.Fatalf("New result returned error: %v", err)
	}
	if wantFirst, err = New(64, 128); err != nil {
		t.Fatalf("New first reference returned error: %v", err)
	}
	if want, err = New(64, 128); err != nil {
		t.Fatalf("New reference returned error: %v", err)
	}
	matMulIntoPure(left, right, wantFirst)
	matMulIntoPure(wantFirst, right, want)

	if execution, available, err = device.NewSharedExecution(); err != nil {
		t.Fatalf("NewSharedExecution returned error: %v", err)
	}
	if !available {
		t.Fatal("NewSharedExecution reported Metal unavailable after availability check")
	}
	if err = execution.Bind(left); err != nil {
		t.Fatalf("Bind input returned error: %v", err)
	}

	metaltest.Enable()
	defer metaltest.Disable()
	if err = left.MatMulInto(right, intermediate); err != nil {
		t.Fatalf("first MatMulInto returned error: %v", err)
	}
	if err = intermediate.MatMulInto(right, result); err != nil {
		t.Fatalf("dependent MatMulInto returned error: %v", err)
	}
	if counters = metaltest.Snapshot(); counters.CommandSubmissions != 0 || counters.Waits != 0 {
		t.Fatalf("counters before Finish = %+v, want no submission or wait", counters)
	}
	if err = execution.Finish(); err != nil {
		t.Fatalf("Finish returned error: %v", err)
	}

	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, 4, 2, 0, 1, 1)
	requireMetalMatrixValues(t, result, want, metalMatMulTestEpsilon)
	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, 4, 2, 1, 1, 1)
}

func Test_MetalExecutionSynchronizesCPUFallbackAndResumes(t *testing.T) {
	var (
		execution    *device.Execution
		left         *Matrix
		right        *Matrix
		intermediate *Matrix
		fallback     *Matrix
		result       *Matrix
		counters     metaltest.Counters
		available    bool
		err          error
	)

	requireMetalAvailable(t)
	left = metalTestMatrix(t, 64, 128, 0.125)
	right = metalTestMatrix(t, 128, 128, -0.25)
	if intermediate, err = New(64, 128); err != nil {
		t.Fatalf("New intermediate returned error: %v", err)
	}
	if fallback, err = New(64, 128); err != nil {
		t.Fatalf("New fallback returned error: %v", err)
	}
	if result, err = New(64, 128); err != nil {
		t.Fatalf("New result returned error: %v", err)
	}
	if execution, available, err = device.NewSharedExecution(); err != nil {
		t.Fatalf("NewSharedExecution returned error: %v", err)
	}
	if !available {
		t.Fatal("NewSharedExecution reported Metal unavailable after availability check")
	}
	if err = execution.Bind(left); err != nil {
		t.Fatalf("Bind input returned error: %v", err)
	}

	metaltest.Enable()
	defer metaltest.Disable()
	if err = left.MatMulInto(right, intermediate); err != nil {
		t.Fatalf("first MatMulInto returned error: %v", err)
	}
	if err = intermediate.AddScalarInto(1, fallback); err != nil {
		t.Fatalf("AddScalarInto fallback returned error: %v", err)
	}
	if err = fallback.MatMulInto(right, result); err != nil {
		t.Fatalf("MatMulInto after fallback returned error: %v", err)
	}
	if err = execution.Finish(); err != nil {
		t.Fatalf("Finish returned error: %v", err)
	}

	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, 5, 3, 1, 2, 2)
	if _, err = result.Values(); err != nil {
		t.Fatalf("result Values returned error: %v", err)
	}
}

func Test_MetalExecutionCompletesRepeatedDestinationWrites(t *testing.T) {
	var (
		execution *device.Execution
		left      *Matrix
		right     *Matrix
		result    *Matrix
		want      *Matrix
		counters  metaltest.Counters
		available bool
		err       error
	)

	requireMetalAvailable(t)
	left = metalTestMatrix(t, 64, 128, 0.375)
	right = metalTestMatrix(t, 128, 128, -0.125)
	if result, err = New(64, 128); err != nil {
		t.Fatalf("New result returned error: %v", err)
	}
	if want, err = New(64, 128); err != nil {
		t.Fatalf("New reference returned error: %v", err)
	}
	matMulIntoPure(left, right, want)
	if execution, available, err = device.NewSharedExecution(); err != nil {
		t.Fatalf("NewSharedExecution returned error: %v", err)
	}
	if !available {
		t.Fatal("NewSharedExecution reported Metal unavailable after availability check")
	}
	if err = execution.Bind(left); err != nil {
		t.Fatalf("Bind input returned error: %v", err)
	}

	metaltest.Enable()
	defer metaltest.Disable()
	if err = left.MatMulInto(right, result); err != nil {
		t.Fatalf("first MatMulInto returned error: %v", err)
	}
	if err = left.MatMulInto(right, result); err != nil {
		t.Fatalf("repeated MatMulInto returned error: %v", err)
	}
	if err = execution.Finish(); err != nil {
		t.Fatalf("Finish returned error: %v", err)
	}

	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, 4, 2, 0, 2, 2)
	requireMetalMatrixValues(t, result, want, metalMatMulTestEpsilon)
}

func Test_MetalMatrixResidencyHostBoundaries(t *testing.T) {
	type testcase struct {
		name      string
		downloads uint64
		run       func(*Matrix) error
	}

	var (
		left     *Matrix
		right    *Matrix
		result   *Matrix
		other    *Matrix
		values   []float32
		counters metaltest.Counters
		tests    []testcase
		test     testcase
		err      error
	)

	requireMetalAvailable(t)
	left = metalTestMatrix(t, 64, 128, 0.125)
	right = metalTestMatrix(t, 128, 128, -0.25)
	other = metalTestMatrix(t, 64, 128, 0.5)
	if result, err = New(64, 128); err != nil {
		t.Fatalf("New result returned error: %v", err)
	}
	values = make([]float32, 64*128)
	tests = []testcase{
		{
			name:      "Values",
			downloads: 1,
			run: func(input *Matrix) (err error) {
				_, err = input.Values()
				return err
			},
		},
		{
			name:      "ValuesInto",
			downloads: 1,
			run: func(input *Matrix) (err error) {
				err = input.ValuesInto(values)
				return err
			},
		},
		{
			name:      "At",
			downloads: 1,
			run: func(input *Matrix) (err error) {
				_, err = input.At(0, 0)
				return err
			},
		},
		{
			name:      "Set",
			downloads: 1,
			run: func(input *Matrix) (err error) {
				err = input.Set(0, 0, 3)
				return err
			},
		},
		{
			name: "Fill",
			run: func(input *Matrix) (err error) {
				err = input.Fill(3)
				return err
			},
		},
		{
			name: "CopyValuesFrom",
			run: func(input *Matrix) (err error) {
				err = input.CopyValuesFrom(values)
				return err
			},
		},
		{
			name:      "SelectRows",
			downloads: 1,
			run: func(input *Matrix) (err error) {
				_, err = input.SelectRows([]int{0, 3})
				return err
			},
		},
		{
			name:      "Transpose",
			downloads: 1,
			run: func(input *Matrix) (err error) {
				_, err = input.Transpose()
				return err
			},
		},
		{
			name:      "Apply",
			downloads: 1,
			run: func(input *Matrix) (err error) {
				_, err = input.Apply(func(value float32) float32 { return value + 1 })
				return err
			},
		},
		{
			name:      "Pairwise",
			downloads: 1,
			run: func(input *Matrix) (err error) {
				err = input.Pairwise(other, func(_, _ int, _, _ float32) (err error) {
					return nil
				})
				return err
			},
		},
	}

	metaltest.Enable()
	defer metaltest.Disable()
	for _, test = range tests {
		t.Run(test.name, func(t *testing.T) {
			if err = left.MatMulInto(right, result); err != nil {
				t.Fatalf("MatMulInto returned error: %v", err)
			}
			metaltest.Reset()
			if err = test.run(result); err != nil {
				t.Fatalf("host boundary returned error: %v", err)
			}
			counters = metaltest.Snapshot()
			if counters.ResultDownloads != test.downloads {
				t.Fatalf("result downloads = %d, want %d", counters.ResultDownloads, test.downloads)
			}
		})
	}
}

func Test_MetalMatrixResidencyLongReuse(t *testing.T) {
	const iterations = 128

	var (
		left     *Matrix
		right    *Matrix
		result   *Matrix
		counters metaltest.Counters
		index    int
		err      error
	)

	requireMetalAvailable(t)
	left = metalTestMatrix(t, 64, 128, 0.375)
	right = metalTestMatrix(t, 128, 128, -0.125)
	if result, err = New(64, 128); err != nil {
		t.Fatalf("New result returned error: %v", err)
	}

	metaltest.Enable()
	defer metaltest.Disable()
	for index = 0; index < iterations; index++ {
		if err = left.MatMulInto(right, result); err != nil {
			t.Fatalf("MatMulInto iteration %d returned error: %v", index, err)
		}
	}
	counters = metaltest.Snapshot()
	requireMetalCounters(t, counters, iterations+2, 2, 0, iterations, iterations)
	if _, err = result.Values(); err != nil {
		t.Fatalf("Values after long reuse returned error: %v", err)
	}
}

func requireMetalCounters(
	tb testing.TB,
	counters metaltest.Counters,
	buffers,
	uploads,
	downloads,
	commands,
	waits uint64,
) {
	tb.Helper()

	if counters.BufferCreations != buffers ||
		counters.InputUploads != uploads ||
		counters.ResultDownloads != downloads ||
		counters.CommandSubmissions != commands ||
		counters.Waits != waits {
		tb.Fatalf(
			"Metal counters = %+v, want buffers=%d uploads=%d downloads=%d commands=%d waits=%d",
			counters,
			buffers,
			uploads,
			downloads,
			commands,
			waits,
		)
	}
}

func requireMetalAvailable(tb testing.TB) {
	tb.Helper()

	if metalAvailable() {
		return
	}

	var message string
	message = metalLastError()
	if strings.Contains(message, "no default device") {
		tb.Skipf("Metal device unavailable: %s", message)
	}

	tb.Fatalf("Metal unavailable: %s", message)
}

func metalTestMatrix(tb testing.TB, rows, cols int, offset float32) (m *Matrix) {
	tb.Helper()

	var (
		values []float32
		err    error
	)

	values = metalTestValues(rows*cols, offset)
	m, err = FromSlice(rows, cols, values)
	if err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}

	return m
}

func metalTestValues(length int, offset float32) (values []float32) {
	var index int

	values = make([]float32, length)
	for index = range values {
		values[index] = offset + float32(index%31)/31
	}

	return values
}

func requireMetalFloat(tb testing.TB, got, want, epsilon float32) {
	tb.Helper()

	if got == want {
		return
	}

	if float32(math.Abs(float64(got-want))) <= epsilon {
		return
	}

	tb.Fatalf(
		"value = %g, want %g, epsilon %g, diff %g",
		got,
		want,
		epsilon,
		float32(math.Abs(float64(got-want))),
	)
}

func requireMetalMatrixValues(tb testing.TB, got, want *Matrix, epsilon float32) {
	tb.Helper()

	var (
		gotValues  []float32
		wantValues []float32
		index      int
		err        error
	)

	gotValues, err = got.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	wantValues, err = want.Values()
	if err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}

	if len(gotValues) != len(wantValues) {
		tb.Fatalf("values length = %d, want %d", len(gotValues), len(wantValues))
	}

	for index = range wantValues {
		requireMetalFloat(tb, gotValues[index], wantValues[index], epsilon)
	}
}
