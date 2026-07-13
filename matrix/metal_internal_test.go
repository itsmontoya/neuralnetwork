//go:build darwin && cgo && metal && !purego

package matrix

import (
	"math"
	"strings"
	"testing"
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
			ok    bool
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

		ok = metalRunMatMul(left, right, got, metalMatMulStandard)
		requireMetalRun(t, ok)
		matMulIntoPure(left, right, want)
		requireMetalMatrixValues(t, got, want, metalMatMulTestEpsilon)
	})

	t.Run("left transpose", func(t *testing.T) {
		var (
			left  *Matrix
			right *Matrix
			got   *Matrix
			want  *Matrix
			ok    bool
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

		ok = metalRunMatMul(left, right, got, metalMatMulLeftTranspose)
		requireMetalRun(t, ok)
		matMulLeftTransposeIntoPure(left, right, want)
		requireMetalMatrixValues(t, got, want, metalMatMulTestEpsilon)
	})

	t.Run("right transpose", func(t *testing.T) {
		var (
			left  *Matrix
			right *Matrix
			got   *Matrix
			want  *Matrix
			ok    bool
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

		ok = metalRunMatMul(left, right, got, metalMatMulRightTranspose)
		requireMetalRun(t, ok)
		matMulRightTransposeIntoPure(left, right, want)
		requireMetalMatrixValues(t, got, want, metalMatMulTestEpsilon)
	})
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

func requireMetalRun(tb testing.TB, ok bool) {
	tb.Helper()

	if ok {
		return
	}

	tb.Fatalf("Metal kernel failed: %s", metalLastError())
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
