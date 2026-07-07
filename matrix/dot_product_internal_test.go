package matrix

import (
	"math"
	"testing"
)

const dotProductEpsilon = 1e-12

func Test_DotProduct(t *testing.T) {
	type testcase struct {
		name  string
		left  []float64
		right []float64
	}

	tests := []testcase{
		{
			name:  "empty",
			left:  []float64{},
			right: []float64{},
		},
		{
			name:  "length one",
			left:  dotProductTestValues(1, 0.25),
			right: dotProductTestValues(1, -0.75),
		},
		{
			name:  "below vector width",
			left:  dotProductTestValues(3, 0.25),
			right: dotProductTestValues(3, -0.75),
		},
		{
			name:  "vector width",
			left:  dotProductTestValues(4, 0.25),
			right: dotProductTestValues(4, -0.75),
		},
		{
			name:  "scalar tail",
			left:  dotProductTestValues(5, 0.25),
			right: dotProductTestValues(5, -0.75),
		},
		{
			name:  "multiple vectors",
			left:  dotProductTestValues(64, 0.25),
			right: dotProductTestValues(64, -0.75),
		},
		{
			name:  "uneven tail",
			left:  dotProductTestValues(257, 0.25),
			right: dotProductTestValues(257, -0.75),
		},
		{
			name:  "inf and nan",
			left:  []float64{1, math.Inf(1), 2, math.NaN()},
			right: []float64{2, 3, math.Inf(-1), 4},
		},
	}

	var (
		tt   testcase
		got  float64
		want float64
	)

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			got = dotProduct(tt.left, tt.right)
			want = dotProductPure(tt.left, tt.right)
			requireDotProductEqual(t, got, want)
		})
	}
}

func dotProductTestValues(length int, offset float64) (values []float64) {
	var index int

	values = make([]float64, length)
	for index = range values {
		values[index] = offset + float64(index%17) - 8
	}

	return values
}

func requireDotProductEqual(tb testing.TB, got, want float64) {
	tb.Helper()

	if math.IsNaN(want) {
		if !math.IsNaN(got) {
			tb.Fatalf("dot product = %g, want NaN", got)
		}

		return
	}

	if got == want {
		return
	}

	if math.Abs(got-want) <= dotProductEpsilon {
		return
	}

	tb.Fatalf(
		"dot product = %g, want %g, epsilon %g, diff %g",
		got,
		want,
		dotProductEpsilon,
		math.Abs(got-want),
	)
}
