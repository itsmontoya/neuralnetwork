package matrix

import (
	"math"
	"testing"
)

const dotProductEpsilon = 1e-4

func Test_DotProduct(t *testing.T) {
	type testcase struct {
		name  string
		left  []float32
		right []float32
	}

	tests := []testcase{
		{
			name:  "empty",
			left:  []float32{},
			right: []float32{},
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
			left:  []float32{1, float32(float32(math.Inf(1))), 2, float32(float32(math.NaN()))},
			right: []float32{2, 3, float32(float32(math.Inf(-1))), 4},
		},
	}

	var (
		tt   testcase
		got  float32
		want float32
	)

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			got = dotProduct(tt.left, tt.right)
			want = dotProductPure(tt.left, tt.right)
			requireDotProductEqual(t, got, want)
		})
	}
}

func dotProductTestValues(length int, offset float32) (values []float32) {
	var index int

	values = make([]float32, length)
	for index = range values {
		values[index] = offset + float32(index%17) - 8
	}

	return values
}

func requireDotProductEqual(tb testing.TB, got, want float32) {
	tb.Helper()

	if math.IsNaN(float64(want)) {
		if !math.IsNaN(float64(got)) {
			tb.Fatalf("dot product = %g, want NaN", got)
		}

		return
	}

	if got == want {
		return
	}

	if float32(math.Abs(float64(got-want))) <= dotProductEpsilon {
		return
	}

	tb.Fatalf(
		"dot product = %g, want %g, epsilon %g, diff %g",
		got,
		want,
		dotProductEpsilon,
		float32(math.Abs(float64(got-want))),
	)
}
