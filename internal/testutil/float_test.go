package testutil

import (
	"math"
	"testing"
)

func Test_AlmostEqual(t *testing.T) {
	type testcase struct {
		name    string
		a       float64
		b       float64
		epsilon float64
		want    bool
	}

	tests := []testcase{
		{
			name:    "exact equality",
			a:       1,
			b:       1,
			epsilon: 0,
			want:    true,
		},
		{
			name:    "within tolerance",
			a:       1,
			b:       1.00001,
			epsilon: 0.0001,
			want:    true,
		},
		{
			name:    "outside tolerance",
			a:       1,
			b:       1.001,
			epsilon: 0.0001,
			want:    false,
		},
		{
			name:    "negative tolerance",
			a:       1,
			b:       1,
			epsilon: -0.0001,
			want:    false,
		},
		{
			name:    "nan is not equal",
			a:       math.NaN(),
			b:       math.NaN(),
			epsilon: 0.0001,
			want:    false,
		},
		{
			name:    "matching infinity",
			a:       math.Inf(1),
			b:       math.Inf(1),
			epsilon: 0,
			want:    true,
		},
		{
			name:    "opposite infinity",
			a:       math.Inf(1),
			b:       math.Inf(-1),
			epsilon: 0.0001,
			want:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got bool
			got = AlmostEqual(tt.a, tt.b, tt.epsilon)

			if got != tt.want {
				t.Fatalf("AlmostEqual(%g, %g, %g) = %t, want %t", tt.a, tt.b, tt.epsilon, got, tt.want)
			}
		})
	}
}

func Test_RequireAlmostEqual(t *testing.T) {
	RequireAlmostEqual(t, 1, 1.00001, 0.0001)
}

func Test_RequireSliceAlmostEqual(t *testing.T) {
	type testcase struct {
		name    string
		got     []float64
		want    []float64
		epsilon float64
	}

	tests := []testcase{
		{
			name:    "empty slices",
			got:     []float64{},
			want:    []float64{},
			epsilon: 0,
		},
		{
			name:    "matching slices",
			got:     []float64{1, 2.00001, 3},
			want:    []float64{1, 2, 3},
			epsilon: 0.0001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			RequireSliceAlmostEqual(t, tt.got, tt.want, tt.epsilon)
		})
	}
}
