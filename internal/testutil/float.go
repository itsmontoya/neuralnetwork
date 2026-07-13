package testutil

import (
	"math"
	"testing"
)

// AlmostEqual reports whether a and b differ by no more than epsilon.
func AlmostEqual(a, b, epsilon float32) (ok bool) {
	if epsilon < 0 {
		return false
	}

	if a == b {
		return true
	}

	if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) {
		return false
	}

	ok = float32(math.Abs(float64(a-b))) <= epsilon
	return ok
}

// RequireAlmostEqual fails tb when got and want differ by more than epsilon.
func RequireAlmostEqual(tb testing.TB, got, want, epsilon float32) {
	tb.Helper()

	if AlmostEqual(got, want, epsilon) {
		return
	}

	tb.Fatalf(
		"values differ: got %g, want %g, epsilon %g, diff %g",
		got,
		want,
		epsilon,
		float32(math.Abs(float64(got-want))),
	)
}

// RequireSliceAlmostEqual fails tb when got and want differ in length or values.
func RequireSliceAlmostEqual(tb testing.TB, got, want []float32, epsilon float32) {
	tb.Helper()

	if len(got) != len(want) {
		tb.Fatalf("slice lengths differ: got %d, want %d", len(got), len(want))
	}

	var index int
	for index = range got {
		if AlmostEqual(got[index], want[index], epsilon) {
			continue
		}

		tb.Fatalf(
			"slice values differ at index %d: got %g, want %g, epsilon %g, diff %g",
			index,
			got[index],
			want[index],
			epsilon,
			float32(math.Abs(float64(got[index]-want[index]))),
		)
	}
}
