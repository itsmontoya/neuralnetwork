package testutil

import (
	"math"
	"testing"
)

// AlmostEqual reports whether a and b differ by no more than epsilon.
func AlmostEqual(a, b, epsilon float64) (ok bool) {
	if epsilon < 0 {
		return false
	}

	if a == b {
		return true
	}

	if math.IsNaN(a) || math.IsNaN(b) {
		return false
	}

	ok = math.Abs(a-b) <= epsilon
	return ok
}

// RequireAlmostEqual fails tb when got and want differ by more than epsilon.
func RequireAlmostEqual(tb testing.TB, got, want, epsilon float64) {
	tb.Helper()

	if AlmostEqual(got, want, epsilon) {
		return
	}

	tb.Fatalf(
		"values differ: got %g, want %g, epsilon %g, diff %g",
		got,
		want,
		epsilon,
		math.Abs(got-want),
	)
}

// RequireSliceAlmostEqual fails tb when got and want differ in length or values.
func RequireSliceAlmostEqual(tb testing.TB, got, want []float64, epsilon float64) {
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
			math.Abs(got[index]-want[index]),
		)
	}
}
