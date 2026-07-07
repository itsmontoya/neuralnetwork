package matrix

import (
	"math"
	"testing"
)

const elementwiseEpsilon = 1e-12

func Test_ElementwiseKernels(t *testing.T) {
	type testcase struct {
		name   string
		length int
	}

	tests := []testcase{
		{name: "empty", length: 0},
		{name: "below vector width", length: 1},
		{name: "vector width", length: 2},
		{name: "scalar tail", length: 3},
		{name: "multiple vectors", length: 16},
		{name: "uneven tail", length: 31},
	}

	var tt testcase
	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				left        []float64
				right       []float64
				got         []float64
				want        []float64
				gotInPlace  []float64
				wantInPlace []float64
			)

			left = elementwiseTestValues(tt.length, 0.25)
			right = elementwiseTestValues(tt.length, -0.75)
			got = make([]float64, tt.length)
			want = make([]float64, tt.length)

			addInto(left, right, got)
			addIntoPure(left, right, want)
			requireElementwiseValues(t, got, want)

			subtractInto(left, right, got)
			subtractIntoPure(left, right, want)
			requireElementwiseValues(t, got, want)

			multiplyElementsInto(left, right, got)
			multiplyElementsIntoPure(left, right, want)
			requireElementwiseValues(t, got, want)

			addScalarInto(left, -0.375, got)
			addScalarIntoPure(left, -0.375, want)
			requireElementwiseValues(t, got, want)

			multiplyScalarInto(left, 1.125, got)
			multiplyScalarIntoPure(left, 1.125, want)
			requireElementwiseValues(t, got, want)

			gotInPlace = cloneElementwiseValues(left)
			wantInPlace = cloneElementwiseValues(left)
			addScaledInPlace(gotInPlace, right, -0.5)
			addScaledInPlacePure(wantInPlace, right, -0.5)
			requireElementwiseValues(t, gotInPlace, wantInPlace)

			gotInPlace = cloneElementwiseValues(left)
			wantInPlace = cloneElementwiseValues(left)
			multiplyScalarInPlace(gotInPlace, 0.875)
			multiplyScalarInPlacePure(wantInPlace, 0.875)
			requireElementwiseValues(t, gotInPlace, wantInPlace)
		})
	}
}

func elementwiseTestValues(length int, offset float64) (values []float64) {
	var (
		index int
		base  []float64
	)

	base = []float64{
		0,
		math.Copysign(0, -1),
		1.5,
		-2.25,
		math.Inf(1),
		math.Inf(-1),
		math.NaN(),
		3.75,
		-4.5,
	}

	values = make([]float64, length)
	for index = range values {
		values[index] = base[index%len(base)]
		if values[index] == 0 || math.IsInf(values[index], 0) || math.IsNaN(values[index]) {
			continue
		}

		values[index] += offset * float64(index%5)
	}

	return values
}

func cloneElementwiseValues(values []float64) (clone []float64) {
	clone = make([]float64, len(values))
	copy(clone, values)
	return clone
}

func requireElementwiseValues(tb testing.TB, got, want []float64) {
	tb.Helper()

	if len(got) != len(want) {
		tb.Fatalf("values length = %d, want %d", len(got), len(want))
	}

	var index int
	for index = range want {
		if math.IsNaN(want[index]) {
			if !math.IsNaN(got[index]) {
				tb.Fatalf("values[%d] = %g, want NaN", index, got[index])
			}

			continue
		}

		if got[index] == want[index] {
			continue
		}

		if math.Abs(got[index]-want[index]) <= elementwiseEpsilon {
			continue
		}

		tb.Fatalf(
			"values[%d] = %g, want %g, epsilon %g, diff %g",
			index,
			got[index],
			want[index],
			elementwiseEpsilon,
			math.Abs(got[index]-want[index]),
		)
	}
}
