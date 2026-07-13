package matrix

import (
	"math"
	"testing"
)

const elementwiseEpsilon = 1e-5

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
				left        []float32
				right       []float32
				got         []float32
				want        []float32
				gotInPlace  []float32
				wantInPlace []float32
			)

			left = elementwiseTestValues(tt.length, 0.25)
			right = elementwiseTestValues(tt.length, -0.75)
			got = make([]float32, tt.length)
			want = make([]float32, tt.length)

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

func elementwiseTestValues(length int, offset float32) (values []float32) {
	var (
		index int
		base  []float32
	)

	base = []float32{
		0,
		float32(math.Copysign(0, -1)),
		1.5,
		-2.25,
		float32(float32(math.Inf(1))),
		float32(float32(math.Inf(-1))),
		float32(float32(math.NaN())),
		3.75,
		-4.5,
	}

	values = make([]float32, length)
	for index = range values {
		values[index] = base[index%len(base)]
		if values[index] == 0 ||
			math.IsInf(float64(values[index]), 0) ||
			math.IsNaN(float64(values[index])) {
			continue
		}

		values[index] += offset * float32(index%5)
	}

	return values
}

func cloneElementwiseValues(values []float32) (clone []float32) {
	clone = make([]float32, len(values))
	copy(clone, values)
	return clone
}

func requireElementwiseValues(tb testing.TB, got, want []float32) {
	tb.Helper()

	if len(got) != len(want) {
		tb.Fatalf("values length = %d, want %d", len(got), len(want))
	}

	var index int
	for index = range want {
		if math.IsNaN(float64(want[index])) {
			if !math.IsNaN(float64(got[index])) {
				tb.Fatalf("values[%d] = %g, want NaN", index, got[index])
			}

			continue
		}

		if got[index] == want[index] {
			continue
		}

		if float32(math.Abs(float64(got[index]-want[index]))) <= elementwiseEpsilon {
			continue
		}

		tb.Fatalf(
			"values[%d] = %g, want %g, epsilon %g, diff %g",
			index,
			got[index],
			want[index],
			elementwiseEpsilon,
			float32(math.Abs(float64(got[index]-want[index]))),
		)
	}
}
