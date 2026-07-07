package matrix

import "testing"

var benchmarkElementwiseValuesSink []float64

func Benchmark_ElementwiseCandidates(b *testing.B) {
	type testcase struct {
		name string
		rows int
		cols int
	}

	tests := []testcase{
		{name: "Small1x1", rows: 1, cols: 1},
		{name: "Small1x2", rows: 1, cols: 2},
		{name: "Small1x3", rows: 1, cols: 3},
		{name: "Small2x2", rows: 2, cols: 2},
		{name: "Medium256x256", rows: 256, cols: 256},
		{name: "Large1024x1024", rows: 1024, cols: 1024},
		{name: "Uneven17x19", rows: 17, cols: 19},
		{name: "Uneven255x257", rows: 255, cols: 257},
	}

	var tt testcase
	for _, tt = range tests {
		b.Run(tt.name, func(b *testing.B) {
			var length int

			length = tt.rows * tt.cols
			b.Run("AddInto", func(b *testing.B) {
				benchmarkElementwiseBinaryInto(b, length, addIntoPure, addInto)
			})
			b.Run("SubtractInto", func(b *testing.B) {
				benchmarkElementwiseBinaryInto(b, length, subtractIntoPure, subtractInto)
			})
			b.Run("MultiplyElementsInto", func(b *testing.B) {
				benchmarkElementwiseBinaryInto(b, length, multiplyElementsIntoPure, multiplyElementsInto)
			})
			b.Run("AddScaledInPlace", func(b *testing.B) {
				benchmarkElementwiseAddScaledInPlace(b, length)
			})
			b.Run("AddScalarInto", func(b *testing.B) {
				benchmarkElementwiseAddScalarInto(b, length)
			})
			b.Run("MultiplyScalarInto", func(b *testing.B) {
				benchmarkElementwiseMultiplyScalarInto(b, length)
			})
			b.Run("MultiplyScalarInPlace", func(b *testing.B) {
				benchmarkElementwiseMultiplyScalarInPlace(b, length)
			})
		})
	}
}

func benchmarkElementwiseBinaryInto(
	b *testing.B,
	length int,
	pure func(left, right, result []float64),
	active func(left, right, result []float64),
) {
	b.Run("Pure", func(b *testing.B) {
		benchmarkElementwiseBinaryIntoFunc(b, length, pure)
	})
	b.Run("Active", func(b *testing.B) {
		benchmarkElementwiseBinaryIntoFunc(b, length, active)
	})
}

func benchmarkElementwiseBinaryIntoFunc(
	b *testing.B,
	length int,
	fn func(left, right, result []float64),
) {
	var (
		left   []float64
		right  []float64
		result []float64
		index  int
	)

	left = benchmarkElementwiseValues(length, 0.25)
	right = benchmarkElementwiseValues(length, -0.75)
	result = make([]float64, length)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		fn(left, right, result)
	}

	benchmarkElementwiseValuesSink = result
}

func benchmarkElementwiseAddScaledInPlace(b *testing.B, length int) {
	b.Run("Pure", func(b *testing.B) {
		benchmarkElementwiseAddScaledInPlaceFunc(b, length, addScaledInPlacePure)
	})
	b.Run("Active", func(b *testing.B) {
		benchmarkElementwiseAddScaledInPlaceFunc(b, length, addScaledInPlace)
	})
}

func benchmarkElementwiseAddScaledInPlaceFunc(
	b *testing.B,
	length int,
	fn func(left, right []float64, scale float64),
) {
	var (
		left  []float64
		right []float64
		index int
	)

	left = benchmarkElementwiseValues(length, 0.25)
	right = benchmarkElementwiseValues(length, -0.75)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		fn(left, right, 0.125)
	}

	benchmarkElementwiseValuesSink = left
}

func benchmarkElementwiseAddScalarInto(b *testing.B, length int) {
	b.Run("Pure", func(b *testing.B) {
		benchmarkElementwiseScalarIntoFunc(b, length, -0.375, addScalarIntoPure)
	})
	b.Run("Active", func(b *testing.B) {
		benchmarkElementwiseScalarIntoFunc(b, length, -0.375, addScalarInto)
	})
}

func benchmarkElementwiseMultiplyScalarInto(b *testing.B, length int) {
	b.Run("Pure", func(b *testing.B) {
		benchmarkElementwiseScalarIntoFunc(b, length, 1.125, multiplyScalarIntoPure)
	})
	b.Run("Active", func(b *testing.B) {
		benchmarkElementwiseScalarIntoFunc(b, length, 1.125, multiplyScalarInto)
	})
}

func benchmarkElementwiseScalarIntoFunc(
	b *testing.B,
	length int,
	value float64,
	fn func(source []float64, value float64, result []float64),
) {
	var (
		source []float64
		result []float64
		index  int
	)

	source = benchmarkElementwiseValues(length, 0.25)
	result = make([]float64, length)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		fn(source, value, result)
	}

	benchmarkElementwiseValuesSink = result
}

func benchmarkElementwiseMultiplyScalarInPlace(b *testing.B, length int) {
	b.Run("Pure", func(b *testing.B) {
		benchmarkElementwiseMultiplyScalarInPlaceFunc(b, length, multiplyScalarInPlacePure)
	})
	b.Run("Active", func(b *testing.B) {
		benchmarkElementwiseMultiplyScalarInPlaceFunc(b, length, multiplyScalarInPlace)
	})
}

func benchmarkElementwiseMultiplyScalarInPlaceFunc(
	b *testing.B,
	length int,
	fn func(source []float64, value float64),
) {
	var (
		source []float64
		index  int
	)

	source = benchmarkElementwiseValues(length, 0.25)

	b.ReportAllocs()
	b.ResetTimer()

	for index = 0; index < b.N; index++ {
		fn(source, 0.999999)
	}

	benchmarkElementwiseValuesSink = source
}

func benchmarkElementwiseValues(length int, offset float64) (values []float64) {
	var index int

	values = make([]float64, length)
	for index = range values {
		values[index] = offset + float64(index%31)/31
	}

	return values
}
