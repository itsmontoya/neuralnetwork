//go:build purego || (!arm64 && !amd64)

package matrix

func addScaledInPlace(left, right []float32, scale float32) {
	addScaledInPlacePure(left, right, scale)
}

func addScalarInto(source []float32, value float32, result []float32) {
	addScalarIntoPure(source, value, result)
}

func multiplyScalarInto(source []float32, value float32, result []float32) {
	multiplyScalarIntoPure(source, value, result)
}

func multiplyScalarInPlace(source []float32, value float32) {
	multiplyScalarInPlacePure(source, value)
}
