//go:build (!arm64 && !amd64) || purego

package matrix

func addInto(left, right, result []float32) {
	addIntoPure(left, right, result)
}

func addScaledInPlace(left, right []float32, scale float32) {
	addScaledInPlacePure(left, right, scale)
}

func subtractInto(left, right, result []float32) {
	subtractIntoPure(left, right, result)
}

func multiplyElementsInto(left, right, result []float32) {
	multiplyElementsIntoPure(left, right, result)
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
