//go:build (!arm64 && !amd64) || purego

package matrix

func addInto(left, right, result []float64) {
	addIntoPure(left, right, result)
}

func addScaledInPlace(left, right []float64, scale float64) {
	addScaledInPlacePure(left, right, scale)
}

func subtractInto(left, right, result []float64) {
	subtractIntoPure(left, right, result)
}

func multiplyElementsInto(left, right, result []float64) {
	multiplyElementsIntoPure(left, right, result)
}

func addScalarInto(source []float64, value float64, result []float64) {
	addScalarIntoPure(source, value, result)
}

func multiplyScalarInto(source []float64, value float64, result []float64) {
	multiplyScalarIntoPure(source, value, result)
}

func multiplyScalarInPlace(source []float64, value float64) {
	multiplyScalarInPlacePure(source, value)
}
