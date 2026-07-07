//go:build amd64 && !purego

package matrix

const elementwiseSIMDMinLength = 16

//go:noescape
func addIntoAMD64(left, right, result []float64)

//go:noescape
func addScaledInPlaceAMD64(left, right []float64, scale float64)

//go:noescape
func subtractIntoAMD64(left, right, result []float64)

//go:noescape
func addScalarIntoAMD64(source []float64, value float64, result []float64)

func addInto(left, right, result []float64) {
	if len(left) < elementwiseSIMDMinLength {
		addIntoPure(left, right, result)
		return
	}

	addIntoAMD64(left, right, result)
}

func addScaledInPlace(left, right []float64, scale float64) {
	if len(left) < elementwiseSIMDMinLength {
		addScaledInPlacePure(left, right, scale)
		return
	}

	addScaledInPlaceAMD64(left, right, scale)
}

func subtractInto(left, right, result []float64) {
	if len(left) < elementwiseSIMDMinLength {
		subtractIntoPure(left, right, result)
		return
	}

	subtractIntoAMD64(left, right, result)
}

func multiplyElementsInto(left, right, result []float64) {
	multiplyElementsIntoPure(left, right, result)
}

func addScalarInto(source []float64, value float64, result []float64) {
	if len(source) < elementwiseSIMDMinLength {
		addScalarIntoPure(source, value, result)
		return
	}

	addScalarIntoAMD64(source, value, result)
}

func multiplyScalarInto(source []float64, value float64, result []float64) {
	multiplyScalarIntoPure(source, value, result)
}

func multiplyScalarInPlace(source []float64, value float64) {
	multiplyScalarInPlacePure(source, value)
}
