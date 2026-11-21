//go:build !arm64 || purego || race

package float32product

const packedRightARM64 = false

func packRightTransposeTile(dense, packed []float32, inner int) {
	for contraction := range inner {
		for lane := range packedRightColumns {
			packed[contraction*packedRightColumns+lane] = dense[lane*inner+contraction]
		}
	}
}

func rightTransposePackedTile(left, packed, output []float32, inner, stride, rows int) {
	for row := range rows {
		var input = left[row*inner:][:inner]
		for lane := range packedRightColumns {
			var sum float32
			for contraction, value := range input {
				sum += value * packed[contraction*packedRightColumns+lane]
			}
			output[row*stride+lane] = sum
		}
	}
}
