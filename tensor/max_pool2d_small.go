package tensor

import "github.com/itsmontoya/neuralnetwork/internal/f32"

// maxPool2DSmallWindowsInto removes geometry for complete, unpadded 2x2/3x3 windows.
// The caller validates geometry and returns early for empty output, so every
// row span and plane offset below fits the checked, nonempty input payload.
func maxPool2DSmallWindowsInto(input, destination *Tensor, config Pool2DConfig) (handled bool) {
	if config.Dilations != [2]int64{1, 1} || config.Pads != [4]int64{} || config.CeilMode {
		return false
	}
	if config.Kernel != [2]int64{2, 2} && config.Kernel != [2]int64{3, 3} {
		return false
	}
	var (
		inputWidth = int(input.shape[3])
		inputPlane = int(input.shape[2]) * inputWidth
		height     = int(destination.shape[2])
		width      = int(destination.shape[3])
		strideH    = int(config.Strides[0])
		strideW    = int(config.Strides[1])
		offset     int
	)

	for plane := 0; plane < int(input.shape[0]*input.shape[1]); plane++ {
		for row := 0; row < height; row++ {
			var (
				start  = plane*inputPlane + row*strideH*inputWidth
				first  = input.floats[start : start+inputWidth]
				second = input.floats[start+inputWidth : start+2*inputWidth]
			)

			if config.Kernel[0] == 2 {
				for col := 0; col < width; col++ {
					var (
						column = col * strideW
						value  = first[column]
					)

					value = maxPoolWindowMaximum(value, first[column+1])
					value = maxPoolWindowMaximum(value, second[column])
					value = maxPoolWindowMaximum(value, second[column+1])
					destination.floats[offset] = value
					offset++
				}
				continue
			}

			third := input.floats[start+2*inputWidth : start+3*inputWidth]
			for col := 0; col < width; col++ {
				var (
					column = col * strideW
					value  = first[column]
				)

				value = maxPoolWindowMaximum(value, first[column+1])
				value = maxPoolWindowMaximum(value, first[column+2])
				value = maxPoolWindowMaximum(value, second[column])
				value = maxPoolWindowMaximum(value, second[column+1])
				value = maxPoolWindowMaximum(value, second[column+2])
				value = maxPoolWindowMaximum(value, third[column])
				value = maxPoolWindowMaximum(value, third[column+1])
				value = maxPoolWindowMaximum(value, third[column+2])
				destination.floats[offset] = value
				offset++
			}
		}
	}
	return true
}

func maxPoolWindowMaximum(value, sample float32) (maximum float32) {
	// Preserve the first equal value (including zero's sign) and the last NaN's bits.
	if sample > value || f32.IsNaN(sample) {
		return sample
	}
	return value
}
