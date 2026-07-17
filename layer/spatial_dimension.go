package layer

import "fmt"

func calculateSpatialOutputDimension(
	context string,
	inputSize, extentSize, strideSize, paddingSize int,
) (outputSize int, err error) {
	var (
		maxInt          int
		paddedInputSize int
	)

	if inputSize <= 0 {
		err = fmt.Errorf("layer: %s input dimension must be positive: got=%d want=>0", context, inputSize)
		return 0, err
	}

	if extentSize <= 0 {
		err = fmt.Errorf("layer: %s extent must be positive: got=%d want=>0", context, extentSize)
		return 0, err
	}

	if strideSize <= 0 {
		err = fmt.Errorf("layer: %s stride must be positive: got=%d want=>0", context, strideSize)
		return 0, err
	}

	if paddingSize < 0 {
		err = fmt.Errorf("layer: %s padding must be non-negative: got=%d want=>=0", context, paddingSize)
		return 0, err
	}

	maxInt = int(^uint(0) >> 1)
	if paddingSize > (maxInt-inputSize)/2 {
		err = fmt.Errorf(
			"layer: %s padded input dimension overflows int: input=%d padding=%d",
			context,
			inputSize,
			paddingSize,
		)
		return 0, err
	}

	paddedInputSize = inputSize + 2*paddingSize
	if extentSize > paddedInputSize {
		err = fmt.Errorf(
			"layer: %s exceeds padded input dimension: got=%d want<=%d",
			context,
			extentSize,
			paddedInputSize,
		)
		return 0, err
	}

	outputSize = (paddedInputSize-extentSize)/strideSize + 1
	return outputSize, nil
}

func checkedProduct3(context string, first, second, third int) (product int, err error) {
	var (
		maxInt       int
		firstProduct int
	)

	maxInt = int(^uint(0) >> 1)
	if first > maxInt/second {
		err = fmt.Errorf("layer: %s overflows int: factors=%dx%dx%d", context, first, second, third)
		return 0, err
	}

	firstProduct = first * second
	if firstProduct > maxInt/third {
		err = fmt.Errorf("layer: %s overflows int: factors=%dx%dx%d", context, first, second, third)
		return 0, err
	}

	product = firstProduct * third
	return product, nil
}
