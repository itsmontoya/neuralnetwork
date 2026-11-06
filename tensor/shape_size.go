package tensor

import "fmt"

func validateShapeSize(shape []int64, dataType DataType) (elements, bytes int64, err error) {
	var (
		byteWidth    int64
		maxHostIndex int64
		dimension    int64
		axis         int
		empty        bool
	)

	if byteWidth, err = dataTypeByteSize(dataType); err != nil {
		return 0, 0, err
	}
	maxHostIndex = int64(int(^uint(0) >> 1))
	for axis, dimension = range shape {
		if dimension < 0 || dimension > maxHostIndex {
			err = fmt.Errorf("tensor: dimension cannot be a host index: axis=%d dimension=%d", axis, dimension)
			return 0, 0, err
		}
		if dimension == 0 {
			empty = true
		}
	}
	if empty {
		return 0, 0, nil
	}

	elements = 1
	for axis, dimension = range shape {
		if elements > maxHostIndex/dimension {
			err = fmt.Errorf("tensor: element count overflows host index: axis=%d dimension=%d", axis, dimension)
			return 0, 0, err
		}
		elements *= dimension
	}
	if elements > maxHostIndex/byteWidth {
		err = fmt.Errorf("tensor: byte count overflows host index: elements=%d byteWidth=%d", elements, byteWidth)
		return 0, 0, err
	}
	bytes = elements * byteWidth
	return elements, bytes, nil
}
