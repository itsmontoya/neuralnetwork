package tensor

import "fmt"

// DataType identifies the concrete numeric storage of a Tensor.
type DataType uint8

const (
	// Invalid identifies an uninitialized or unsupported tensor type.
	Invalid DataType = iota
	// Float32 stores IEEE 754 binary32 values without conversion.
	Float32
	// Int64 stores signed 64-bit integers without conversion.
	Int64
)

func dataTypeByteSize(dataType DataType) (bytes int64, err error) {
	switch dataType {
	case Float32:
		return 4, nil
	case Int64:
		return 8, nil
	default:
		err = fmt.Errorf("tensor: unsupported data type: %d", dataType)
		return 0, err
	}
}
