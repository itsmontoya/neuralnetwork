package tensor

import "fmt"

func validateTensorType(name string, value *Tensor, dataType DataType) (err error) {
	if err = value.Validate(); err != nil {
		return fmt.Errorf("tensor: %s is invalid: %w", name, err)
	}
	if value.dataType != dataType {
		err = fmt.Errorf("tensor: %s has data type %d, want %d", name, value.dataType, dataType)
		return err
	}
	return nil
}

// validateDestination requires already validated inputs; nil optional inputs
// are ignored. Storage identities detect shallow-copy aliases, including empties.
func validateDestination(destination *Tensor, dataType DataType, shape []int64, inputs ...*Tensor) (err error) {
	var (
		index int
		input *Tensor
	)

	if err = validateTensorType("destination", destination, dataType); err != nil {
		return err
	}
	if !shapesEqual(destination.shape, shape) {
		// Formatting a local copy keeps successful callers' shape scratch on the stack.
		expected := make([]int64, len(shape))
		copy(expected, shape)
		err = fmt.Errorf("tensor: destination shape is %v, want %v", destination.shape, expected)
		return err
	}
	for index, input = range inputs {
		if input != nil && destination.identity == input.identity {
			err = fmt.Errorf("tensor: destination aliases input %d", index)
			return err
		}
	}
	return nil
}
