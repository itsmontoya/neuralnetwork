package matrix

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

func init() {
	var adapter device.ExecutionAdapter
	var err error

	adapter.Bind = bindMatrixExecution
	adapter.Execution = matrixExecutionValue
	adapter.Unbind = unbindMatrixExecution
	adapter.ReLUForward = forwardReLUExecution
	adapter.ReLUBackward = backwardReLUExecution
	adapter.Reset = resetMatrixExecution
	adapter.Record = recordExecutionActivity
	if err = device.RegisterExecutionAdapter(adapter); err != nil {
		panic(fmt.Sprintf("matrix: register device execution adapter: %v", err))
	}
}

func forwardReLUExecution(input, output any) (handled bool, err error) {
	var (
		inputMatrix  *Matrix
		outputMatrix *Matrix
		ok           bool
	)

	if inputMatrix, ok = input.(*Matrix); !ok {
		err = fmt.Errorf("matrix: ReLU input has type %T, want *matrix.Matrix", input)
		return false, err
	}
	if outputMatrix, ok = output.(*Matrix); !ok {
		err = fmt.Errorf("matrix: ReLU output has type %T, want *matrix.Matrix", output)
		return false, err
	}
	if err = inputMatrix.validate(); err != nil {
		return false, err
	}
	if err = outputMatrix.requireShape("destination", inputMatrix.rows, inputMatrix.cols); err != nil {
		return false, err
	}
	if err = inheritExecution(outputMatrix, inputMatrix); err != nil {
		return false, err
	}

	handled, err = reluForwardDevice(inputMatrix, outputMatrix)
	return handled, err
}

func backwardReLUExecution(
	input,
	outputGradient,
	inputGradient any,
) (handled bool, err error) {
	var (
		inputMatrix          *Matrix
		outputGradientMatrix *Matrix
		inputGradientMatrix  *Matrix
		ok                   bool
	)

	if inputMatrix, ok = input.(*Matrix); !ok {
		err = fmt.Errorf("matrix: ReLU input has type %T, want *matrix.Matrix", input)
		return false, err
	}
	if outputGradientMatrix, ok = outputGradient.(*Matrix); !ok {
		err = fmt.Errorf("matrix: ReLU output gradient has type %T, want *matrix.Matrix", outputGradient)
		return false, err
	}
	if inputGradientMatrix, ok = inputGradient.(*Matrix); !ok {
		err = fmt.Errorf("matrix: ReLU input gradient has type %T, want *matrix.Matrix", inputGradient)
		return false, err
	}
	if err = inputMatrix.validate(); err != nil {
		return false, err
	}
	if err = outputGradientMatrix.requireShape(
		"output gradient",
		inputMatrix.rows,
		inputMatrix.cols,
	); err != nil {
		return false, err
	}
	if err = inputGradientMatrix.requireShape(
		"destination",
		inputMatrix.rows,
		inputMatrix.cols,
	); err != nil {
		return false, err
	}
	if inputGradientMatrix == outputGradientMatrix {
		err = errors.New("matrix: ReLU input gradient must not alias output gradient")
		return false, err
	}
	if err = inheritExecution(inputGradientMatrix, inputMatrix, outputGradientMatrix); err != nil {
		return false, err
	}

	handled, err = reluBackwardDevice(inputMatrix, outputGradientMatrix, inputGradientMatrix)
	return handled, err
}

func resetMatrixExecution(value any) (handled bool, err error) {
	var matrixValue *Matrix
	var ok bool

	if matrixValue, ok = value.(*Matrix); !ok {
		err = fmt.Errorf("matrix: reset value has type %T, want *matrix.Matrix", value)
		return false, err
	}
	if err = matrixValue.validate(); err != nil {
		return false, err
	}

	handled, err = resetDevice(matrixValue)
	return handled, err
}

func bindMatrixExecution(value any, execution *device.Execution) (key any, err error) {
	var matrixValue *Matrix
	var ok bool

	if matrixValue, ok = value.(*Matrix); !ok {
		err = fmt.Errorf("matrix: execution value has type %T, want *matrix.Matrix", value)
		return nil, err
	}
	if matrixValue == nil {
		err = errors.New("matrix: bind execution: matrix is nil")
		return nil, err
	}
	if execution == nil || execution.Runtime() == nil {
		err = errors.New("matrix: bind execution: execution is nil")
		return nil, err
	}
	if err = matrixValue.ensureResidency(execution.Runtime()); err != nil {
		return nil, err
	}
	if err = matrixValue.residency.BindExecution(execution); err != nil {
		return nil, fmt.Errorf("matrix: bind execution: %w", err)
	}

	key = matrixValue
	return key, nil
}

func matrixExecutionValue(value any) (execution *device.Execution, err error) {
	var matrixValue *Matrix
	var ok bool

	if matrixValue, ok = value.(*Matrix); !ok {
		err = fmt.Errorf("matrix: execution value has type %T, want *matrix.Matrix", value)
		return nil, err
	}
	if matrixValue == nil || matrixValue.residency == nil {
		return nil, nil
	}

	execution = matrixValue.residency.Execution()
	return execution, nil
}

func unbindMatrixExecution(key any, execution *device.Execution) (err error) {
	var matrixValue *Matrix
	var ok bool

	if matrixValue, ok = key.(*Matrix); !ok {
		err = fmt.Errorf("matrix: execution binding key has type %T, want *matrix.Matrix", key)
		return err
	}
	if matrixValue == nil || matrixValue.residency == nil {
		return nil
	}

	if err = matrixValue.residency.UnbindExecution(execution); err != nil {
		return fmt.Errorf("matrix: unbind execution: %w", err)
	}
	return nil
}

func (m *Matrix) execution() (execution *device.Execution) {
	if m == nil || m.residency == nil {
		return nil
	}

	execution = m.residency.Execution()
	return execution
}

func compatibleExecution(values ...*Matrix) (execution *device.Execution, err error) {
	var (
		value   *Matrix
		current *device.Execution
	)

	for _, value = range values {
		current = value.execution()
		if current == nil {
			continue
		}
		if !current.Active() {
			err = errors.New("matrix: value is bound to a closed execution")
			return nil, err
		}
		if execution != nil && execution != current {
			err = errors.New("matrix: operands belong to different executions")
			return nil, err
		}
		execution = current
	}
	return execution, nil
}

func inheritExecution(destination *Matrix, sources ...*Matrix) (err error) {
	var (
		execution *device.Execution
		current   *device.Execution
		source    *Matrix
	)

	if execution = destination.execution(); execution != nil && !execution.Active() {
		err = errors.New("matrix: destination is bound to a closed execution")
		return err
	}
	for _, source = range sources {
		current = source.execution()
		if current == nil {
			continue
		}
		if !current.Active() {
			err = errors.New("matrix: source is bound to a closed execution")
			return err
		}
		if execution != nil && execution != current {
			err = errors.New("matrix: operands belong to different executions")
			return err
		}
		execution = current
	}
	if execution == nil {
		return nil
	}

	err = execution.Bind(destination)
	return err
}

func (m *Matrix) executionBarrier(reason device.Boundary) (err error) {
	var execution *device.Execution

	execution = m.execution()
	if execution == nil {
		return nil
	}
	if err = execution.Barrier(reason); err != nil {
		return fmt.Errorf("matrix: device execution barrier: %w", err)
	}
	return nil
}
