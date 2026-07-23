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
	adapter.Record = recordExecutionActivity
	if err = device.RegisterExecutionAdapter(adapter); err != nil {
		panic(fmt.Sprintf("matrix: register device execution adapter: %v", err))
	}
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
