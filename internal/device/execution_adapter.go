package device

import (
	"errors"
	"sync"
)

var (
	executionAdapterMutex sync.RWMutex
	registeredAdapter     ExecutionAdapter
	adapterRegistered     bool
)

// ExecutionAdapter connects opaque matrix values to private executions.
//
// The adapter is registered once by the matrix package during initialization.
// Per-call execution state remains on the values and is never stored globally.
type ExecutionAdapter struct {
	Bind                            func(value any, execution *Execution) (key any, err error)
	Execution                       func(value any) (execution *Execution, err error)
	Unbind                          func(key any, execution *Execution) (err error)
	ReLUForward                     func(input, output any) (handled bool, err error)
	ReLUBackward                    func(input, outputGradient, inputGradient any) (handled bool, err error)
	CategoricalCrossEntropyValue    func(predictions, targets any, epsilon float32) (value float32, handled bool, err error)
	CategoricalCrossEntropyGradient func(predictions, targets, gradient any, epsilon float32) (handled bool, err error)
	SGD                             func(updates []ParameterUpdate, learningRate float32) (handled bool, err error)
	Reset                           func(value any) (handled bool, err error)
	Record                          func(snapshot ExecutionSnapshot)
}

// RegisterExecutionAdapter installs the process-wide immutable matrix adapter.
func RegisterExecutionAdapter(adapter ExecutionAdapter) (err error) {
	if adapter.Bind == nil || adapter.Execution == nil || adapter.Unbind == nil {
		err = errors.New("device: execution adapter is incomplete")
		return err
	}

	executionAdapterMutex.Lock()
	defer executionAdapterMutex.Unlock()
	if adapterRegistered {
		err = errors.New("device: execution adapter is already registered")
		return err
	}

	registeredAdapter = adapter
	adapterRegistered = true
	return nil
}

// BoundExecution returns the active execution attached to an opaque matrix.
func BoundExecution(value any) (execution *Execution, err error) {
	var adapter ExecutionAdapter

	if value == nil {
		return nil, nil
	}
	if adapter, err = currentExecutionAdapter(); err != nil {
		return nil, err
	}

	execution, err = adapter.Execution(value)
	return execution, err
}

// ReLUForward attempts the private built-in ReLU device operation.
func ReLUForward(input, output any) (handled bool, err error) {
	var adapter ExecutionAdapter

	if adapter, err = currentExecutionAdapter(); err != nil {
		return false, err
	}
	if adapter.ReLUForward == nil {
		return false, nil
	}

	handled, err = adapter.ReLUForward(input, output)
	return handled, err
}

// ReLUBackward attempts the private built-in ReLU device operation.
func ReLUBackward(input, outputGradient, inputGradient any) (handled bool, err error) {
	var adapter ExecutionAdapter

	if adapter, err = currentExecutionAdapter(); err != nil {
		return false, err
	}
	if adapter.ReLUBackward == nil {
		return false, nil
	}

	handled, err = adapter.ReLUBackward(input, outputGradient, inputGradient)
	return handled, err
}

// CategoricalCrossEntropyValue attempts the private device-resident loss operation.
func CategoricalCrossEntropyValue(
	predictions,
	targets any,
	epsilon float32,
) (value float32, handled bool, err error) {
	var adapter ExecutionAdapter

	if adapter, err = currentExecutionAdapter(); err != nil {
		return 0, false, err
	}
	if adapter.CategoricalCrossEntropyValue == nil {
		return 0, false, nil
	}

	value, handled, err = adapter.CategoricalCrossEntropyValue(predictions, targets, epsilon)
	return value, handled, err
}

// CategoricalCrossEntropyGradient attempts the private device-resident loss-gradient operation.
func CategoricalCrossEntropyGradient(
	predictions,
	targets,
	gradient any,
	epsilon float32,
) (handled bool, err error) {
	var adapter ExecutionAdapter

	if adapter, err = currentExecutionAdapter(); err != nil {
		return false, err
	}
	if adapter.CategoricalCrossEntropyGradient == nil {
		return false, nil
	}

	handled, err = adapter.CategoricalCrossEntropyGradient(
		predictions,
		targets,
		gradient,
		epsilon,
	)
	return handled, err
}

// SGD attempts one private device-resident parameter update transaction.
func SGD(updates []ParameterUpdate, learningRate float32) (handled bool, err error) {
	var adapter ExecutionAdapter

	if adapter, err = currentExecutionAdapter(); err != nil {
		return false, err
	}
	if adapter.SGD == nil {
		return false, nil
	}

	handled, err = adapter.SGD(updates, learningRate)
	return handled, err
}

// Reset attempts a private device-resident zero fill.
func Reset(value any) (handled bool, err error) {
	var adapter ExecutionAdapter

	if adapter, err = currentExecutionAdapter(); err != nil {
		return false, err
	}
	if adapter.Reset == nil {
		return false, nil
	}

	handled, err = adapter.Reset(value)
	return handled, err
}

func currentExecutionAdapter() (adapter ExecutionAdapter, err error) {
	executionAdapterMutex.RLock()
	adapter = registeredAdapter
	var registered bool
	registered = adapterRegistered
	executionAdapterMutex.RUnlock()
	if !registered {
		err = errors.New("device: execution adapter is not registered")
		return adapter, err
	}

	return adapter, nil
}
