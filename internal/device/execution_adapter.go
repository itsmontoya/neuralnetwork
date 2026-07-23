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
	Bind      func(value any, execution *Execution) (key any, err error)
	Execution func(value any) (execution *Execution, err error)
	Unbind    func(key any, execution *Execution) (err error)
	Record    func(snapshot ExecutionSnapshot)
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
