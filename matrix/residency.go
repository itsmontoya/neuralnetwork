package matrix

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

func (m *Matrix) ensureHostCurrent() (err error) {
	var (
		execution  *device.Execution
		downloaded bool
	)

	if m == nil || m.residency == nil {
		return nil
	}
	if execution = m.execution(); execution != nil {
		if err = execution.Error(); err != nil {
			return fmt.Errorf("matrix: pending device execution failed: %w", err)
		}
		var pending bool
		_, pending = m.residency.PendingBuffer(execution)
		if pending {
			if err = m.executionBarrier(device.BoundaryHostObservation); err != nil {
				return err
			}
		}
	}

	if downloaded, err = m.residency.EnsureHost(m.data); err != nil {
		return fmt.Errorf("matrix: synchronize host values: %w", err)
	}
	if downloaded {
		if execution != nil {
			execution.RecordDownload(uint64(len(m.data)) * 4)
		} else {
			recordResidencyDownload(uint64(len(m.data)) * 4)
		}
	}
	return nil
}

func (m *Matrix) markHostWrite() (err error) {
	var (
		execution *device.Execution
		used      bool
	)

	if m == nil || m.residency == nil {
		return nil
	}
	if execution = m.execution(); execution != nil {
		if err = execution.Error(); err != nil {
			return fmt.Errorf("matrix: pending device execution failed: %w", err)
		}
		if _, used = m.residency.PendingBuffer(execution); !used {
			if used, err = execution.Uses(m); err != nil {
				return fmt.Errorf("matrix: inspect pending device use: %w", err)
			}
		}
		if used {
			if err = m.executionBarrier(device.BoundaryHostMutation); err != nil {
				return err
			}
		}
	}

	if err = m.residency.MarkHostWrite(); err != nil {
		return fmt.Errorf("matrix: publish host values: %w", err)
	}
	return nil
}

func (m *Matrix) ensureDeviceBuffer(runtimeValue *device.Runtime) (
	buffer *device.Buffer,
	allocated bool,
	uploaded bool,
	err error,
) {
	if err = m.ensureResidency(runtimeValue); err != nil {
		return nil, false, false, err
	}
	if buffer, allocated, uploaded, err = m.residency.EnsureDevice(m.data); err != nil {
		return nil, false, false, fmt.Errorf("matrix: prepare device values: %w", err)
	}
	return buffer, allocated, uploaded, nil
}

func (m *Matrix) ensureExecutionDeviceBuffer(execution *device.Execution) (
	buffer *device.Buffer,
	allocated bool,
	uploaded bool,
	err error,
) {
	if execution == nil {
		err = errors.New("matrix: prepare execution device values: execution is nil")
		return nil, false, false, err
	}
	if m != nil && m.residency != nil {
		if buffer, uploaded = m.residency.PendingBuffer(execution); uploaded {
			return buffer, false, false, nil
		}
	}

	buffer, allocated, uploaded, err = m.ensureDeviceBuffer(execution.Runtime())
	return buffer, allocated, uploaded, err
}

func (m *Matrix) beginDeviceWrite(runtimeValue *device.Runtime) (
	buffer *device.Buffer,
	allocated bool,
	err error,
) {
	if err = m.ensureResidency(runtimeValue); err != nil {
		return nil, false, err
	}
	if buffer, allocated, err = m.residency.BeginDeviceWrite(); err != nil {
		return nil, false, fmt.Errorf("matrix: begin device write: %w", err)
	}
	return buffer, allocated, nil
}

func (m *Matrix) beginExecutionDeviceWrite(execution *device.Execution) (
	buffer *device.Buffer,
	allocated bool,
	err error,
) {
	var pending bool

	if execution == nil {
		err = errors.New("matrix: begin execution device write: execution is nil")
		return nil, false, err
	}
	if m != nil && m.residency != nil {
		_, pending = m.residency.PendingBuffer(execution)
		if pending {
			if err = execution.Barrier(device.BoundaryHostMutation); err != nil {
				return nil, false, fmt.Errorf("matrix: complete previous destination write: %w", err)
			}
		}
	}

	buffer, allocated, err = m.beginDeviceWrite(execution.Runtime())
	return buffer, allocated, err
}

func (m *Matrix) publishDeviceWrite(buffer *device.Buffer) (err error) {
	if m == nil || m.residency == nil {
		err = errors.New("matrix: publish device write: residency is nil")
		return err
	}
	if err = m.residency.PublishDeviceWrite(buffer); err != nil {
		return fmt.Errorf("matrix: publish device write: %w", err)
	}
	return nil
}

func (m *Matrix) failDeviceWrite(buffer *device.Buffer, cause error) (err error) {
	if m == nil || m.residency == nil || buffer == nil {
		return nil
	}

	if err = m.residency.FailDeviceWrite(buffer, cause); err != nil {
		return fmt.Errorf("matrix: discard failed device write: %w", err)
	}
	if err = m.residency.RestoreCommitted(); err != nil {
		return fmt.Errorf("matrix: restore committed device value: %w", err)
	}
	return nil
}

func (m *Matrix) ensureResidency(runtimeValue *device.Runtime) (err error) {
	if m == nil {
		err = errors.New("matrix: create residency: matrix is nil")
		return err
	}
	if runtimeValue == nil {
		err = errors.New("matrix: create residency: runtime is nil")
		return err
	}
	if m.residency != nil {
		if m.residency.Runtime() != runtimeValue {
			err = errors.New("matrix: residency belongs to another runtime")
			return err
		}
		return nil
	}

	if m.residency, err = device.NewResidency(runtimeValue, uint64(len(m.data))); err != nil {
		return fmt.Errorf("matrix: create residency: %w", err)
	}
	return nil
}

func (m *Matrix) detachDevice() (err error) {
	if m == nil || m.residency == nil {
		return nil
	}
	if err = m.ensureHostCurrent(); err != nil {
		return err
	}
	if err = m.markHostWrite(); err != nil {
		return err
	}
	if err = m.residency.Release(); err != nil {
		return fmt.Errorf("matrix: release device residency: %w", err)
	}
	m.residency = nil
	return nil
}

func copyMatrixHost(source, destination *Matrix) (err error) {
	if err = inheritExecution(destination, source); err != nil {
		return err
	}
	if err = source.ensureHostCurrent(); err != nil {
		return err
	}
	if err = destination.markHostWrite(); err != nil {
		return err
	}

	copy(destination.data, source.data)
	return nil
}
