package matrix

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

func (m *Matrix) ensureHostCurrent() (err error) {
	var downloaded bool

	if m == nil || m.residency == nil {
		return nil
	}

	if downloaded, err = m.residency.EnsureHost(m.data); err != nil {
		return fmt.Errorf("matrix: synchronize host values: %w", err)
	}
	if downloaded {
		recordResidencyDownload()
	}
	return nil
}

func (m *Matrix) markHostWrite() (err error) {
	if m == nil || m.residency == nil {
		return nil
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
	if err = source.ensureHostCurrent(); err != nil {
		return err
	}
	if err = destination.markHostWrite(); err != nil {
		return err
	}

	copy(destination.data, source.data)
	return nil
}
