//go:build darwin && cgo && metal && !purego

package matrix

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

func copyMatrix(source, destination *Matrix) (err error) {
	var (
		execution *device.Execution
		snapshot  device.ResidencySnapshot
	)

	if execution, err = compatibleExecution(source, destination); err != nil {
		return err
	}
	if execution != nil && execution.Activated() {
		err = copyMatrixDevice(source, destination, execution, false)
		if err != nil {
			metalRecordFailure(err)
		}
		return err
	}

	if source.residency == nil {
		return copyMatrixHost(source, destination)
	}
	snapshot = source.residency.Snapshot()
	if !snapshot.HasBuffer ||
		snapshot.DeviceRevision != snapshot.LogicalRevision ||
		snapshot.HostRevision == snapshot.LogicalRevision {
		return copyMatrixHost(source, destination)
	}

	err = copyMatrixDevice(source, destination, nil, true)
	if err != nil {
		metalRecordFailure(err)
	}
	return err
}

func copyMatrixDevice(
	source,
	destination *Matrix,
	execution *device.Execution,
	owned bool,
) (err error) {
	var (
		sourceBuffer      *device.Buffer
		destinationBuffer *device.Buffer
		publication       device.Publication
		allocated         bool
		uploaded          bool
	)

	if execution == nil {
		execution = device.NewExecution(source.residency.Runtime())
	}
	if execution == nil {
		err = errors.New("matrix: create Metal copy execution: runtime is nil")
		return err
	}
	if owned {
		defer func() {
			if err != nil && execution.Active() {
				err = errors.Join(err, execution.Abort(err))
			}
		}()
	}
	if err = execution.Bind(source); err != nil {
		return fmt.Errorf("matrix: bind Metal copy source: %w", err)
	}
	if err = execution.Bind(destination); err != nil {
		return fmt.Errorf("matrix: bind Metal copy destination: %w", err)
	}

	if sourceBuffer, allocated, uploaded, err = source.ensureExecutionDeviceBuffer(execution); err != nil {
		return fmt.Errorf("matrix: prepare Metal copy source: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(source.data))*4)
	if destinationBuffer, allocated, err = destination.beginExecutionDeviceWrite(execution); err != nil {
		return fmt.Errorf("matrix: prepare Metal copy destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, false, 0)
	publication.Publish = func() (publishErr error) {
		if publishErr = destination.publishDeviceWrite(destinationBuffer); publishErr != nil {
			return publishErr
		}
		destination.residency.RecordDeviceCopy()
		return nil
	}
	publication.Discard = func(cause error) (discardErr error) {
		discardErr = destination.failDeviceWrite(destinationBuffer, cause)
		return discardErr
	}
	if err = execution.EncodeCopy(
		sourceBuffer,
		destinationBuffer,
		uint64(len(destination.data))*4,
		publication,
	); err != nil {
		return fmt.Errorf("matrix: encode Metal copy: %w", err)
	}
	if err = execution.MarkRead(source); err != nil {
		return fmt.Errorf("matrix: record Metal copy source use: %w", err)
	}
	if owned {
		if err = execution.Finish(); err != nil {
			return fmt.Errorf("matrix: finish Metal copy execution: %w", err)
		}
	}
	return nil
}
