//go:build darwin && cgo && metal && !purego

package matrix

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
)

func copyMatrix(source, destination *Matrix) (err error) {
	var snapshot device.ResidencySnapshot

	if source.residency == nil {
		return copyMatrixHost(source, destination)
	}
	snapshot = source.residency.Snapshot()
	if !snapshot.HasBuffer ||
		snapshot.DeviceRevision != snapshot.LogicalRevision ||
		snapshot.HostRevision == snapshot.LogicalRevision {
		return copyMatrixHost(source, destination)
	}

	err = copyMatrixDevice(source, destination)
	if err != nil {
		metalRecordFailure(err)
	}
	return err
}

func copyMatrixDevice(source, destination *Matrix) (err error) {
	var (
		runtimeValue      *device.Runtime
		sourceBuffer      *device.Buffer
		destinationBuffer *device.Buffer
		scope             *device.Scope
		activity          metalBridgeActivity
		allocated         bool
		uploaded          bool
		published         bool
	)

	runtimeValue = source.residency.Runtime()
	if metaltest.Enabled() {
		defer func() {
			metaltest.RecordBridgeActivity(
				activity.bufferCreations,
				activity.inputUploads,
				activity.resultDownloads,
				activity.commandSubmissions,
				activity.waits,
			)
		}()
	}

	if sourceBuffer, allocated, uploaded, err = source.ensureDeviceBuffer(runtimeValue); err != nil {
		return fmt.Errorf("matrix: prepare Metal copy source: %w", err)
	}
	activity.recordDevicePreparation(allocated, uploaded)
	if destinationBuffer, allocated, err = destination.beginDeviceWrite(runtimeValue); err != nil {
		return fmt.Errorf("matrix: prepare Metal copy destination: %w", err)
	}
	if allocated {
		activity.bufferCreations++
	}
	defer func() {
		var cleanupErr error
		if !published {
			cleanupErr = destination.failDeviceWrite(destinationBuffer, err)
			if cleanupErr != nil {
				err = errors.Join(err, cleanupErr)
			}
		}
	}()

	if scope, err = runtimeValue.NewScope(); err != nil {
		return fmt.Errorf("matrix: create Metal copy scope: %w", err)
	}
	defer func() {
		var releaseErr error
		if releaseErr = scope.Release(); err == nil && releaseErr != nil {
			err = fmt.Errorf("matrix: release Metal copy scope: %w", releaseErr)
		}
	}()
	if err = scope.EncodeCopy(sourceBuffer, destinationBuffer); err != nil {
		return fmt.Errorf("matrix: encode Metal copy: %w", err)
	}
	if err = scope.Commit(); err != nil {
		return fmt.Errorf("matrix: commit Metal copy: %w", err)
	}
	activity.commandSubmissions++
	if err = scope.Wait(); err != nil {
		activity.waits++
		return fmt.Errorf("matrix: wait for Metal copy: %w", err)
	}
	activity.waits++
	if err = destination.publishDeviceWrite(destinationBuffer); err != nil {
		return err
	}
	destination.residency.RecordDeviceCopy()
	published = true
	return nil
}
