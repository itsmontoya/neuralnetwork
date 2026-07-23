//go:build darwin && cgo && metal && !purego

package matrix

import (
	"errors"
	"fmt"
	"math"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

func categoricalCrossEntropyValueDevice(
	predictions,
	targets *Matrix,
	epsilon float32,
) (value float32, handled bool, err error) {
	var execution *device.Execution

	if execution, err = compatibleExecution(predictions, targets); err != nil {
		return 0, false, err
	}
	if execution == nil || !execution.Activated() ||
		!metalElementwiseSupported(predictions) ||
		!metalElementwiseSupported(targets) {
		return 0, false, nil
	}

	value, err = categoricalCrossEntropyValueMetal(
		execution,
		predictions,
		targets,
		epsilon,
	)
	return value, true, err
}

func categoricalCrossEntropyValueMetal(
	execution *device.Execution,
	predictions,
	targets *Matrix,
	epsilon float32,
) (value float32, err error) {
	var (
		predictionBuffer *device.Buffer
		targetBuffer     *device.Buffer
		resultBuffer     *device.Buffer
		resultValues     []float32
		resultStorage    [device.CategoricalCrossEntropyResultCount]float32
		snapshot         device.ResidencySnapshot
		publication      device.Publication
		status           uint32
		allocated        bool
		uploaded         bool
	)

	if err = execution.Bind(predictions); err != nil {
		return 0, fmt.Errorf("matrix: bind Metal categorical predictions: %w", err)
	}
	if err = execution.Bind(targets); err != nil {
		return 0, fmt.Errorf("matrix: bind Metal categorical targets: %w", err)
	}
	if predictionBuffer, allocated, uploaded, err =
		predictions.ensureExecutionDeviceBuffer(execution); err != nil {
		return 0, fmt.Errorf("matrix: prepare Metal categorical predictions: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(predictions.data))*4)
	if targetBuffer, allocated, uploaded, err =
		targets.ensureExecutionDeviceBuffer(execution); err != nil {
		return 0, fmt.Errorf("matrix: prepare Metal categorical targets: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(targets.data))*4)
	if resultBuffer, err = execution.Runtime().NewBuffer(
		device.CategoricalCrossEntropyResultCount,
	); err != nil {
		return 0, fmt.Errorf("matrix: allocate Metal categorical result: %w", err)
	}
	execution.RecordDevicePreparation(true, false, 0)
	defer resultBuffer.Release()

	publication.Publish = func() (publishErr error) {
		return nil
	}
	publication.Discard = func(error) (discardErr error) {
		return nil
	}
	if err = execution.EncodeCategoricalCrossEntropy(
		predictionBuffer,
		targetBuffer,
		resultBuffer,
		uint32(predictions.rows),
		uint32(predictions.cols),
		epsilon,
		device.CategoricalCrossEntropyResultCount*4,
		publication,
	); err != nil {
		return 0, fmt.Errorf("matrix: encode Metal categorical cross entropy: %w", err)
	}
	if err = execution.MarkRead(predictions); err != nil {
		return 0, fmt.Errorf("matrix: record Metal categorical prediction use: %w", err)
	}
	if err = execution.MarkRead(targets); err != nil {
		return 0, fmt.Errorf("matrix: record Metal categorical target use: %w", err)
	}
	if err = execution.Barrier(device.BoundaryHostObservation); err != nil {
		return 0, fmt.Errorf("matrix: complete Metal categorical cross entropy: %w", err)
	}

	resultValues = resultStorage[:]
	if err = resultBuffer.Download(resultValues); err != nil {
		return 0, fmt.Errorf("matrix: download Metal categorical result: %w", err)
	}
	execution.RecordDownload(device.CategoricalCrossEntropyResultCount * 4)
	status = math.Float32bits(resultValues[1])
	switch status {
	case 0:
	case 1:
		err = device.CategoricalTargetError{
			Row:       math.Float32bits(resultValues[2]),
			Column:    math.Float32bits(resultValues[3]),
			Value:     resultValues[4],
			NonBinary: true,
		}
		return 0, err
	case 2:
		err = device.CategoricalTargetError{
			Row:  math.Float32bits(resultValues[2]),
			Ones: math.Float32bits(resultValues[3]),
		}
		return 0, err
	default:
		err = fmt.Errorf("matrix: Metal categorical diagnostic status is invalid: %d", status)
		return 0, err
	}

	snapshot = targets.residency.Snapshot()
	if err = execution.RecordValidation(targets, snapshot.LogicalRevision); err != nil {
		return 0, fmt.Errorf("matrix: record Metal categorical target validation: %w", err)
	}
	value = resultValues[0]
	return value, nil
}

func categoricalCrossEntropyGradientDevice(
	predictions,
	targets,
	gradient *Matrix,
	epsilon float32,
) (handled bool, err error) {
	var (
		execution        *device.Execution
		predictionBuffer *device.Buffer
		targetBuffer     *device.Buffer
		gradientBuffer   *device.Buffer
		snapshot         device.ResidencySnapshot
		validated        bool
		allocated        bool
		uploaded         bool
	)

	if execution, err = compatibleExecution(predictions, targets, gradient); err != nil {
		return false, err
	}
	if execution == nil || !execution.Activated() ||
		!metalElementwiseSupported(predictions) ||
		!metalElementwiseSupported(targets) {
		return false, nil
	}
	if targets.residency == nil {
		return false, nil
	}
	snapshot = targets.residency.Snapshot()
	if validated, err = execution.Validated(targets, snapshot.LogicalRevision); err != nil {
		return false, fmt.Errorf("matrix: inspect Metal categorical target validation: %w", err)
	}
	if !validated {
		return false, nil
	}
	if err = gradient.requireShape(
		"categorical gradient destination",
		predictions.rows,
		predictions.cols,
	); err != nil {
		return false, err
	}
	if err = inheritExecution(gradient, predictions, targets); err != nil {
		return false, err
	}
	if err = execution.Bind(gradient); err != nil {
		return false, fmt.Errorf("matrix: bind Metal categorical gradient: %w", err)
	}
	if predictionBuffer, allocated, uploaded, err =
		predictions.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal categorical gradient predictions: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(predictions.data))*4)
	if targetBuffer, allocated, uploaded, err =
		targets.ensureExecutionDeviceBuffer(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal categorical gradient targets: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(targets.data))*4)
	if gradientBuffer, allocated, err = gradient.beginExecutionDeviceWrite(execution); err != nil {
		return false, fmt.Errorf("matrix: prepare Metal categorical gradient destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, false, 0)
	if err = execution.EncodeCategoricalCrossEntropyGradient(
		predictionBuffer,
		targetBuffer,
		gradientBuffer,
		uint32(predictions.rows),
		uint32(predictions.cols),
		epsilon,
		uint64(len(gradient.data))*4,
		deviceWritePublication(gradient, gradientBuffer),
	); err != nil {
		return false, fmt.Errorf("matrix: encode Metal categorical gradient: %w", err)
	}
	if err = execution.MarkRead(predictions); err != nil {
		return false, fmt.Errorf("matrix: record Metal categorical gradient prediction use: %w", err)
	}
	if err = execution.MarkRead(targets); err != nil {
		return false, fmt.Errorf("matrix: record Metal categorical gradient target use: %w", err)
	}

	return true, nil
}

func sgdDevice(updates []device.ParameterUpdate, learningRate float32) (handled bool, err error) {
	var (
		execution     *device.Execution
		values        *Matrix
		gradient      *Matrix
		current       *device.Execution
		update        device.ParameterUpdate
		transient     uint64
		matrixBytes   uint64
		index         int
		previousIndex int
		ok            bool
		pending       bool
	)

	if len(updates) == 0 {
		return false, nil
	}
	for index, update = range updates {
		if values, gradient, err = parameterUpdateMatrices(update, index); err != nil {
			return false, err
		}
		if !metalElementwiseSupported(values) || !metalElementwiseSupported(gradient) {
			return false, nil
		}
		if current, err = compatibleExecution(values, gradient); err != nil {
			return false, err
		}
		if current == nil {
			return false, nil
		}
		if execution != nil && execution != current {
			err = errors.New("matrix: SGD parameters belong to different executions")
			return false, err
		}
		execution = current
		for previousIndex = 0; previousIndex < index; previousIndex++ {
			if updates[previousIndex].Values == update.Values ||
				updates[previousIndex].Gradient == update.Gradient {
				return false, nil
			}
		}
		matrixBytes = uint64(len(values.data)) * 4
		if transient > ^uint64(0)-matrixBytes {
			return false, nil
		}
		transient += matrixBytes
		if gradient.residency == nil {
			return false, nil
		}
		_, pending = gradient.residency.PendingBuffer(execution)
		if !pending {
			return false, nil
		}
	}
	if execution == nil || !execution.Activated() ||
		!execution.CanEncodeAtomic(uint64(len(updates))*2, transient) {
		return false, nil
	}

	for index, update = range updates {
		if values, gradient, err = parameterUpdateMatrices(update, index); err != nil {
			return false, err
		}
		if err = encodeSGDValue(execution, values, gradient, learningRate); err != nil {
			return false, err
		}
	}
	for index, update = range updates {
		if _, gradient, err = parameterUpdateMatrices(update, index); err != nil {
			return false, err
		}
		if err = encodeSGDReset(execution, gradient); err != nil {
			return false, err
		}
	}

	ok = true
	return ok, nil
}

func parameterUpdateMatrices(
	update device.ParameterUpdate,
	index int,
) (values, gradient *Matrix, err error) {
	var ok bool

	if values, ok = update.Values.(*Matrix); !ok {
		err = fmt.Errorf(
			"matrix: SGD parameter %d values have type %T, want *matrix.Matrix",
			index,
			update.Values,
		)
		return nil, nil, err
	}
	if gradient, ok = update.Gradient.(*Matrix); !ok {
		err = fmt.Errorf(
			"matrix: SGD parameter %d gradient has type %T, want *matrix.Matrix",
			index,
			update.Gradient,
		)
		return nil, nil, err
	}
	if err = values.sameShape(gradient); err != nil {
		return nil, nil, fmt.Errorf("matrix: SGD parameter %d shape: %w", index, err)
	}
	return values, gradient, nil
}

func encodeSGDValue(
	execution *device.Execution,
	values,
	gradient *Matrix,
	learningRate float32,
) (err error) {
	var (
		valuesBuffer        *device.Buffer
		gradientBuffer      *device.Buffer
		updatedValuesBuffer *device.Buffer
		allocated           bool
		uploaded            bool
	)

	if valuesBuffer, allocated, uploaded, err = values.ensureExecutionDeviceBuffer(execution); err != nil {
		return fmt.Errorf("matrix: prepare Metal SGD values: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(values.data))*4)
	if gradientBuffer, allocated, uploaded, err =
		gradient.ensureExecutionDeviceBuffer(execution); err != nil {
		return fmt.Errorf("matrix: prepare Metal SGD gradient: %w", err)
	}
	execution.RecordDevicePreparation(allocated, uploaded, uint64(len(gradient.data))*4)
	if updatedValuesBuffer, allocated, err = values.beginExecutionDeviceWrite(execution); err != nil {
		return fmt.Errorf("matrix: prepare Metal SGD destination: %w", err)
	}
	execution.RecordDevicePreparation(allocated, false, 0)
	if err = execution.EncodeAddScaled(
		valuesBuffer,
		gradientBuffer,
		updatedValuesBuffer,
		-learningRate,
		uint64(len(values.data))*4,
		deviceWritePublication(values, updatedValuesBuffer),
	); err != nil {
		return fmt.Errorf("matrix: encode Metal SGD update: %w", err)
	}
	if err = execution.MarkRead(values); err != nil {
		return fmt.Errorf("matrix: record Metal SGD value use: %w", err)
	}
	if err = execution.MarkRead(gradient); err != nil {
		return fmt.Errorf("matrix: record Metal SGD gradient use: %w", err)
	}
	return nil
}

func encodeSGDReset(execution *device.Execution, gradient *Matrix) (err error) {
	var (
		resetBuffer *device.Buffer
		pending     bool
	)

	if gradient == nil || gradient.residency == nil {
		err = errors.New("matrix: prepare Metal SGD gradient reset: residency is nil")
		return err
	}
	if resetBuffer, pending = gradient.residency.PendingBuffer(execution); !pending {
		err = errors.New("matrix: prepare Metal SGD gradient reset: gradient is not pending")
		return err
	}
	if err = execution.EncodeDependentFill(resetBuffer, 0); err != nil {
		return fmt.Errorf("matrix: encode Metal SGD gradient reset: %w", err)
	}
	return nil
}
