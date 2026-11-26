package gpu

import (
	"slices"
	"sync/atomic"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

type fakeBackend struct {
	bufferErr           error
	scopeErr            error
	liveBuffers         atomic.Int32
	liveScopes          atomic.Int32
	submissions         atomic.Int32
	waits               atomic.Int32
	uploads             atomic.Int32
	downloads           atomic.Int32
	matmuls             atomic.Int32
	elementwise         atomic.Int32
	scales              atomic.Int32
	broadcasts          atomic.Int32
	softmaxes           atomic.Int32
	layerNorms          atomic.Int32
	batchNorms          atomic.Int32
	pool2Ds             atomic.Int32
	transposes          atomic.Int32
	copies              atomic.Int32
	matMulLeftTranspose atomic.Int32
	batchedMatMuls      atomic.Int32
}

func (b *fakeBackend) newBuffer(count, bytes uint64, dtype DType) (buffer allocation, err error) {
	if b.bufferErr != nil {
		return nil, b.bufferErr
	}

	var value fakeAllocation
	value.backend = b
	// Large logical allocations let tests validate shader limits without reserving memory.
	if count < 1024 {
		switch {
		case dtype == Float32:
			value.values = make([]float32, count)
		case dtype.is16Bit():
			value.bits16 = make([]uint16, count)
		default:
			value.raw = make([]byte, bytes)
		}
	}
	b.liveBuffers.Add(1)
	return &value, nil
}

func (b *fakeBackend) newScope() (scope commandBuffer, err error) {
	if b.scopeErr != nil {
		return nil, b.scopeErr
	}

	var value fakeCommandBuffer
	value.backend = b
	b.liveScopes.Add(1)
	return &value, nil
}

type fakeAllocation struct {
	backend       *fakeBackend
	values        []float32
	bits16        []uint16
	raw           []byte
	uploadStarted chan struct{}
	uploadGate    <-chan struct{}
}

func (a *fakeAllocation) Upload(values []float32) (err error) {
	a.backend.uploads.Add(1)
	if a.uploadStarted != nil {
		close(a.uploadStarted)
		<-a.uploadGate
	}
	copy(a.values, values)
	return nil
}

func (a *fakeAllocation) Download(destination []float32) (err error) {
	a.backend.downloads.Add(1)
	copy(destination, a.values)
	return nil
}

func (a *fakeAllocation) UploadUint16(values []uint16) (err error) {
	a.backend.uploads.Add(1)
	if a.uploadStarted != nil {
		close(a.uploadStarted)
		<-a.uploadGate
	}
	copy(a.bits16, values)
	return nil
}

func (a *fakeAllocation) DownloadUint16(destination []uint16) (err error) {
	a.backend.downloads.Add(1)
	copy(destination, a.bits16)
	return nil
}

func (a *fakeAllocation) UploadBytes(values []byte) (err error) {
	a.backend.uploads.Add(1)
	if a.uploadStarted != nil {
		close(a.uploadStarted)
		<-a.uploadGate
	}
	if a.raw == nil {
		a.raw = make([]byte, len(values))
	}
	copy(a.raw, values)
	return nil
}

func (a *fakeAllocation) DownloadBytes(destination []byte) (err error) {
	a.backend.downloads.Add(1)
	copy(destination, a.raw)
	return nil
}

func (a *fakeAllocation) Release() {
	a.backend.liveBuffers.Add(-1)
	a.values = nil
	a.bits16 = nil
	a.raw = nil
}

type fakeCommandBuffer struct {
	backend     *fakeBackend
	commands    []func()
	encodeErr   error
	commitErr   error
	waitErr     error
	waitStarted chan struct{}
	waitGate    <-chan struct{}
}

func (c *fakeCommandBuffer) encodeFill(destination allocation, value float32) (err error) {
	var target = destination.(*fakeAllocation)
	c.commands = append(c.commands, func() {
		for i := range target.values {
			target.values[i] = value
		}
	})
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeCopy(source, destination allocation) (err error) {
	c.backend.copies.Add(1)
	var (
		input  = source.(*fakeAllocation)
		output = destination.(*fakeAllocation)
	)
	c.commands = append(c.commands, func() {
		copy(output.values, input.values)
		copy(output.bits16, input.bits16)
		copy(output.raw, input.raw)
	})
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeGatherRows(source, destination allocation, columns uint64, rows []int64, sourceDType, destinationDType DType) (err error) {
	var (
		input   = source.(*fakeAllocation)
		output  = destination.(*fakeAllocation)
		width   = int(columns)
		indices = slices.Clone(rows)
	)
	c.commands = append(c.commands, func() {
		for i, row := range indices {
			var start = int(row) * width
			switch {
			case sourceDType == Float32 && destinationDType == Float32:
				copy(output.values[i*width:(i+1)*width], input.values[start:start+width])
			case sourceDType == destinationDType && sourceDType.IsQuantized():
				var rowBytes = int((columns / QuantBlockSize) * sourceDType.BlockBytes())
				copy(output.raw[i*rowBytes:(i+1)*rowBytes], input.raw[int(row)*rowBytes:(int(row)+1)*rowBytes])
			case sourceDType == destinationDType:
				copy(output.bits16[i*width:(i+1)*width], input.bits16[start:start+width])
			case sourceDType == Float16 && destinationDType == Float32:
				for col := range width {
					output.values[i*width+col] = Float16ToFloat32(input.bits16[start+col])
				}
			case sourceDType == BFloat16 && destinationDType == Float32:
				for col := range width {
					output.values[i*width+col] = BFloat16ToFloat32(input.bits16[start+col])
				}
			case sourceDType == Q8_0 && destinationDType == Float32:
				var rowBytes = int((columns / QuantBlockSize) * QuantBlockBytesQ8_0)
				dequantizeQ8_0Into(input.raw[int(row)*rowBytes:(int(row)+1)*rowBytes], output.values[i*width:(i+1)*width])
			case sourceDType == Q4_0 && destinationDType == Float32:
				var rowBytes = int((columns / QuantBlockSize) * QuantBlockBytesQ4_0)
				dequantizeQ4_0Into(input.raw[int(row)*rowBytes:(int(row)+1)*rowBytes], output.values[i*width:(i+1)*width])
			case sourceDType == Q4_1 && destinationDType == Float32:
				var rowBytes = int((columns / QuantBlockSize) * QuantBlockBytesQ4_1)
				dequantizeQ4_1Into(input.raw[int(row)*rowBytes:(int(row)+1)*rowBytes], output.values[i*width:(i+1)*width])
			}
		}
	})
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeMatMul(_, _, _ allocation, _, _, _ uint32, _ bool, _ DType) (err error) {
	c.backend.matmuls.Add(1)
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeElementwise(_, _, _ allocation, _ device.ElementwiseOperation) (err error) {
	c.backend.elementwise.Add(1)
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeScale(_, _ allocation, _ float32) (err error) {
	c.backend.scales.Add(1)
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeBroadcast(_, _, _ allocation, _ device.BroadcastOperation, _ device.BroadcastSpec) (err error) {
	c.backend.broadcasts.Add(1)
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeSoftmax(_, _ allocation, _, _, _ uint32) (err error) {
	c.backend.softmaxes.Add(1)
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeLayerNorm(_, _, _, _ allocation, _, _ uint32, _ float32, _, _ DType) (err error) {
	c.backend.layerNorms.Add(1)
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeBatchNorm(_, _, _, _, _, _ allocation, _, _, _ uint32, _ float32) (err error) {
	c.backend.batchNorms.Add(1)
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodePool2D(_, _ allocation, _ device.Pool2DSpec) (err error) {
	c.backend.pool2Ds.Add(1)
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeTranspose(_, _ allocation, _ device.TransposeSpec) (err error) {
	c.backend.transposes.Add(1)
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeMatMulLeftTranspose(_, _, _ allocation, _, _, _ uint32, _ DType) (err error) {
	c.backend.matMulLeftTranspose.Add(1)
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeBatchedMatMul(_, _, _ allocation, _ device.BatchedMatMulSpec) (err error) {
	c.backend.batchedMatMuls.Add(1)
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeKVAppend(keys, values, target allocation, plane, prefix uint32, cacheDType DType) (err error) {
	var (
		output    = target.(*fakeAllocation)
		keyRows   = keys.(*fakeAllocation)
		valueRows = values.(*fakeAllocation)
	)
	c.commands = append(c.commands, func() {
		if cacheDType == Float16 {
			for i, v := range keyRows.values {
				output.bits16[int(prefix)+i] = Float32ToFloat16(v)
			}
			for i, v := range valueRows.values {
				output.bits16[int(plane+prefix)+i] = Float32ToFloat16(v)
			}
			return
		}
		copy(output.values[prefix:plane], keyRows.values)
		copy(output.values[plane+prefix:], valueRows.values)
	})
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeAttentionMath(_ device.AttentionMathOperation, _, _, _, _ allocation, _ [6]uint32, _ uint32) (err error) {
	return c.encodeErr
}

func (c *fakeCommandBuffer) Commit() (err error) {
	if c.commitErr != nil {
		return c.commitErr
	}
	c.backend.submissions.Add(1)
	return nil
}

func (c *fakeCommandBuffer) Wait() (err error) {
	c.backend.waits.Add(1)
	if c.waitStarted != nil {
		close(c.waitStarted)
		<-c.waitGate
	}
	for _, command := range c.commands {
		command()
	}
	return c.waitErr
}

func (c *fakeCommandBuffer) Release() (err error) {
	c.commands = nil
	c.backend.liveScopes.Add(-1)
	return nil
}
