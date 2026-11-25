package gpu

import (
	"slices"
	"sync/atomic"

	"github.com/itsmontoya/neuralnetwork/internal/device"
)

type fakeBackend struct {
	bufferErr   error
	scopeErr    error
	liveBuffers atomic.Int32
	liveScopes  atomic.Int32
	submissions atomic.Int32
	waits       atomic.Int32
	uploads     atomic.Int32
	downloads   atomic.Int32
	matmuls     atomic.Int32
	elementwise atomic.Int32
}

func (b *fakeBackend) newBuffer(count uint64, dtype DType) (buffer allocation, err error) {
	if b.bufferErr != nil {
		return nil, b.bufferErr
	}

	var value fakeAllocation
	value.backend = b
	// Large logical allocations let tests validate shader limits without reserving memory.
	if count < 1024 {
		if dtype == Float32 {
			value.values = make([]float32, count)
		} else {
			value.bits16 = make([]uint16, count)
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

func (a *fakeAllocation) Release() {
	a.backend.liveBuffers.Add(-1)
	a.values = nil
	a.bits16 = nil
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
	var (
		input  = source.(*fakeAllocation)
		output = destination.(*fakeAllocation)
	)
	c.commands = append(c.commands, func() {
		copy(output.values, input.values)
		copy(output.bits16, input.bits16)
	})
	return c.encodeErr
}

func (c *fakeCommandBuffer) encodeGatherRows(source, destination allocation, columns uint64, rows []int64) (err error) {
	var (
		input   = source.(*fakeAllocation)
		output  = destination.(*fakeAllocation)
		width   = int(columns)
		indices = slices.Clone(rows)
	)
	c.commands = append(c.commands, func() {
		for i, row := range indices {
			var start = int(row) * width
			copy(output.values[i*width:(i+1)*width], input.values[start:start+width])
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

func (c *fakeCommandBuffer) encodeKVAppend(history, keys, values, target allocation, plane, prefix uint32) (err error) {
	var (
		output    = target.(*fakeAllocation)
		keyRows   = keys.(*fakeAllocation)
		valueRows = values.(*fakeAllocation)
	)
	c.commands = append(c.commands, func() {
		if history != nil {
			var active = history.(*fakeAllocation)
			copy(output.values[:prefix], active.values[:prefix])
			copy(output.values[plane:plane+prefix], active.values[plane:plane+prefix])
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
