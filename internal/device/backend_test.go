package device

import (
	"errors"
	"sync"
)

type testBuffer struct {
	values   []float32
	released bool
}

type testCommand func()

type testScope struct {
	commands  []testCommand
	committed bool
	completed bool
	released  bool
}

type testBackend struct {
	mutex         sync.Mutex
	availableFlag bool
	availableErr  error
	newBufferErr  error
	newScopeErr   error
	encodeErr     error
	commitErr     error
	waitErr       error
	resources     ResourceSnapshot
}

func newTestBackend() (out *testBackend) {
	var value testBackend
	value.availableFlag = true
	return &value
}

func (b *testBackend) available() (available bool, err error) {
	return b.availableFlag, b.availableErr
}

func (b *testBackend) newBuffer(bytes uint64) (handle any, err error) {
	var value testBuffer

	if b.newBufferErr != nil {
		return nil, b.newBufferErr
	}

	value.values = make([]float32, bytes/4)
	b.mutex.Lock()
	b.resources.LiveBuffers++
	b.resources.LiveBufferBytes += bytes
	b.resources.CreatedBuffers++
	if b.resources.LiveBuffers > b.resources.PeakBuffers {
		b.resources.PeakBuffers = b.resources.LiveBuffers
	}
	if b.resources.LiveBufferBytes > b.resources.PeakBufferBytes {
		b.resources.PeakBufferBytes = b.resources.LiveBufferBytes
	}
	b.mutex.Unlock()
	return &value, nil
}

func (b *testBackend) upload(handle any, values []float32) (err error) {
	var buffer *testBuffer

	if buffer = testBufferHandle(handle); buffer == nil || buffer.released {
		return ErrReleased
	}
	copy(buffer.values, values)
	return nil
}

func (b *testBackend) download(handle any, values []float32) (err error) {
	var buffer *testBuffer

	if buffer = testBufferHandle(handle); buffer == nil || buffer.released {
		return ErrReleased
	}
	copy(values, buffer.values)
	return nil
}

func (b *testBackend) releaseBuffer(handle any) {
	var buffer *testBuffer

	if buffer = testBufferHandle(handle); buffer == nil || buffer.released {
		return
	}
	buffer.released = true
	b.mutex.Lock()
	b.resources.LiveBuffers--
	b.resources.LiveBufferBytes -= uint64(len(buffer.values) * 4)
	b.resources.ReleasedBuffers++
	b.mutex.Unlock()
}

func (b *testBackend) newScope() (handle any, err error) {
	var value testScope

	if b.newScopeErr != nil {
		return nil, b.newScopeErr
	}

	b.mutex.Lock()
	b.resources.LiveScopes++
	b.resources.CreatedScopes++
	if b.resources.LiveScopes > b.resources.PeakScopes {
		b.resources.PeakScopes = b.resources.LiveScopes
	}
	b.mutex.Unlock()
	return &value, nil
}

func (b *testBackend) encodeCopy(scope, source, destination any, _ uint64) (err error) {
	var (
		commandScope      *testScope
		sourceBuffer      *testBuffer
		destinationBuffer *testBuffer
	)

	if b.encodeErr != nil {
		return b.encodeErr
	}
	commandScope = testScopeHandle(scope)
	sourceBuffer = testBufferHandle(source)
	destinationBuffer = testBufferHandle(destination)
	if commandScope == nil || sourceBuffer == nil || destinationBuffer == nil {
		return errors.New("test backend: nil copy handle")
	}
	commandScope.commands = append(commandScope.commands, func() {
		copy(destinationBuffer.values, sourceBuffer.values)
	})
	return nil
}

func (b *testBackend) encodeFill(scope, handle any, value float32, _ uint64) (err error) {
	var (
		commandScope *testScope
		buffer       *testBuffer
	)

	if b.encodeErr != nil {
		return b.encodeErr
	}
	commandScope = testScopeHandle(scope)
	buffer = testBufferHandle(handle)
	if commandScope == nil || buffer == nil {
		return errors.New("test backend: nil fill handle")
	}
	commandScope.commands = append(commandScope.commands, func() {
		var index int
		for index = range buffer.values {
			buffer.values[index] = value
		}
	})
	return nil
}

func (b *testBackend) encodeMatMul(
	any,
	any,
	any,
	any,
	matMulDimensions,
	Operation,
) (err error) {
	return b.encodeErr
}

func (b *testBackend) commit(handle any) (err error) {
	var scope *testScope

	if b.commitErr != nil {
		return b.commitErr
	}
	if scope = testScopeHandle(handle); scope == nil {
		return errors.New("test backend: nil commit handle")
	}
	scope.committed = true
	b.mutex.Lock()
	b.resources.SubmittedCommands++
	b.mutex.Unlock()
	return nil
}

func (b *testBackend) completed(handle any) (complete bool, err error) {
	var scope *testScope

	if scope = testScopeHandle(handle); scope == nil {
		return false, errors.New("test backend: nil completion handle")
	}
	return scope.completed, nil
}

func (b *testBackend) wait(handle any) (err error) {
	var (
		scope   *testScope
		command testCommand
	)

	if b.waitErr != nil {
		return b.waitErr
	}
	if scope = testScopeHandle(handle); scope == nil {
		return errors.New("test backend: nil wait handle")
	}
	if !scope.completed {
		for _, command = range scope.commands {
			command()
		}
		scope.completed = true
		b.mutex.Lock()
		b.resources.CompletedCommands++
		b.mutex.Unlock()
	}
	return nil
}

func (b *testBackend) releaseScope(handle any) {
	var scope *testScope

	if scope = testScopeHandle(handle); scope == nil || scope.released {
		return
	}
	scope.released = true
	b.mutex.Lock()
	b.resources.LiveScopes--
	b.resources.ReleasedScopes++
	b.mutex.Unlock()
}

func (b *testBackend) resourceSnapshot() (snapshot ResourceSnapshot) {
	b.mutex.Lock()
	snapshot = b.resources
	b.mutex.Unlock()
	return snapshot
}

func (b *testBackend) resetResourcePeaks() (err error) {
	b.mutex.Lock()
	defer b.mutex.Unlock()
	if b.resources.LiveBuffers != 0 || b.resources.LiveScopes != 0 {
		return errors.New("test backend: live resources")
	}
	b.resources = ResourceSnapshot{}
	return nil
}

func testBufferHandle(handle any) (buffer *testBuffer) {
	buffer, _ = handle.(*testBuffer)
	return buffer
}

func testScopeHandle(handle any) (scope *testScope) {
	scope, _ = handle.(*testScope)
	return scope
}
