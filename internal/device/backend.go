package device

type backend interface {
	available() (available bool, err error)
	newBuffer(bytes uint64) (handle any, err error)
	upload(handle any, values []float32) (err error)
	download(handle any, values []float32) (err error)
	releaseBuffer(handle any)
	newScope() (handle any, err error)
	encodeCopy(scope, source, destination any, bytes uint64) (err error)
	encodeFill(scope, buffer any, value float32, count uint64) (err error)
	encodeMatMul(scope, left, right, result any, dimensions matMulDimensions, operation Operation) (err error)
	commit(scope any) (err error)
	completed(scope any) (complete bool, err error)
	wait(scope any) (err error)
	releaseScope(scope any)
	resourceSnapshot() (snapshot ResourceSnapshot)
	resetResourcePeaks() (err error)
}
