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
	encodeAddRowVector(scope, values, rowVector any, rows, cols uint32) (err error)
	encodeAddScaled(scope, left, right, result any, scale float32, count uint32) (err error)
	encodeReLU(scope, input, result any, count uint32) (err error)
	encodeReLUBackward(scope, input, outputGradient, result any, count uint32) (err error)
	encodeSoftmaxRows(scope, input, result any, rows, cols uint32) (err error)
	encodeSoftmaxRowsBackward(scope, input, outputGradient, result any, rows, cols uint32) (err error)
	encodeColumnSums(scope, input, result any, rows, cols uint32, accumulate bool) (err error)
	encodeCategoricalCrossEntropy(
		scope,
		predictions,
		targets,
		result any,
		rows,
		cols uint32,
		epsilon float32,
	) (err error)
	encodeCategoricalCrossEntropyGradient(
		scope,
		predictions,
		targets,
		result any,
		rows,
		cols uint32,
		epsilon float32,
	) (err error)
	encodeMatMul(scope, left, right, result any, dimensions matMulDimensions, operation Operation) (err error)
	commit(scope any) (err error)
	completed(scope any) (complete bool, err error)
	wait(scope any) (err error)
	releaseScope(scope any) (err error)
	resourceSnapshot() (snapshot ResourceSnapshot)
	resetResourcePeaks() (err error)
}
