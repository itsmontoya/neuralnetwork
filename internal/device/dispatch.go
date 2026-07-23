package device

const (
	metalMatMulColdMinOperations uint64 = 1 << 26
	metalMatMulWarmMinOperations uint64 = 1 << 22
)

// MatMulEligible reports whether total multiplication work amortizes the
// current process-level Metal readiness cost.
func MatMulEligible(operations uint64) (eligible bool) {
	eligible = matMulEligibleForRuntime(operations, sharedRuntimeReady.Load())
	return eligible
}

func matMulEligibleForRuntime(operations uint64, runtimeReady bool) (eligible bool) {
	var minimum uint64

	minimum = metalMatMulColdMinOperations
	if runtimeReady {
		minimum = metalMatMulWarmMinOperations
	}

	eligible = operations >= minimum
	return eligible
}
