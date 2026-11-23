//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"testing"
)

// Test_MetalGateUpExample shows how gate and up projections can feed activation
// and multiplication in a single command buffer with one terminal readback.
func Test_MetalGateUpExample(t *testing.T) {
	const rows, hidden, width = 2, 3, 5
	var (
		gpu         = requireMetalDevice(t)
		ctx         = context.Background()
		input       = requireTensor(t, gpu, rows, hidden)
		gateWeights = requireTensor(t, gpu, width, hidden)
		upWeights   = requireTensor(t, gpu, width, hidden)
		gate        = requireTensor(t, gpu, rows, width)
		up          = requireTensor(t, gpu, rows, width)
		activated   = requireTensor(t, gpu, rows, width)
		output      = requireTensor(t, gpu, rows, width)
		activations = []float32{1, -2, 0.5, 0, 3, -1}
		gateRows    = []float32{2, -1, 4, -2, 1, -4, 0, 3, -2, 1, 0, -2, 0, 0, 0}
		upRows      = []float32{1, 1, 0, -1, 1, 0, 0, 0, 1, -1, 0, 2, 3, 2, 1}
	)
	requireError(t, input.Upload(activations), nil)
	requireError(t, gateWeights.Upload(gateRows), nil)
	requireError(t, upWeights.Upload(upRows), nil)

	var (
		native = gpu.state.backend.(metalBackend).runtime
		before = native.ResourceSnapshot()
		scope  = requireScope(t, gpu)
	)
	requireError(t, scope.MatMulRightTranspose(input, gateWeights, gate), nil)
	requireError(t, scope.MatMulRightTranspose(input, upWeights, up), nil)
	requireError(t, scope.SiLU(gate, activated), nil)
	requireError(t, scope.Mul(activated, up, output), nil)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)

	var got, want [rows * width]float32
	requireError(t, output.DownloadInto(got[:]), nil)
	for row := range rows {
		for col := range width {
			var gateValue, upValue float32
			for k := range hidden {
				gateValue += activations[row*hidden+k] * gateRows[col*hidden+k]
				upValue += activations[row*hidden+k] * upRows[col*hidden+k]
			}
			want[row*width+col] = cpuSiLU(gateValue) * upValue
		}
	}
	checkElementwise(t, got[:], want[:], 2e-6)
	var after = native.ResourceSnapshot()
	if after.SubmittedCommands != before.SubmittedCommands+1 ||
		after.CompletedCommands != before.CompletedCommands+1 || after.LiveScopes != before.LiveScopes {
		t.Fatalf("gate/up example should complete one scope: before %+v, after %+v", before, after)
	}
}
