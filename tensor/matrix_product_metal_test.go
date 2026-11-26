//go:build darwin && arm64 && cgo && metal && !purego

package tensor

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/float32product"
)

func Test_MatrixProduct_MetalUsesHostPath(t *testing.T) {
	var (
		runtimeValue *device.Runtime
		available    bool
		before       device.ResourceSnapshot
		after        device.ResourceSnapshot
		err          error
	)

	if runtimeValue, available, err = device.ExperimentalRuntime(); err != nil {
		t.Fatal(err)
	}
	if !available {
		t.Skip("Metal runtime is not available on this machine")
	}

	const (
		rows  = 16
		inner = 128
		cols  = 256
	)

	var (
		leftValues     = make([]float32, rows*inner)
		leftTValues    = make([]float32, inner*rows)
		rightValues    = make([]float32, inner*cols)
		rightTValues   = make([]float32, cols*inner)
		biasValues     = make([]float32, cols)
		left, leftT    *Tensor
		right, rightT  *Tensor
		bias           *Tensor
		matMulOut      *Tensor
		gemmOut        *Tensor
		preparedOut    *Tensor
		preparedWeight *PreparedRightTranspose
		gemmConfigs    []GemmConfig
		config         GemmConfig
		index          int
	)

	for index = range leftValues {
		leftValues[index] = float32(index%29-14) / 13
	}
	for index = range leftTValues {
		leftTValues[index] = float32(index%31-15) / 17
	}
	for index = range rightValues {
		rightValues[index] = float32(index%37-18) / 19
	}
	for index = range rightTValues {
		rightTValues[index] = float32(index%41-20) / 23
	}
	for index = range biasValues {
		biasValues[index] = float32(index%11-5) / 7
	}

	if left, err = FromFloat32([]int64{rows, inner}, leftValues); err != nil {
		t.Fatal(err)
	}
	if leftT, err = FromFloat32([]int64{inner, rows}, leftTValues); err != nil {
		t.Fatal(err)
	}
	if right, err = FromFloat32([]int64{inner, cols}, rightValues); err != nil {
		t.Fatal(err)
	}
	if rightT, err = FromFloat32([]int64{cols, inner}, rightTValues); err != nil {
		t.Fatal(err)
	}
	if bias, err = FromFloat32([]int64{cols}, biasValues); err != nil {
		t.Fatal(err)
	}
	if matMulOut, err = New(Float32, []int64{rows, cols}); err != nil {
		t.Fatal(err)
	}
	if gemmOut, err = New(Float32, []int64{rows, cols}); err != nil {
		t.Fatal(err)
	}
	if preparedOut, err = New(Float32, []int64{rows, cols}); err != nil {
		t.Fatal(err)
	}

	if preparedWeight, err = PrepareRightTranspose([]int64{cols, inner}, rightTValues); err != nil {
		t.Fatal(err)
	}
	if preparedWeight.packed != float32product.CanPackRightTranspose(inner, cols) {
		t.Fatalf("preparedWeight.packed = %v, want %v", preparedWeight.packed, float32product.CanPackRightTranspose(inner, cols))
	}

	before = runtimeValue.ResourceSnapshot()

	if err = MatMulInto(left, right, matMulOut); err != nil {
		t.Fatal(err)
	}

	gemmConfigs = []GemmConfig{
		{Alpha: 1, Beta: 1},
		{TransposeA: true, Alpha: 1, Beta: 1},
		{TransposeB: true, Alpha: 1, Beta: 1},
		{TransposeA: true, TransposeB: true, Alpha: 1, Beta: 1},
	}
	for _, config = range gemmConfigs {
		var (
			a = left
			b = right
		)
		if config.TransposeA {
			a = leftT
		}
		if config.TransposeB {
			b = rightT
		}
		if err = GemmInto(a, b, bias, gemmOut, config); err != nil {
			t.Fatal(err)
		}
	}

	if err = GemmInto(left, rightT, bias, gemmOut, GemmConfig{TransposeB: true, Alpha: 1, Beta: 1}); err != nil {
		t.Fatal(err)
	}
	if err = preparedWeight.GemmInto(left, bias, preparedOut, GemmConfig{TransposeB: true, Alpha: 1, Beta: 1}); err != nil {
		t.Fatal(err)
	}
	checkPreparedTestBits(t, preparedOut.floats, gemmOut.floats)

	after = runtimeValue.ResourceSnapshot()

	if after.CreatedBuffers != before.CreatedBuffers ||
		after.LiveBuffers != before.LiveBuffers ||
		after.CreatedScopes != before.CreatedScopes ||
		after.SubmittedCommands != before.SubmittedCommands ||
		after.CompletedCommands != before.CompletedCommands {
		t.Fatalf("tensor matrix products triggered Metal activity: before=%+v after=%+v", before, after)
	}
}
