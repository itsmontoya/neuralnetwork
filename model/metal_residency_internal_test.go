//go:build darwin && cgo && metal && !purego

package model_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

func Test_MetalParameterMutationAndSerializationCoherence(t *testing.T) {
	var (
		dense            *layer.Dense
		hostDense        *layer.Dense
		network          *model.Sequential
		hostNetwork      *model.Sequential
		loaded           *model.Sequential
		weights          *matrix.Matrix
		right            *matrix.Matrix
		result           *matrix.Matrix
		loadedParameters []*optimizer.Parameter
		want             []float32
		counters         metaltest.Counters
		buffer           bytes.Buffer
		hostBuffer       bytes.Buffer
		available        bool
		err              error
	)
	if _, available, err = device.SharedRuntime(); err != nil {
		t.Fatalf("SharedRuntime returned error: %v", err)
	}
	if !available {
		t.Skip("Metal device unavailable")
	}

	if dense, err = layer.NewDense(256, 128, layer.ZeroWeights); err != nil {
		t.Fatalf("NewDense returned error: %v", err)
	}
	if network, err = model.NewSequential(dense); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	weights = dense.Weights().Values()
	right = metalResidencyMatrix(t, 128, 128, 0.25)
	if result, err = matrix.New(256, 128); err != nil {
		t.Fatalf("New result returned error: %v", err)
	}

	metaltest.Enable()
	defer metaltest.Disable()
	if err = weights.MatMulInto(right, result); err != nil {
		t.Fatalf("first parameter MatMulInto returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.InputUploads != 2 {
		t.Fatalf("first parameter uploads = %d, want 2", counters.InputUploads)
	}

	if err = weights.Set(0, 0, 1); err != nil {
		t.Fatalf("parameter Set returned error: %v", err)
	}
	if err = weights.MatMulInto(right, result); err != nil {
		t.Fatalf("parameter MatMulInto after mutation returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.InputUploads != 3 {
		t.Fatalf("uploads after actual parameter mutation = %d, want 3", counters.InputUploads)
	}

	if err = result.MatMulInto(right, weights); err != nil {
		t.Fatalf("device-newer parameter destination returned error: %v", err)
	}
	metaltest.Reset()
	if err = network.Save(&buffer); err != nil {
		t.Fatalf("Save device-newer parameter returned error: %v", err)
	}
	counters = metaltest.Snapshot()
	if counters.ResultDownloads != 1 {
		t.Fatalf("Save result downloads = %d, want 1", counters.ResultDownloads)
	}
	if strings.Contains(buffer.String(), "residency") ||
		strings.Contains(buffer.String(), "revision") ||
		strings.Contains(buffer.String(), "device") {
		t.Fatalf("serialized model contains runtime residency metadata: %s", buffer.String())
	}

	if want, err = weights.Values(); err != nil {
		t.Fatalf("parameter Values after Save returned error: %v", err)
	}
	if hostDense, err = layer.NewDense(256, 128, func(_, _ int) (initialized *matrix.Matrix, err error) {
		initialized, err = matrix.FromSlice(256, 128, want)
		return initialized, err
	}); err != nil {
		t.Fatalf("NewDense host equivalent returned error: %v", err)
	}
	if hostNetwork, err = model.NewSequential(hostDense); err != nil {
		t.Fatalf("NewSequential host equivalent returned error: %v", err)
	}
	if err = hostNetwork.Save(&hostBuffer); err != nil {
		t.Fatalf("Save host equivalent returned error: %v", err)
	}
	if !bytes.Equal(buffer.Bytes(), hostBuffer.Bytes()) {
		t.Fatal("resident and host-equivalent serialized version 1 documents differ")
	}
	if loaded, err = model.LoadSequential(bytes.NewReader(buffer.Bytes())); err != nil {
		t.Fatalf("LoadSequential returned error: %v", err)
	}
	loadedParameters = loaded.Parameters()
	if len(loadedParameters) != 2 {
		t.Fatalf("loaded parameter count = %d, want 2", len(loadedParameters))
	}
	requireMetalResidencyValues(t, loadedParameters[0].Values(), want)
}

func metalResidencyMatrix(tb testing.TB, rows, cols int, offset float32) (out *matrix.Matrix) {
	tb.Helper()

	var (
		values []float32
		index  int
		err    error
	)

	values = make([]float32, rows*cols)
	for index = range values {
		values[index] = offset + float32(index%17)/17
	}
	if out, err = matrix.FromSlice(rows, cols, values); err != nil {
		tb.Fatalf("FromSlice returned error: %v", err)
	}
	return out
}

func requireMetalResidencyValues(tb testing.TB, got *matrix.Matrix, want []float32) {
	tb.Helper()

	var (
		values []float32
		index  int
		err    error
	)

	if values, err = got.Values(); err != nil {
		tb.Fatalf("Values returned error: %v", err)
	}
	if len(values) != len(want) {
		tb.Fatalf("value length = %d, want %d", len(values), len(want))
	}
	for index = range want {
		if values[index] != want[index] {
			tb.Fatalf("value %d = %g, want %g", index, values[index], want[index])
		}
	}
}
