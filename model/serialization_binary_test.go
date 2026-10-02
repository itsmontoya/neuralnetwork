package model_test

import (
	"bytes"
	"math/rand"
	"strconv"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_Sequential_SaveLoadBinaryRoundTripsEverySupportedLayer(t *testing.T) {
	var (
		layers       []layer.Layer
		currentLayer layer.Layer
		index        int
	)

	layers = binarySerializationLayers(t)
	for index, currentLayer = range layers {
		t.Run(binarySerializationLayerName(index), func(t *testing.T) {
			var (
				original       *model.Sequential
				loaded         *model.Sequential
				firstBinary    bytes.Buffer
				secondBinary   bytes.Buffer
				firstJSON      bytes.Buffer
				secondJSON     bytes.Buffer
				firstChecksum  model.ModelChecksum
				loadedChecksum model.ModelChecksum
				secondChecksum model.ModelChecksum
				err            error
			)

			if original, err = model.NewSequential(currentLayer); err != nil {
				t.Fatalf("NewSequential returned error: %v", err)
			}
			if firstChecksum, err = original.SaveBinary(&firstBinary); err != nil {
				t.Fatalf("SaveBinary returned error: %v", err)
			}
			if loaded, loadedChecksum, err = model.LoadSequentialBinary(bytes.NewReader(firstBinary.Bytes())); err != nil {
				t.Fatalf("LoadSequentialBinary returned error: %v", err)
			}
			if secondChecksum, err = loaded.SaveBinary(&secondBinary); err != nil {
				t.Fatalf("loaded SaveBinary returned error: %v", err)
			}

			if firstChecksum != loadedChecksum || firstChecksum != secondChecksum {
				t.Fatal("binary checksums differ across round trip")
			}
			if !bytes.Equal(firstBinary.Bytes(), secondBinary.Bytes()) {
				t.Fatal("binary encoding is not deterministic across round trip")
			}
			if err = original.Save(&firstJSON); err != nil {
				t.Fatalf("Save returned error: %v", err)
			}
			if err = loaded.Save(&secondJSON); err != nil {
				t.Fatalf("loaded Save returned error: %v", err)
			}
			if !bytes.Equal(firstJSON.Bytes(), secondJSON.Bytes()) {
				t.Fatalf("JSON state differs after binary round trip:\nfirst:\n%s\nsecond:\n%s", firstJSON.String(), secondJSON.String())
			}
		})
	}
}

func Test_Sequential_SaveLoadBinaryPreservesPrediction(t *testing.T) {
	var (
		network        *model.Sequential
		loaded         *model.Sequential
		input          *matrix.Matrix
		before         *matrix.Matrix
		after          *matrix.Matrix
		document       bytes.Buffer
		savedChecksum  model.ModelChecksum
		loadedChecksum model.ModelChecksum
		err            error
	)

	network = binarySerializationCNN(t)
	input = mustMatrix(t, 2, 16, []float32{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
		-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
	})
	if before, err = network.Predict(input); err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}
	if savedChecksum, err = network.SaveBinary(&document); err != nil {
		t.Fatalf("SaveBinary returned error: %v", err)
	}
	if loaded, loadedChecksum, err = model.LoadSequentialBinary(bytes.NewReader(document.Bytes())); err != nil {
		t.Fatalf("LoadSequentialBinary returned error: %v", err)
	}
	if after, err = loaded.Predict(input); err != nil {
		t.Fatalf("loaded Predict returned error: %v", err)
	}

	if savedChecksum != loadedChecksum {
		t.Fatal("loaded checksum differs from saved checksum")
	}
	if len(savedChecksum.String()) != 64 {
		t.Fatalf("checksum string length = %d, want 64", len(savedChecksum.String()))
	}
	requireMatrixValues(t, after, mustValues(t, before))
}

func Test_LoadSequentialBinaryRejectsInvalidArtifacts(t *testing.T) {
	type testcase struct {
		name      string
		document  []byte
		wantError string
	}

	var (
		valid    []byte
		tests    []testcase
		mutated  []byte
		trailing []byte
	)

	valid = binarySerializationDocument(t)
	mutated = append([]byte(nil), valid...)
	mutated[0] ^= 0xff
	tests = append(tests, testcase{name: "magic", document: mutated, wantError: "unsupported magic"})
	mutated = append([]byte(nil), valid...)
	mutated[8] = 2
	tests = append(tests, testcase{name: "version", document: mutated, wantError: "unsupported version 2"})
	mutated = append([]byte(nil), valid...)
	mutated[len(mutated)-33] ^= 0x01
	tests = append(tests, testcase{name: "payload corruption", document: mutated, wantError: "checksum mismatch"})
	mutated = append([]byte(nil), valid...)
	mutated[len(mutated)-1] ^= 0x01
	tests = append(tests, testcase{name: "checksum corruption", document: mutated, wantError: "checksum mismatch"})
	trailing = append(append([]byte(nil), valid...), 0)
	tests = append(tests, testcase{name: "trailing data", document: trailing, wantError: "trailing data"})

	for _, cut := range []int{0, 1, 7, 8, 9, len(valid) / 2, len(valid) - 1} {
		tests = append(tests, testcase{
			name:      "truncated at " + strconv.Itoa(cut),
			document:  valid[:cut],
			wantError: "read",
		})
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				loaded   *model.Sequential
				checksum model.ModelChecksum
				err      error
			)

			loaded, checksum, err = model.LoadSequentialBinary(bytes.NewReader(tt.document))
			if err == nil {
				t.Fatal("LoadSequentialBinary error = nil, want error")
			}
			if loaded != nil {
				t.Fatal("LoadSequentialBinary returned model on error")
			}
			if checksum != (model.ModelChecksum{}) {
				t.Fatal("LoadSequentialBinary returned checksum on error")
			}
			if !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("LoadSequentialBinary error = %q, want substring %q", err, tt.wantError)
			}
		})
	}
}

func Test_SequentialBinaryRejectsNilIO(t *testing.T) {
	var (
		network  *model.Sequential
		loaded   *model.Sequential
		checksum model.ModelChecksum
		err      error
	)

	if network, err = model.NewSequential(); err != nil {
		t.Fatalf("NewSequential returned error: %v", err)
	}
	if checksum, err = network.SaveBinary(nil); err == nil || !strings.Contains(err.Error(), "writer is nil") {
		t.Fatalf("SaveBinary error = %v, want nil writer error", err)
	}
	if checksum != (model.ModelChecksum{}) {
		t.Fatal("SaveBinary returned checksum on error")
	}
	if loaded, checksum, err = model.LoadSequentialBinary(nil); err == nil || !strings.Contains(err.Error(), "reader is nil") {
		t.Fatalf("LoadSequentialBinary error = %v, want nil reader error", err)
	}
	if loaded != nil || checksum != (model.ModelChecksum{}) {
		t.Fatal("LoadSequentialBinary returned output on error")
	}
}

func FuzzLoadSequentialBinary(f *testing.F) {
	var valid []byte

	valid = binarySerializationDocument(f)
	f.Add([]byte{})
	f.Add(valid)
	f.Fuzz(func(t *testing.T, document []byte) {
		_, _, _ = model.LoadSequentialBinary(bytes.NewReader(document))
	})
}

func binarySerializationLayers(tb testing.TB) (layers []layer.Layer) {
	var (
		segmented         *activation.SegmentedSoftmax
		batchNorm         *layer.BatchNormalization
		batchNorm2D       *layer.BatchNormalization2D
		dropout           *layer.Dropout
		spatialShape      layer.SpatialShape
		convConfig        layer.Conv2DConfig
		poolConfig        layer.MaxPool2DConfig
		batchNorm2DConfig layer.BatchNormalization2DConfig
		err               error
	)

	tb.Helper()
	if segmented, err = activation.NewSegmentedSoftmax(2, 3); err != nil {
		tb.Fatalf("NewSegmentedSoftmax returned error: %v", err)
	}
	if batchNorm, err = layer.NewBatchNormalizationWithConfig(3, 0.8, 1e-4); err != nil {
		tb.Fatalf("NewBatchNormalizationWithConfig returned error: %v", err)
	}
	spatialShape = mustSerializationSpatialShape(tb, 2, 3, 4)
	batchNorm2DConfig.InputShape = spatialShape
	batchNorm2DConfig.Momentum = 0.75
	batchNorm2DConfig.Epsilon = 1e-4
	if batchNorm2D, err = layer.NewBatchNormalization2D(batchNorm2DConfig); err != nil {
		tb.Fatalf("NewBatchNormalization2D returned error: %v", err)
	}
	convConfig = mustSerializationConv2DConfig(tb, spatialShape, 3, 2, 2, 1, 1, 0, 0)
	poolConfig = mustSerializationMaxPool2DConfig(tb, spatialShape, 2, 2, 1, 1)
	if dropout, err = layer.NewDropout(0.25, rand.New(rand.NewSource(7))); err != nil {
		tb.Fatalf("NewDropout returned error: %v", err)
	}

	layers = []layer.Layer{
		mustActivationLayer(tb, segmented),
		batchNorm,
		batchNorm2D,
		mustSerializationConv2D(tb, convConfig, make([]float32, 2*2*2*3), make([]float32, 3)),
		mustSerializationDense(tb, 2, 3, []float32{1, 2, 3, 4, 5, 6}, []float32{7, 8, 9}),
		dropout,
		mustSerializationFlatten(tb, 2, 3, 4),
		mustSerializationGatherLastValid(tb, 3, 2),
		mustSerializationLastStep(tb, 3, 2),
		mustSerializationMaxPool2D(tb, poolConfig),
		mustSerializationSimpleRNN(
			tb,
			3,
			2,
			2,
			[]float32{0.1, 0.2, 0.3, 0.4},
			[]float32{0.5, 0.6, 0.7, 0.8},
			[]float32{0.9, 1},
		),
	}
	return layers
}

func binarySerializationCNN(tb testing.TB) (network *model.Sequential) {
	var (
		inputShape  layer.SpatialShape
		convConfig  layer.Conv2DConfig
		poolConfig  layer.MaxPool2DConfig
		convolution *layer.Conv2D
		relu        *layer.Activation
		pooling     *layer.MaxPool2D
		flatten     *layer.Flatten
		output      *layer.Dense
		err         error
	)

	tb.Helper()
	inputShape = mustSerializationSpatialShape(tb, 1, 4, 4)
	convConfig = mustSerializationConv2DConfig(tb, inputShape, 2, 3, 3, 1, 1, 1, 1)
	convolution = mustSerializationConv2D(
		tb,
		convConfig,
		[]float32{
			0.1, -0.1, 0.2, -0.2, 0.3, -0.3,
			0.4, -0.4, 0.5, -0.5, 0.6, -0.6,
			0.7, -0.7, 0.8, -0.8, 0.9, -0.9,
		},
		[]float32{0.25, -0.25},
	)
	relu = mustActivationLayer(tb, activation.ReLU{})
	poolConfig = mustSerializationMaxPool2DConfig(tb, convolution.OutputShape(), 2, 2, 2, 2)
	pooling = mustSerializationMaxPool2D(tb, poolConfig)
	if flatten, err = layer.NewFlatten(pooling.OutputShape()); err != nil {
		tb.Fatalf("NewFlatten returned error: %v", err)
	}
	output = mustSerializationDense(
		tb,
		flatten.OutputSize(),
		3,
		[]float32{
			0.1, 0.2, 0.3,
			0.4, 0.5, 0.6,
			0.7, 0.8, 0.9,
			1, 1.1, 1.2,
			1.3, 1.4, 1.5,
			1.6, 1.7, 1.8,
			1.9, 2, 2.1,
			2.2, 2.3, 2.4,
		},
		[]float32{0.1, 0.2, 0.3},
	)
	if network, err = model.NewSequential(convolution, relu, pooling, flatten, output); err != nil {
		tb.Fatalf("NewSequential returned error: %v", err)
	}
	return network
}

func binarySerializationDocument(tb testing.TB) (document []byte) {
	var (
		network *model.Sequential
		buffer  bytes.Buffer
		err     error
	)

	tb.Helper()
	network = binarySerializationCNN(tb)
	if _, err = network.SaveBinary(&buffer); err != nil {
		tb.Fatalf("SaveBinary returned error: %v", err)
	}
	document = append([]byte(nil), buffer.Bytes()...)
	return document
}

func binarySerializationLayerName(index int) (name string) {
	var names []string

	names = []string{
		"activation",
		"batch_normalization",
		"batch_normalization2d",
		"conv2d",
		"dense",
		"dropout",
		"flatten",
		"gather_last_valid",
		"last_step",
		"max_pool2d",
		"simple_rnn",
	}
	if index >= 0 && index < len(names) {
		return names[index]
	}
	name = strconv.Itoa(index)
	return name
}
