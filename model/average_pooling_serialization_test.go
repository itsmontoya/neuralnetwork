package model_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/model"
)

func Test_Sequential_SaveLoadRoundTripWithAveragePooling(t *testing.T) {
	type testcase struct {
		name       string
		current    layer.Layer
		input      *matrix.Matrix
		wantFields []string
	}

	var (
		inputShape     layer.SpatialShape
		averageConfig  layer.AveragePool2DConfig
		adaptiveConfig layer.AdaptiveAveragePool2DConfig
		average        *layer.AveragePool2D
		adaptive       *layer.AdaptiveAveragePool2D
		tests          []testcase
		err            error
	)

	inputShape = mustSerializationSpatialShape(t, 1, 3, 5)
	if averageConfig, err = layer.NewAveragePool2DConfig(inputShape, 2, 2, 2, 2); err != nil {
		t.Fatalf("NewAveragePool2DConfig returned error: %v", err)
	}
	if average, err = layer.NewAveragePool2D(averageConfig); err != nil {
		t.Fatalf("NewAveragePool2D returned error: %v", err)
	}
	if adaptiveConfig, err = layer.NewAdaptiveAveragePool2DConfig(inputShape, 2, 2); err != nil {
		t.Fatalf("NewAdaptiveAveragePool2DConfig returned error: %v", err)
	}
	if adaptive, err = layer.NewAdaptiveAveragePool2D(adaptiveConfig); err != nil {
		t.Fatalf("NewAdaptiveAveragePool2D returned error: %v", err)
	}
	tests = []testcase{
		{
			name:    "average",
			current: average,
			input: mustMatrix(t, 1, 15, []float32{
				1, 2, 3, 4, 5,
				6, 7, 8, 9, 10,
				11, 12, 13, 14, 15,
			}),
			wantFields: []string{
				`"type": "average_pool2d"`,
				`"window_height": 2`,
				`"window_width": 2`,
				`"stride_height": 2`,
				`"stride_width": 2`,
			},
		},
		{
			name:    "adaptive average",
			current: adaptive,
			input: mustMatrix(t, 1, 15, []float32{
				1, 2, 3, 4, 5,
				6, 7, 8, 9, 10,
				11, 12, 13, 14, 15,
			}),
			wantFields: []string{
				`"type": "adaptive_average_pool2d"`,
				`"output_height": 2`,
				`"output_width": 2`,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				network        *model.Sequential
				jsonLoaded     *model.Sequential
				binaryLoaded   *model.Sequential
				before         *matrix.Matrix
				afterJSON      *matrix.Matrix
				afterBinary    *matrix.Matrix
				jsonDocument   bytes.Buffer
				binaryDocument bytes.Buffer
				field          string
				err            error
			)

			if network, err = model.NewSequential(tt.current); err != nil {
				t.Fatalf("NewSequential returned error: %v", err)
			}
			if before, err = network.Predict(tt.input); err != nil {
				t.Fatalf("Predict returned error: %v", err)
			}
			if err = network.Save(&jsonDocument); err != nil {
				t.Fatalf("Save returned error: %v", err)
			}
			for _, field = range tt.wantFields {
				if !strings.Contains(jsonDocument.String(), field) {
					t.Fatalf("JSON document missing %q: %s", field, jsonDocument.String())
				}
			}
			if jsonLoaded, err = model.LoadSequential(bytes.NewReader(jsonDocument.Bytes())); err != nil {
				t.Fatalf("LoadSequential returned error: %v", err)
			}
			if afterJSON, err = jsonLoaded.Predict(tt.input); err != nil {
				t.Fatalf("JSON loaded Predict returned error: %v", err)
			}
			if _, err = network.SaveBinary(&binaryDocument); err != nil {
				t.Fatalf("SaveBinary returned error: %v", err)
			}
			if binaryLoaded, _, err = model.LoadSequentialBinary(bytes.NewReader(binaryDocument.Bytes())); err != nil {
				t.Fatalf("LoadSequentialBinary returned error: %v", err)
			}
			if afterBinary, err = binaryLoaded.Predict(tt.input); err != nil {
				t.Fatalf("binary loaded Predict returned error: %v", err)
			}

			requireMatrixValues(t, afterJSON, mustValues(t, before))
			requireMatrixValues(t, afterBinary, mustValues(t, before))
		})
	}
}

func Test_LoadSequential_RejectsMalformedAveragePooling(t *testing.T) {
	type testcase struct {
		name      string
		layerJSON string
		wantError string
	}

	var (
		tests    []testcase
		tt       testcase
		document string
		loaded   *model.Sequential
		err      error
	)

	tests = []testcase{
		{
			name:      "average missing input shape",
			layerJSON: `{"type":"average_pool2d","window_height":2,"window_width":2,"stride_height":1,"stride_width":1}`,
			wantError: "average_pool2d input shape is missing",
		},
		{
			name:      "average invalid window",
			layerJSON: `{"type":"average_pool2d","input_channels":1,"input_height":2,"input_width":2,"window_height":3,"window_width":2,"stride_height":1,"stride_width":1}`,
			wantError: "average pool2d configuration load failed",
		},
		{
			name:      "adaptive missing input shape",
			layerJSON: `{"type":"adaptive_average_pool2d","output_height":1,"output_width":1}`,
			wantError: "adaptive_average_pool2d input shape is missing",
		},
		{
			name:      "adaptive invalid output",
			layerJSON: `{"type":"adaptive_average_pool2d","input_channels":1,"input_height":2,"input_width":2,"output_height":0,"output_width":1}`,
			wantError: "adaptive average pool2d configuration load failed",
		},
	}

	for _, tt = range tests {
		t.Run(tt.name, func(t *testing.T) {
			document = `{"format":"neuralnetwork.sequential","version":1,"layers":[` + tt.layerJSON + `]}`
			loaded, err = model.LoadSequential(strings.NewReader(document))
			if err == nil {
				t.Fatal("LoadSequential error = nil, want error")
			}
			if loaded != nil {
				t.Fatal("LoadSequential returned model on error")
			}
			if !strings.Contains(err.Error(), "layer 0") || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("LoadSequential error = %q, want layer context and %q", err, tt.wantError)
			}
		})
	}
}
