package data_test

import (
	"bytes"
	"io"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_LoadCSV_ReadsDatasetWithHeader(t *testing.T) {
	var (
		reader  *strings.Reader
		config  data.CSVConfig
		dataset *data.Dataset
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	reader = strings.NewReader("x1,x2,y\n1,2,0\n3.5,4.5,1\n")
	config = data.CSVConfig{
		InputColumns:  2,
		TargetColumns: 1,
		HasHeader:     true,
	}

	dataset, err = data.LoadCSV(reader, config)
	if err != nil {
		t.Fatalf("LoadCSV returned error: %v", err)
	}

	if dataset.SampleCount() != 2 {
		t.Fatalf("SampleCount = %d, want 2", dataset.SampleCount())
	}

	if dataset.InputSize() != 2 {
		t.Fatalf("InputSize = %d, want 2", dataset.InputSize())
	}

	if dataset.TargetSize() != 1 {
		t.Fatalf("TargetSize = %d, want 1", dataset.TargetSize())
	}

	inputs, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	targets, err = dataset.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	requireMatrixValues(t, inputs, []float32{1, 2, 3.5, 4.5})
	requireMatrixValues(t, targets, []float32{0, 1})
}

func Test_LoadCSV_ReadsMultipleTargetsWithoutHeader(t *testing.T) {
	var (
		reader  *strings.Reader
		config  data.CSVConfig
		dataset *data.Dataset
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	reader = strings.NewReader("1,2,10,20\n3,4,30,40\n")
	config = data.CSVConfig{
		InputColumns:  2,
		TargetColumns: 2,
	}

	dataset, err = data.LoadCSV(reader, config)
	if err != nil {
		t.Fatalf("LoadCSV returned error: %v", err)
	}

	inputs, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	targets, err = dataset.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	requireMatrixValues(t, inputs, []float32{1, 2, 3, 4})
	requireMatrixValues(t, targets, []float32{10, 20, 30, 40})
}

func Test_LoadCSV_TrimsWhitespaceAndSkipsBlankRecords(t *testing.T) {
	var (
		reader  *strings.Reader
		config  data.CSVConfig
		dataset *data.Dataset
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	reader = strings.NewReader("x,y\n 1 , 10 \n   \n2,20\n")
	config = data.CSVConfig{
		InputColumns:  1,
		TargetColumns: 1,
		HasHeader:     true,
	}

	dataset, err = data.LoadCSV(reader, config)
	if err != nil {
		t.Fatalf("LoadCSV returned error: %v", err)
	}

	inputs, err = dataset.Inputs()
	if err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	targets, err = dataset.Targets()
	if err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	requireMatrixValues(t, inputs, []float32{1, 2})
	requireMatrixValues(t, targets, []float32{10, 20})
}

func Test_LoadCSV_DoesNotRetainReusedRecordsOrSource(t *testing.T) {
	var (
		csvData         []byte
		config          data.CSVConfig
		dataset         *data.Dataset
		inputs          *matrix.Matrix
		targets         *matrix.Matrix
		returnedInputs  *matrix.Matrix
		returnedTargets *matrix.Matrix
		index           int
		err             error
	)

	csvData = []byte("1,10\n2,20\n3,30\n")
	config.InputColumns = 1
	config.TargetColumns = 1
	if dataset, err = data.LoadCSV(bytes.NewReader(csvData), config); err != nil {
		t.Fatalf("LoadCSV returned error: %v", err)
	}

	for index = range csvData {
		if csvData[index] >= '0' && csvData[index] <= '9' {
			csvData[index] = '9'
		}
	}

	if inputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("Inputs returned error: %v", err)
	}

	if targets, err = dataset.Targets(); err != nil {
		t.Fatalf("Targets returned error: %v", err)
	}

	requireMatrixValues(t, inputs, []float32{1, 2, 3})
	requireMatrixValues(t, targets, []float32{10, 20, 30})

	if err = inputs.Set(0, 0, 99); err != nil {
		t.Fatalf("inputs Set returned error: %v", err)
	}

	if err = targets.Set(0, 0, 98); err != nil {
		t.Fatalf("targets Set returned error: %v", err)
	}

	if returnedInputs, err = dataset.Inputs(); err != nil {
		t.Fatalf("second Inputs returned error: %v", err)
	}

	if returnedTargets, err = dataset.Targets(); err != nil {
		t.Fatalf("second Targets returned error: %v", err)
	}

	requireMatrixValues(t, returnedInputs, []float32{1, 2, 3})
	requireMatrixValues(t, returnedTargets, []float32{10, 20, 30})
}

func Test_LoadCSV_RejectsInvalidInput(t *testing.T) {
	type testcase struct {
		name   string
		reader io.Reader
		config data.CSVConfig
	}

	tests := []testcase{
		{
			name:   "nil reader",
			reader: nil,
			config: data.CSVConfig{
				InputColumns:  1,
				TargetColumns: 1,
			},
		},
		{
			name:   "input columns",
			reader: strings.NewReader("1,2\n"),
			config: data.CSVConfig{
				InputColumns:  0,
				TargetColumns: 1,
			},
		},
		{
			name:   "target columns",
			reader: strings.NewReader("1,2\n"),
			config: data.CSVConfig{
				InputColumns:  1,
				TargetColumns: 0,
			},
		},
		{
			name:   "empty",
			reader: strings.NewReader("x,y\n"),
			config: data.CSVConfig{
				InputColumns:  1,
				TargetColumns: 1,
				HasHeader:     true,
			},
		},
		{
			name:   "column mismatch",
			reader: strings.NewReader("1,2,3\n"),
			config: data.CSVConfig{
				InputColumns:  1,
				TargetColumns: 1,
			},
		},
		{
			name:   "parse error",
			reader: strings.NewReader("1,nope\n"),
			config: data.CSVConfig{
				InputColumns:  1,
				TargetColumns: 1,
			},
		},
		{
			name:   "empty value",
			reader: strings.NewReader("1,\n"),
			config: data.CSVConfig{
				InputColumns:  1,
				TargetColumns: 1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				dataset *data.Dataset
				err     error
			)

			dataset, err = data.LoadCSV(tt.reader, tt.config)
			if err == nil {
				t.Fatal("LoadCSV error = nil, want error")
			}

			if dataset != nil {
				t.Fatal("LoadCSV returned dataset on error")
			}
		})
	}
}
