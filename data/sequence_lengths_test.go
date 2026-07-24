package data_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
)

func Test_NewSequenceLengths_ValidatesValues(t *testing.T) {
	type testcase struct {
		name       string
		steps      int
		values     []int
		wantDetail string
	}

	var tests []testcase
	tests = []testcase{
		{
			name:       "zero steps",
			steps:      0,
			values:     []int{1},
			wantDetail: "steps=0",
		},
		{
			name:       "negative steps",
			steps:      -1,
			values:     []int{1},
			wantDetail: "steps=-1",
		},
		{
			name:       "nil values",
			steps:      2,
			values:     nil,
			wantDetail: "values are empty",
		},
		{
			name:       "empty values",
			steps:      2,
			values:     []int{},
			wantDetail: "values are empty",
		},
		{
			name:       "zero length",
			steps:      2,
			values:     []int{1, 0},
			wantDetail: "row=1 value=0",
		},
		{
			name:       "negative length",
			steps:      2,
			values:     []int{-1},
			wantDetail: "row=0 value=-1",
		},
		{
			name:       "length exceeds steps",
			steps:      2,
			values:     []int{1, 3},
			wantDetail: "row=1 value=3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				lengths *data.SequenceLengths
				err     error
			)

			lengths, err = data.NewSequenceLengths(tt.steps, tt.values)
			if err == nil {
				t.Fatal("NewSequenceLengths error = nil, want error")
			}

			if lengths != nil {
				t.Fatal("NewSequenceLengths returned lengths on error")
			}

			if !strings.Contains(err.Error(), "data: sequence lengths") {
				t.Fatalf("error = %q, want sequence lengths context", err)
			}

			if !strings.Contains(err.Error(), tt.wantDetail) {
				t.Fatalf("error = %q, want detail %q", err, tt.wantDetail)
			}
		})
	}
}

func Test_SequenceLengths_ValidatesNilAndZeroValues(t *testing.T) {
	type testcase struct {
		name    string
		lengths *data.SequenceLengths
	}

	var (
		zero  data.SequenceLengths
		tests []testcase
	)
	tests = []testcase{
		{name: "nil", lengths: nil},
		{name: "zero", lengths: &zero},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				values []int
				err    error
			)

			if err = tt.lengths.Validate(); err == nil {
				t.Fatal("Validate error = nil, want error")
			}

			if values, err = tt.lengths.Values(); err == nil {
				t.Fatal("Values error = nil, want error")
			}

			if values != nil {
				t.Fatal("Values returned values on error")
			}
		})
	}
}

func Test_SequenceLengths_CopiesCallerValuesAndAccessorResults(t *testing.T) {
	var (
		source      []int
		lengths     *data.SequenceLengths
		first       []int
		second      []int
		destination []int
		err         error
	)

	source = []int{1, 3, 2}
	lengths, err = data.NewSequenceLengths(3, source)
	if err != nil {
		t.Fatalf("NewSequenceLengths returned error: %v", err)
	}

	source[0] = 3
	first, err = lengths.Values()
	if err != nil {
		t.Fatalf("Values returned error: %v", err)
	}

	first[1] = 1
	second, err = lengths.Values()
	if err != nil {
		t.Fatalf("Values returned error: %v", err)
	}

	destination = []int{9, 9, 9}
	if err = lengths.ValuesInto(destination); err != nil {
		t.Fatalf("ValuesInto returned error: %v", err)
	}

	requireIntValues(t, second, []int{1, 3, 2})
	requireIntValues(t, destination, []int{1, 3, 2})
	if lengths.Steps() != 3 {
		t.Fatalf("Steps = %d, want 3", lengths.Steps())
	}

	if lengths.SampleCount() != 3 {
		t.Fatalf("SampleCount = %d, want 3", lengths.SampleCount())
	}
}

func Test_SequenceLengths_SelectRowsIntoPreservesOrderAndRepeats(t *testing.T) {
	var (
		lengths     *data.SequenceLengths
		destination []int
		err         error
	)

	lengths, err = data.NewSequenceLengths(4, []int{1, 2, 3, 4})
	if err != nil {
		t.Fatalf("NewSequenceLengths returned error: %v", err)
	}

	destination = []int{9, 9, 9, 9}
	if err = lengths.SelectRowsInto([]int{3, 0, 3, 1}, destination); err != nil {
		t.Fatalf("SelectRowsInto returned error: %v", err)
	}

	requireIntValues(t, destination, []int{4, 1, 4, 2})
}

func Test_SequenceLengths_DestinationOperationsRejectInvalidArgumentsBeforeWrite(t *testing.T) {
	type testcase struct {
		name        string
		indexes     []int
		destination []int
		valuesOnly  bool
	}

	var (
		lengths *data.SequenceLengths
		tests   []testcase
		err     error
	)

	lengths, err = data.NewSequenceLengths(3, []int{1, 2, 3})
	if err != nil {
		t.Fatalf("NewSequenceLengths returned error: %v", err)
	}

	tests = []testcase{
		{
			name:        "values destination too short",
			destination: []int{9, 9},
			valuesOnly:  true,
		},
		{
			name:        "empty indexes",
			indexes:     []int{},
			destination: []int{9},
		},
		{
			name:        "negative index",
			indexes:     []int{0, -1},
			destination: []int{9, 9},
		},
		{
			name:        "index too large",
			indexes:     []int{0, 3},
			destination: []int{9, 9},
		},
		{
			name:        "selected destination too short",
			indexes:     []int{0, 1},
			destination: []int{9},
		},
		{
			name:        "selected destination too long",
			indexes:     []int{0, 1},
			destination: []int{9, 9, 9},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				before []int
				gotErr error
			)

			before = append([]int(nil), tt.destination...)
			if tt.valuesOnly {
				gotErr = lengths.ValuesInto(tt.destination)
			} else {
				gotErr = lengths.SelectRowsInto(tt.indexes, tt.destination)
			}

			if gotErr == nil {
				t.Fatal("destination operation error = nil, want error")
			}

			requireIntValues(t, tt.destination, before)
		})
	}
}

func requireIntValues(tb testing.TB, got, want []int) {
	tb.Helper()

	if !reflect.DeepEqual(got, want) {
		tb.Fatalf("values = %v, want %v", got, want)
	}
}
