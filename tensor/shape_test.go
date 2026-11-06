package tensor_test

import (
	"math"
	"slices"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_ValidateShape(t *testing.T) {
	type testcase struct {
		name     string
		shape    []int64
		dataType tensor.DataType
		elements int64
		bytes    int64
		wantErr  bool
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		tests = []testcase{
			{name: "float32 scalar", dataType: tensor.Float32, elements: 1, bytes: 4},
			{name: "int64 scalar", dataType: tensor.Int64, shape: []int64{}, elements: 1, bytes: 8},
			{name: "matrix", dataType: tensor.Float32, shape: []int64{2, 3}, elements: 6, bytes: 24},
			{name: "empty", dataType: tensor.Int64, shape: []int64{2, 0, 3}},
			{name: "unused large product", dataType: tensor.Float32, shape: []int64{limit, 0, limit}},
			{name: "largest float32 payload", dataType: tensor.Float32, shape: []int64{limit / 4}, elements: limit / 4, bytes: (limit / 4) * 4},
			{name: "negative after zero", dataType: tensor.Int64, shape: []int64{0, -1}, wantErr: true},
			{name: "element overflow", dataType: tensor.Float32, shape: []int64{limit, 2}, wantErr: true},
			{name: "float32 byte overflow", dataType: tensor.Float32, shape: []int64{limit/4 + 1}, wantErr: true},
			{name: "int64 byte overflow", dataType: tensor.Int64, shape: []int64{limit/8 + 1}, wantErr: true},
			{name: "invalid type scalar", dataType: tensor.Invalid, wantErr: true},
			{name: "unsupported type empty", dataType: tensor.DataType(255), shape: []int64{0}, wantErr: true},
		}
	)
	if limit < math.MaxInt64 {
		tests = append(tests, testcase{name: "host dimension overflow", dataType: tensor.Float32, shape: []int64{0, limit + 1}, wantErr: true})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				before          = slices.Clone(tt.shape)
				elements, bytes int64
				err             error
			)
			elements, bytes, err = tensor.ValidateShape(tt.shape, tt.dataType)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ValidateShape error=%v, wantErr=%v", err, tt.wantErr)
			}
			if elements != tt.elements || bytes != tt.bytes {
				t.Fatalf("sizes=(%d, %d), want (%d, %d)", elements, bytes, tt.elements, tt.bytes)
			}
			if !slices.Equal(tt.shape, before) {
				t.Fatal("shape changed")
			}
		})
	}
}

func Test_NormalizeAxis(t *testing.T) {
	type testcase struct {
		name    string
		axis    int64
		rank    int
		want    int
		wantErr bool
	}
	var tests = []testcase{
		{name: "first", rank: 3, axis: 0},
		{name: "last", rank: 3, axis: 2, want: 2},
		{name: "negative last", rank: 3, axis: -1, want: 2},
		{name: "negative first", rank: 3, axis: -3},
		{name: "scalar", wantErr: true},
		{name: "negative scalar axis", axis: -1, wantErr: true},
		{name: "negative rank", rank: -1, wantErr: true},
		{name: "past end", rank: 3, axis: 3, wantErr: true},
		{name: "before beginning", rank: 3, axis: -4, wantErr: true},
		{name: "minimum axis", rank: 3, axis: math.MinInt64, wantErr: true},
		{name: "maximum axis", rank: 3, axis: math.MaxInt64, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				axis int
				err  error
			)
			axis, err = tensor.NormalizeAxis(tt.axis, tt.rank)
			if (err != nil) != tt.wantErr || axis != tt.want {
				t.Fatalf("NormalizeAxis=(%d, %v), want axis=%d wantErr=%v", axis, err, tt.want, tt.wantErr)
			}
		})
	}
}

func Test_BroadcastShape(t *testing.T) {
	type testcase struct {
		name        string
		left, right []int64
		want        []int64
		wantErr     bool
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		tests = []testcase{
			{name: "scalars", want: []int64{}},
			{name: "scalar left", right: []int64{2, 3}, want: []int64{2, 3}},
			{name: "scalar right", left: []int64{2, 3}, want: []int64{2, 3}},
			{name: "trailing vector", left: []int64{2, 3, 4}, right: []int64{4}, want: []int64{2, 3, 4}},
			{name: "two expanded axes", left: []int64{2, 1, 4}, right: []int64{1, 3, 1}, want: []int64{2, 3, 4}},
			{name: "channel bias", left: []int64{2, 3, 4, 5}, right: []int64{1, 3, 1, 1}, want: []int64{2, 3, 4, 5}},
			{name: "zero with one", left: []int64{0, 3}, right: []int64{1, 3}, want: []int64{0, 3}},
			{name: "one with zero", left: []int64{1, 3}, right: []int64{0, 1}, want: []int64{0, 3}},
			{name: "equal zeros", left: []int64{2, 0, 3}, right: []int64{0, 1}, want: []int64{2, 0, 3}},
			{name: "empty large dimensions", left: []int64{limit, 0, limit}, want: []int64{limit, 0, limit}},
			{name: "mismatched trailing", left: []int64{2, 3}, right: []int64{2}, wantErr: true},
			{name: "mismatched rank prefix", left: []int64{2, 3, 4}, right: []int64{5, 1}, wantErr: true},
			{name: "zero with two", left: []int64{0, 3}, right: []int64{2, 3}, wantErr: true},
			{name: "negative after zero", left: []int64{0, -1}, right: []int64{1}, wantErr: true},
			{name: "right negative", left: []int64{1}, right: []int64{-1}, wantErr: true},
			{name: "input payload overflow", left: []int64{limit/4 + 1}, right: []int64{1}, wantErr: true},
			{name: "output element overflow", left: []int64{limit / 4, 1}, right: []int64{1, 8}, wantErr: true},
			{name: "output byte overflow", left: []int64{limit / 4, 1}, right: []int64{1, 2}, wantErr: true},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				leftBefore  = slices.Clone(tt.left)
				rightBefore = slices.Clone(tt.right)
				shape       []int64
				err         error
			)
			shape, err = tensor.BroadcastShape(tt.left, tt.right)
			if (err != nil) != tt.wantErr {
				t.Fatalf("BroadcastShape error=%v, wantErr=%v", err, tt.wantErr)
			}
			if !slices.Equal(shape, tt.want) || (tt.wantErr && shape != nil) {
				t.Fatalf("shape=%v, want %v", shape, tt.want)
			}
			if len(shape) > 0 {
				shape[0] = -99
			}
			if !slices.Equal(tt.left, leftBefore) || !slices.Equal(tt.right, rightBefore) {
				t.Fatal("broadcast result aliases or mutated an input shape")
			}
		})
	}
}

func Test_ValidateBroadcastTo(t *testing.T) {
	type testcase struct {
		name                string
		source, destination []int64
		wantErr             bool
	}
	var tests = []testcase{
		{name: "scalars"},
		{name: "scalar bias", destination: []int64{2, 3}},
		{name: "vector bias", source: []int64{3}, destination: []int64{2, 3}},
		{name: "column bias", source: []int64{2, 1}, destination: []int64{2, 3}},
		{name: "matrix bias", source: []int64{2, 3}, destination: []int64{2, 3}},
		{name: "one to empty", source: []int64{1}, destination: []int64{0}},
		{name: "equal empties", source: []int64{0, 3}, destination: []int64{0, 3}},
		{name: "higher source rank", source: []int64{1, 2, 3}, destination: []int64{2, 3}, wantErr: true},
		{name: "would enlarge dimension", source: []int64{2, 3}, destination: []int64{1, 3}, wantErr: true},
		{name: "incompatible", source: []int64{4}, destination: []int64{2, 3}, wantErr: true},
		{name: "empty cannot enlarge one", source: []int64{0}, destination: []int64{1}, wantErr: true},
		{name: "vector to scalar", source: []int64{1}, wantErr: true},
		{name: "negative source", source: []int64{-1}, destination: []int64{2}, wantErr: true},
		{name: "negative destination", source: []int64{1}, destination: []int64{-1}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				sourceBefore      = slices.Clone(tt.source)
				destinationBefore = slices.Clone(tt.destination)
				err               error
			)
			err = tensor.ValidateBroadcastTo(tt.source, tt.destination)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ValidateBroadcastTo error=%v, wantErr=%v", err, tt.wantErr)
			}
			if !slices.Equal(tt.source, sourceBefore) || !slices.Equal(tt.destination, destinationBefore) {
				t.Fatal("broadcast validation mutated a shape")
			}
		})
	}
}
