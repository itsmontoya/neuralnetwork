package tensor_test

import (
	"math"
	"slices"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_New(t *testing.T) {
	type testcase struct {
		name     string
		shape    []int64
		elements int64
	}

	var (
		maxHostIndex = int64(int(^uint(0) >> 1))
		tests        = []testcase{
			{name: "scalar nil shape", elements: 1},
			{name: "scalar empty shape", shape: []int64{}, elements: 1},
			{name: "vector", shape: []int64{3}, elements: 3},
			{name: "matrix", shape: []int64{2, 3}, elements: 6},
			{name: "empty vector", shape: []int64{0}},
			{name: "empty matrix", shape: []int64{0, 3}},
			{name: "empty middle dimension", shape: []int64{2, 0, 3}},
			{name: "empty avoids unused product", shape: []int64{maxHostIndex, 0, maxHostIndex}},
		}
		dataTypes = []tensor.DataType{tensor.Float32, tensor.Int64}
	)

	for _, dataType := range dataTypes {
		var typeName = "float32"
		if dataType == tensor.Int64 {
			typeName = "int64"
		}
		for _, tt := range tests {
			t.Run(tt.name+"/"+typeName, func(t *testing.T) {
				var (
					value     *tensor.Tensor
					floats    []float32
					integers  []int64
					byteWidth int64 = 4
					err       error
				)

				if value, err = tensor.New(dataType, tt.shape); err != nil {
					t.Fatal(err)
				}
				if err = value.Validate(); err != nil {
					t.Fatal(err)
				}
				if value.DataType() != dataType || value.Rank() != len(tt.shape) || value.Len() != tt.elements {
					t.Fatalf("metadata: type=%d rank=%d length=%d", value.DataType(), value.Rank(), value.Len())
				}
				if !slices.Equal(value.Shape(), tt.shape) {
					t.Fatalf("shape=%v, want %v", value.Shape(), tt.shape)
				}
				if dataType == tensor.Int64 {
					byteWidth = 8
				}
				if value.ByteSize() != tt.elements*byteWidth {
					t.Fatalf("bytes=%d, want %d", value.ByteSize(), tt.elements*byteWidth)
				}
				if dataType == tensor.Float32 {
					if floats, err = value.Float32Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(floats, make([]float32, int(tt.elements))) {
						t.Fatalf("float32 values=%v, want zero values", floats)
					}
					if _, err = value.Int64Values(); err == nil {
						t.Fatal("int64 accessor accepted float32 storage")
					}
					return
				}
				if integers, err = value.Int64Values(); err != nil {
					t.Fatal(err)
				}
				if !slices.Equal(integers, make([]int64, int(tt.elements))) {
					t.Fatalf("int64 values=%v, want zero values", integers)
				}
				if _, err = value.Float32Values(); err == nil {
					t.Fatal("float32 accessor accepted int64 storage")
				}
			})
		}
	}
}

func Test_New_Invalid(t *testing.T) {
	type testcase struct {
		name     string
		dataType tensor.DataType
		shape    []int64
	}

	var (
		maxHostIndex = int64(int(^uint(0) >> 1))
		tests        = []testcase{
			{name: "uninitialized type", dataType: tensor.Invalid},
			{name: "unsupported type", dataType: tensor.DataType(255)},
			{name: "negative", dataType: tensor.Float32, shape: []int64{-1}},
			{name: "negative after empty", dataType: tensor.Int64, shape: []int64{0, -1}},
			{name: "negative before empty", dataType: tensor.Float32, shape: []int64{-1, 0}},
			{name: "element overflow", dataType: tensor.Float32, shape: []int64{maxHostIndex, 2}},
			{name: "float32 bytes overflow", dataType: tensor.Float32, shape: []int64{maxHostIndex/4 + 1}},
			{name: "int64 bytes overflow", dataType: tensor.Int64, shape: []int64{maxHostIndex/8 + 1}},
		}
	)
	if maxHostIndex < math.MaxInt64 {
		tests = append(tests, testcase{name: "host dimension overflow", dataType: tensor.Float32, shape: []int64{0, maxHostIndex + 1}})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				value *tensor.Tensor
				err   error
			)
			if value, err = tensor.New(tt.dataType, tt.shape); err == nil || value != nil {
				t.Fatalf("New returned value=%v err=%v, want nil value and error", value, err)
			}
		})
	}
}

func Test_FromFloat32_Ownership(t *testing.T) {
	var (
		shape  = []int64{2, 2}
		input  = []float32{1, 2, 3, 4}
		want   = []float32{1, 2, 3, 4}
		value  *tensor.Tensor
		clone  *tensor.Tensor
		values []float32
		copied []int64
		err    error
	)

	if value, err = tensor.FromFloat32(shape, input); err != nil {
		t.Fatal(err)
	}
	if clone, err = value.Clone(); err != nil {
		t.Fatal(err)
	}
	shape[0] = 99
	input[0] = 99
	if values, err = value.Float32Values(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values, want) {
		t.Fatalf("constructor retained input: got %v", values)
	}
	values[1] = 99
	copied = value.Shape()
	copied[1] = 99
	for _, current := range []*tensor.Tensor{value, clone} {
		if values, err = current.Float32Values(); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(values, want) || !slices.Equal(current.Shape(), []int64{2, 2}) {
			t.Fatalf("value changed through copied input/accessor: values=%v shape=%v", values, current.Shape())
		}
	}
}

func Test_FromInt64_OwnershipAndPrecision(t *testing.T) {
	var (
		shape  = []int64{4}
		input  = []int64{16777217, -16777217, math.MaxInt64, math.MinInt64}
		want   = []int64{16777217, -16777217, math.MaxInt64, math.MinInt64}
		value  *tensor.Tensor
		clone  *tensor.Tensor
		values []int64
		copied []int64
		err    error
	)

	if value, err = tensor.FromInt64(shape, input); err != nil {
		t.Fatal(err)
	}
	if clone, err = value.Clone(); err != nil {
		t.Fatal(err)
	}
	shape[0] = 99
	input[0] = 99
	if values, err = value.Int64Values(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values, want) {
		t.Fatalf("constructor retained or converted input: got %v", values)
	}
	values[1] = 99
	copied = value.Shape()
	copied[0] = 99
	for _, current := range []*tensor.Tensor{value, clone} {
		if values, err = current.Int64Values(); err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(values, want) || !slices.Equal(current.Shape(), []int64{4}) {
			t.Fatalf("integer value changed: values=%v shape=%v", values, current.Shape())
		}
	}
}

func Test_FromFloat32_BitPatterns(t *testing.T) {
	var (
		bits   = []uint32{0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234}
		input  = make([]float32, len(bits))
		value  *tensor.Tensor
		clone  *tensor.Tensor
		values []float32
		err    error
	)
	for index, pattern := range bits {
		input[index] = math.Float32frombits(pattern)
	}
	if value, err = tensor.FromFloat32([]int64{int64(len(input))}, input); err != nil {
		t.Fatal(err)
	}
	if clone, err = value.Clone(); err != nil {
		t.Fatal(err)
	}
	for _, current := range []*tensor.Tensor{value, clone} {
		if values, err = current.Float32Values(); err != nil {
			t.Fatal(err)
		}
		for index, pattern := range bits {
			if math.Float32bits(values[index]) != pattern {
				t.Fatalf("bits at %d: got %08x, want %08x", index, math.Float32bits(values[index]), pattern)
			}
		}
	}
}

func Test_TypedConstruction(t *testing.T) {
	type testcase struct {
		name    string
		shape   []int64
		values  []int64
		wantErr bool
	}
	var tests = []testcase{
		{name: "scalar", values: []int64{7}},
		{name: "empty", shape: []int64{2, 0, 3}},
		{name: "scalar missing value", wantErr: true},
		{name: "scalar extra value", values: []int64{1, 2}, wantErr: true},
		{name: "vector too short", shape: []int64{2}, values: []int64{1}, wantErr: true},
		{name: "vector too long", shape: []int64{1}, values: []int64{1, 2}, wantErr: true},
		{name: "empty extra value", shape: []int64{0}, values: []int64{1}, wantErr: true},
		{name: "negative after zero", shape: []int64{0, -1}, wantErr: true},
		{name: "overflow", shape: []int64{math.MaxInt64, 2}, wantErr: true},
		{name: "length checked before allocation", shape: []int64{1 << 30}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				floatInput = make([]float32, len(tt.values))
				floatValue *tensor.Tensor
				intValue   *tensor.Tensor
				floats     []float32
				integers   []int64
				err        error
			)
			for index, value := range tt.values {
				floatInput[index] = float32(value)
			}
			floatValue, err = tensor.FromFloat32(tt.shape, floatInput)
			if (err != nil) != tt.wantErr {
				t.Fatalf("FromFloat32 error=%v, wantErr=%v", err, tt.wantErr)
			}
			if tt.wantErr && floatValue != nil {
				t.Fatal("invalid FromFloat32 returned a tensor")
			}
			intValue, err = tensor.FromInt64(tt.shape, tt.values)
			if (err != nil) != tt.wantErr {
				t.Fatalf("FromInt64 error=%v, wantErr=%v", err, tt.wantErr)
			}
			if tt.wantErr {
				if intValue != nil {
					t.Fatal("invalid FromInt64 returned a tensor")
				}
				return
			}
			if floats, err = floatValue.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if integers, err = intValue.Int64Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(floats, floatInput) || !slices.Equal(integers, tt.values) {
				t.Fatalf("values: float32=%v int64=%v", floats, integers)
			}
		})
	}
}

func Test_Tensor_InvalidReceivers(t *testing.T) {
	var (
		zero  tensor.Tensor
		tests = []*tensor.Tensor{nil, &zero}
	)
	for _, value := range tests {
		var (
			clone *tensor.Tensor
			err   error
		)
		if err = value.Validate(); err == nil {
			t.Fatal("invalid receiver passed Validate")
		}
		if value.DataType() != tensor.Invalid || value.Shape() != nil || value.Rank() != 0 || value.Len() != 0 || value.ByteSize() != 0 {
			t.Fatal("invalid receiver returned initialized metadata")
		}
		if _, err = value.Float32Values(); err == nil {
			t.Fatal("invalid receiver returned float32 values")
		}
		if _, err = value.Int64Values(); err == nil {
			t.Fatal("invalid receiver returned int64 values")
		}
		if clone, err = value.Clone(); err == nil || clone != nil {
			t.Fatalf("invalid receiver clone=%v err=%v", clone, err)
		}
	}
}
