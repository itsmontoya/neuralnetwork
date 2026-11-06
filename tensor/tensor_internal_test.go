package tensor

import "testing"

func Test_Tensor_StorageIdentity(t *testing.T) {
	var (
		shapes = [][]int64{nil, {0}, {2}}
		types  = []DataType{Float32, Int64}
	)
	for _, shape := range shapes {
		for _, dataType := range types {
			var (
				first, second, clone *Tensor
				shallow              Tensor
				err                  error
			)
			if first, err = New(dataType, shape); err != nil {
				t.Fatal(err)
			}
			if second, err = New(dataType, shape); err != nil {
				t.Fatal(err)
			}
			if clone, err = first.Clone(); err != nil {
				t.Fatal(err)
			}
			shallow = *first
			if first.identity == nil || first.identity != shallow.identity {
				t.Fatal("shallow copy lost its storage identity")
			}
			if first.identity == second.identity || first.identity == clone.identity || second.identity == clone.identity {
				t.Fatal("independent tensors share a storage identity")
			}
			if len(shape) == 0 || first.Len() == 0 {
				continue
			}
			clone.shape[0] = 99
			if first.shape[0] != shape[0] {
				t.Fatal("clone shares shape storage")
			}
			if dataType == Float32 {
				clone.floats[0] = 99
				if first.floats[0] != 0 {
					t.Fatal("clone shares float32 storage")
				}
				continue
			}
			clone.integers[0] = 99
			if first.integers[0] != 0 {
				t.Fatal("clone shares int64 storage")
			}
		}
	}
}

func Test_Tensor_ValidateStorage(t *testing.T) {
	type testcase struct {
		name  string
		value Tensor
	}
	var tests = []testcase{
		{name: "missing identity", value: Tensor{dataType: Float32, floats: []float32{1}}},
		{name: "invalid type", value: Tensor{identity: new(byte), dataType: Invalid}},
		{name: "float32 missing values", value: Tensor{identity: new(byte), dataType: Float32}},
		{name: "int64 missing values", value: Tensor{identity: new(byte), dataType: Int64}},
		{name: "float32 wrong storage", value: Tensor{identity: new(byte), dataType: Float32, floats: []float32{1}, integers: []int64{1}}},
		{name: "int64 wrong storage", value: Tensor{identity: new(byte), dataType: Int64, floats: []float32{1}, integers: []int64{1}}},
		{name: "negative shape", value: Tensor{identity: new(byte), dataType: Float32, shape: []int64{-1}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.value.Validate() == nil {
				t.Fatal("malformed tensor passed validation")
			}
			if tt.value.DataType() != Invalid || tt.value.Shape() != nil || tt.value.Rank() != 0 || tt.value.Len() != 0 || tt.value.ByteSize() != 0 {
				t.Fatal("malformed tensor returned initialized metadata")
			}
		})
	}
}
