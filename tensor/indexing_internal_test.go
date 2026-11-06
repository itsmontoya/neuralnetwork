package tensor

import (
	"math"
	"slices"
	"testing"
)

func Test_ContiguousStrides(t *testing.T) {
	type testcase struct {
		name    string
		shape   []int64
		want    []int64
		wantErr bool
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		tests = []testcase{
			{name: "scalar", want: []int64{}},
			{name: "vector", shape: []int64{4}, want: []int64{1}},
			{name: "rank three", shape: []int64{2, 3, 4}, want: []int64{12, 4, 1}},
			{name: "empty final dimension", shape: []int64{2, 3, 0}, want: []int64{0, 0, 1}},
			{name: "empty middle dimension", shape: []int64{2, 0, 3}, want: []int64{0, 3, 1}},
			{name: "large usable suffix", shape: []int64{limit, 0, limit}, want: []int64{0, limit, 1}},
			{name: "overflowing empty suffix", shape: []int64{0, limit, 2}, wantErr: true},
			{name: "overflowing product", shape: []int64{limit, 2}, wantErr: true},
			{name: "invalid empty dimension", shape: []int64{0, -1}, wantErr: true},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				before  = slices.Clone(tt.shape)
				strides []int64
				err     error
			)
			strides, err = contiguousStrides(tt.shape)
			if (err != nil) != tt.wantErr || !slices.Equal(strides, tt.want) {
				t.Fatalf("strides=%v error=%v, want %v wantErr=%v", strides, err, tt.want, tt.wantErr)
			}
			if !slices.Equal(tt.shape, before) {
				t.Fatal("stride planning changed shape")
			}
		})
	}
}

func Test_ShapeOffset(t *testing.T) {
	type testcase struct {
		name           string
		shape, indices []int64
		want           int
		wantErr        bool
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		tests = []testcase{
			{name: "scalar"},
			{name: "first", shape: []int64{2, 3, 4}, indices: []int64{0, 0, 0}},
			{name: "middle", shape: []int64{2, 3, 4}, indices: []int64{1, 1, 2}, want: 18},
			{name: "last", shape: []int64{2, 3, 4}, indices: []int64{1, 2, 3}, want: 23},
			{name: "large valid offset", shape: []int64{limit}, indices: []int64{limit - 1}, want: int(limit - 1)},
			{name: "negative index", shape: []int64{2}, indices: []int64{-1}, wantErr: true},
			{name: "past end", shape: []int64{2}, indices: []int64{2}, wantErr: true},
			{name: "missing index", shape: []int64{2}, wantErr: true},
			{name: "extra scalar index", indices: []int64{0}, wantErr: true},
			{name: "empty has no index", shape: []int64{2, 0}, indices: []int64{0, 0}, wantErr: true},
			{name: "negative dimension", shape: []int64{-1}, indices: []int64{0}, wantErr: true},
			{name: "size overflow", shape: []int64{limit, 2}, indices: []int64{0, 0}, wantErr: true},
			{name: "stride overflow", shape: []int64{0, limit, 2}, indices: []int64{0, 0, 0}, wantErr: true},
			{name: "minimum index", shape: []int64{2}, indices: []int64{math.MinInt64}, wantErr: true},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				offset int
				err    error
			)
			offset, err = shapeOffset(tt.shape, tt.indices)
			if (err != nil) != tt.wantErr || offset != tt.want {
				t.Fatalf("offset=%d error=%v, want %d wantErr=%v", offset, err, tt.want, tt.wantErr)
			}
		})
	}
}

func Test_CheckedArithmetic(t *testing.T) {
	type testcase struct {
		name        string
		left, right int64
		wantSum     int64
		wantProduct int64
		sumErr      bool
		productErr  bool
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		tests = []testcase{
			{name: "zeros"},
			{name: "small", left: 2, right: 3, wantSum: 5, wantProduct: 6},
			{name: "limit and zero", left: limit, wantSum: limit},
			{name: "sum overflow", left: limit, right: 1, sumErr: true, wantProduct: limit},
			{name: "product overflow", left: limit / 2, right: 3, wantSum: limit/2 + 3, productErr: true},
			{name: "both overflow", left: limit, right: 2, sumErr: true, productErr: true},
			{name: "negative left", left: -1, sumErr: true, productErr: true},
			{name: "negative right even with zero", right: -1, sumErr: true, productErr: true},
		}
	)
	if limit < math.MaxInt64 {
		tests = append(tests, testcase{name: "host operand overflow", left: limit + 1, sumErr: true, productErr: true})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				result int64
				err    error
			)
			result, err = checkedAdd(tt.left, tt.right)
			if (err != nil) != tt.sumErr || result != tt.wantSum {
				t.Fatalf("sum=%d error=%v, want %d wantErr=%v", result, err, tt.wantSum, tt.sumErr)
			}
			result, err = checkedMultiply(tt.left, tt.right)
			if (err != nil) != tt.productErr || result != tt.wantProduct {
				t.Fatalf("product=%d error=%v, want %d wantErr=%v", result, err, tt.wantProduct, tt.productErr)
			}
		})
	}
}

func Test_CheckedProduct(t *testing.T) {
	type testcase struct {
		name       string
		dimensions []int64
		want       int64
		wantErr    bool
	}
	var (
		limit = int64(int(^uint(0) >> 1))
		tests = []testcase{
			{name: "empty product", want: 1},
			{name: "normal", dimensions: []int64{2, 3, 4}, want: 24},
			{name: "host limit", dimensions: []int64{limit}, want: limit},
			{name: "zero skips unused product", dimensions: []int64{limit, limit, 0}},
			{name: "negative after zero", dimensions: []int64{0, -1}, wantErr: true},
			{name: "overflow", dimensions: []int64{limit, 2}, wantErr: true},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				product int64
				err     error
			)
			product, err = checkedProduct(tt.dimensions)
			if (err != nil) != tt.wantErr || product != tt.want {
				t.Fatalf("product=%d error=%v, want %d wantErr=%v", product, err, tt.want, tt.wantErr)
			}
		})
	}
}

func Test_ShapesEqual(t *testing.T) {
	type testcase struct {
		name        string
		left, right []int64
		want        bool
	}
	var tests = []testcase{
		{name: "scalars", left: nil, right: []int64{}, want: true},
		{name: "same empty dimensions", left: []int64{2, 0, 3}, right: []int64{2, 0, 3}, want: true},
		{name: "different rank", left: []int64{3}, right: []int64{1, 3}},
		{name: "different dimensions", left: []int64{2, 3}, right: []int64{2, 4}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if shapesEqual(tt.left, tt.right) != tt.want {
				t.Fatalf("equality=%v, want %v", !tt.want, tt.want)
			}
		})
	}
}
