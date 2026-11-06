package tensor_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_Products_Reference(t *testing.T) {
	type fixture struct {
		Name        string   `json:"name"`
		Operation   string   `json:"operation"`
		LeftShape   []int64  `json:"left_shape"`
		LeftBits    []uint32 `json:"left_bits"`
		RightShape  []int64  `json:"right_shape"`
		RightBits   []uint32 `json:"right_bits"`
		HasBias     bool     `json:"has_bias"`
		BiasShape   []int64  `json:"bias_shape"`
		BiasBits    []uint32 `json:"bias_bits"`
		TransposeA  bool     `json:"transpose_a"`
		TransposeB  bool     `json:"transpose_b"`
		AlphaBits   uint32   `json:"alpha_bits"`
		BetaBits    uint32   `json:"beta_bits"`
		OutputShape []int64  `json:"output_shape"`
		OutputBits  []uint32 `json:"output_bits"`
	}
	var (
		contents []byte
		fixtures []fixture
		err      error
	)

	if contents, err = os.ReadFile("testdata/products.json"); err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(contents, &fixtures); err != nil {
		t.Fatal(err)
	}
	if len(fixtures) != 49 {
		t.Fatalf("fixture count=%d, want 49", len(fixtures))
	}
	for _, tt := range fixtures {
		t.Run(tt.Name, func(t *testing.T) {
			var (
				left, right, bias, output *tensor.Tensor
				shape                     []int64
				values                    []float32
				config                    tensor.GemmConfig
			)

			config.TransposeA = tt.TransposeA
			config.TransposeB = tt.TransposeB
			config.Alpha = math.Float32frombits(tt.AlphaBits)
			config.Beta = math.Float32frombits(tt.BetaBits)
			if left, err = tensor.FromFloat32(tt.LeftShape, valuesFromBits(tt.LeftBits)); err != nil {
				t.Fatal(err)
			}
			if right, err = tensor.FromFloat32(tt.RightShape, valuesFromBits(tt.RightBits)); err != nil {
				t.Fatal(err)
			}
			if tt.HasBias {
				if bias, err = tensor.FromFloat32(tt.BiasShape, valuesFromBits(tt.BiasBits)); err != nil {
					t.Fatal(err)
				}
			}
			switch tt.Operation {
			case "MatMul":
				shape, err = tensor.MatMulShape(left.Shape(), right.Shape())
			case "Gemm":
				shape, err = tensor.GemmShape(left.Shape(), right.Shape(), config)
			default:
				t.Fatalf("unknown operation %q", tt.Operation)
			}
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(shape, tt.OutputShape) {
				t.Fatalf("shape=%v, want %v", shape, tt.OutputShape)
			}
			if output, err = tensor.New(tensor.Float32, shape); err != nil {
				t.Fatal(err)
			}
			if tt.Operation == "MatMul" {
				err = tensor.MatMulInto(left, right, output)
			} else {
				err = tensor.GemmInto(left, right, bias, output, config)
			}
			if err != nil {
				t.Fatal(err)
			}
			if values, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			checkProductValues(t, values, valuesFromBits(tt.OutputBits))
			for index, input := range []*tensor.Tensor{left, right, bias} {
				if input == nil {
					continue
				}
				if values, err = input.Float32Values(); err != nil {
					t.Fatal(err)
				}
				checkFloat32Bits(t, values, [][]uint32{tt.LeftBits, tt.RightBits, tt.BiasBits}[index])
			}
		})
	}
}

func Test_ProductShapes(t *testing.T) {
	type testcase struct {
		name    string
		left    []int64
		right   []int64
		want    []int64
		gemm    bool
		config  tensor.GemmConfig
		wantErr bool
	}
	var (
		maxHostIndex = int64(int(^uint(0) >> 1))
		tests        = []testcase{
			{name: "dot scalar", left: []int64{3}, right: []int64{3}},
			{name: "left scalar", right: []int64{1}, wantErr: true},
			{name: "right scalar", left: []int64{1}, wantErr: true},
			{name: "unequal inner", left: []int64{2, 3}, right: []int64{4, 2}, wantErr: true},
			{name: "unequal batch", left: []int64{2, 2, 3}, right: []int64{3, 3, 2}, wantErr: true},
			{name: "zero batch with one", left: []int64{0, 2, 3}, right: []int64{1, 3, 2}, want: []int64{0, 2, 2}},
			{name: "zero batch with two", left: []int64{0, 2, 3}, right: []int64{2, 3, 2}, wantErr: true},
			{name: "negative", left: []int64{0, -1}, right: []int64{1, 1}, wantErr: true},
			{name: "input byte overflow", left: []int64{maxHostIndex/4 + 1, 1}, right: []int64{1, 1}, wantErr: true},
			{name: "output overflow", left: []int64{maxHostIndex, 0}, right: []int64{0, 2}, wantErr: true},
			{name: "unused huge batch", left: []int64{maxHostIndex, maxHostIndex, 0, 0}, right: []int64{1, 1, 0, 0}, want: []int64{maxHostIndex, maxHostIndex, 0, 0}},
			{name: "Gemm rectangular", gemm: true, left: []int64{2, 3}, right: []int64{3, 4}, want: []int64{2, 4}},
			{name: "Gemm both transposed", gemm: true, left: []int64{3, 2}, right: []int64{4, 3}, config: tensor.GemmConfig{TransposeA: true, TransposeB: true}, want: []int64{2, 4}},
			{name: "Gemm left vector", gemm: true, left: []int64{3}, right: []int64{3, 4}, wantErr: true},
			{name: "Gemm right vector", gemm: true, left: []int64{2, 3}, right: []int64{3}, wantErr: true},
			{name: "Gemm scalar", gemm: true, wantErr: true},
			{name: "Gemm inner mismatch", gemm: true, left: []int64{2, 3}, right: []int64{4, 2}, wantErr: true},
			{name: "Gemm negative transposed", gemm: true, left: []int64{-1, 0}, right: []int64{1, 1}, config: tensor.GemmConfig{TransposeA: true}, wantErr: true},
		}
	)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				shape       []int64
				leftBefore  = slices.Clone(tt.left)
				rightBefore = slices.Clone(tt.right)
				err         error
			)
			if tt.gemm {
				shape, err = tensor.GemmShape(tt.left, tt.right, tt.config)
			} else {
				shape, err = tensor.MatMulShape(tt.left, tt.right)
			}
			if (err != nil) != tt.wantErr {
				t.Fatalf("shape error=%v, wantErr=%v", err, tt.wantErr)
			}
			if !slices.Equal(tt.left, leftBefore) || !slices.Equal(tt.right, rightBefore) {
				t.Fatal("shape planner changed input metadata")
			}
			if tt.wantErr {
				if shape != nil {
					t.Fatal("invalid shape returned a result")
				}
				return
			}
			if !slices.Equal(shape, tt.want) {
				t.Fatalf("shape=%v, want %v", shape, tt.want)
			}
			if len(shape) > 0 {
				shape[0] = 99
				if !slices.Equal(tt.left, leftBefore) || !slices.Equal(tt.right, rightBefore) {
					t.Fatal("result shape aliases input metadata")
				}
			}
		})
	}
}

func Test_Products_ArgumentErrors(t *testing.T) {
	var (
		tests = []string{
			"left nil", "right nil", "destination nil", "left zero", "right zero", "destination zero",
			"left integer", "right integer", "destination integer", "left scalar", "right scalar",
			"wrong inner", "wrong destination shape", "wrong destination rank",
		}
		operations = []struct {
			name string
			run  func(*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) error
		}{
			{name: "MatMul", run: tensor.MatMulInto},
			{name: "Gemm", run: func(left, right, output *tensor.Tensor) (err error) {
				return tensor.GemmInto(left, right, nil, output, tensor.GemmConfig{Alpha: 1})
			}},
		}
	)
	for _, operation := range operations {
		for _, name := range tests {
			t.Run(operation.name+"/"+name, func(t *testing.T) {
				var (
					left, right, output *tensor.Tensor
					zero                tensor.Tensor
					before              = []float32{99, 99, 99, 99}
					values              []float32
					integers            []int64
					err                 error
				)
				if left, err = tensor.FromFloat32([]int64{2, 2}, []float32{1, 2, 3, 4}); err != nil {
					t.Fatal(err)
				}
				if right, err = left.Clone(); err != nil {
					t.Fatal(err)
				}
				if output, err = tensor.FromFloat32([]int64{2, 2}, before); err != nil {
					t.Fatal(err)
				}
				switch name {
				case "left nil":
					left = nil
				case "right nil":
					right = nil
				case "destination nil":
					output = nil
				case "left zero":
					left = &zero
				case "right zero":
					right = &zero
				case "destination zero":
					output = &zero
				case "left integer":
					left, err = tensor.New(tensor.Int64, []int64{2, 2})
				case "right integer":
					right, err = tensor.New(tensor.Int64, []int64{2, 2})
				case "destination integer":
					output, err = tensor.FromInt64([]int64{2, 2}, []int64{99, 99, 99, 99})
				case "left scalar":
					left, err = tensor.FromFloat32(nil, []float32{1})
				case "right scalar":
					right, err = tensor.FromFloat32(nil, []float32{1})
				case "wrong inner":
					right, err = tensor.New(tensor.Float32, []int64{3, 2})
				case "wrong destination shape":
					output, err = tensor.FromFloat32([]int64{1, 4}, before)
				case "wrong destination rank":
					output, err = tensor.FromFloat32([]int64{4}, before)
				}
				if err != nil {
					t.Fatal(err)
				}
				if err = operation.run(left, right, output); err == nil {
					t.Fatal("invalid product arguments succeeded")
				}
				if output.DataType() == tensor.Float32 {
					if values, err = output.Float32Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(values, before) {
						t.Fatalf("destination changed on error: %v", values)
					}
				} else if output.DataType() == tensor.Int64 {
					if integers, err = output.Int64Values(); err != nil {
						t.Fatal(err)
					}
					if !slices.Equal(integers, []int64{99, 99, 99, 99}) {
						t.Fatalf("integer destination changed on error: %v", integers)
					}
				}
			})
		}
	}
}

func Test_Products_AliasesAndEmptyContractions(t *testing.T) {
	var (
		shapes = [][]int64{{2, 2}, {0, 0}}
		err    error
	)
	for _, shape := range shapes {
		var (
			left, right, output *tensor.Tensor
			shallow             tensor.Tensor
		)
		if left, err = tensor.New(tensor.Float32, shape); err != nil {
			t.Fatal(err)
		}
		if right, err = left.Clone(); err != nil {
			t.Fatal(err)
		}
		if output, err = left.Clone(); err != nil {
			t.Fatal(err)
		}
		for _, input := range []*tensor.Tensor{left, right} {
			shallow = *input
			for _, alias := range []*tensor.Tensor{input, &shallow} {
				if err = tensor.MatMulInto(left, right, alias); err == nil {
					t.Fatal("MatMul destination alias accepted")
				}
				if err = tensor.GemmInto(left, right, nil, alias, tensor.GemmConfig{}); err == nil {
					t.Fatal("Gemm destination alias accepted with zero coefficients")
				}
			}
		}
		if err = tensor.MatMulInto(left, left, output); err != nil {
			t.Fatal(err)
		}
		if err = tensor.GemmInto(left, left, nil, output, tensor.GemmConfig{Alpha: 1}); err != nil {
			t.Fatal(err)
		}
		shallow = *output
		for _, bias := range []*tensor.Tensor{output, &shallow} {
			if err = tensor.GemmInto(left, right, bias, output, tensor.GemmConfig{}); err == nil {
				t.Fatal("Gemm bias destination alias accepted with zero coefficients")
			}
		}
	}
	for _, shape := range [][]int64{nil, {2, 3}} {
		var (
			left, right, output *tensor.Tensor
			values              []float32
		)
		if len(shape) == 0 {
			if left, err = tensor.New(tensor.Float32, []int64{0}); err != nil {
				t.Fatal(err)
			}
			right, err = tensor.New(tensor.Float32, []int64{0})
		} else {
			if left, err = tensor.New(tensor.Float32, []int64{2, 0}); err != nil {
				t.Fatal(err)
			}
			right, err = tensor.New(tensor.Float32, []int64{0, 3})
		}
		if err != nil {
			t.Fatal(err)
		}
		if output, err = tensor.New(tensor.Float32, shape); err != nil {
			t.Fatal(err)
		}
		if err = tensor.MatMulInto(left, right, output); err != nil {
			t.Fatal(err)
		}
		if values, err = output.Float32Values(); err != nil {
			t.Fatal(err)
		}
		for _, value := range values {
			if math.Float32bits(value) != 0 {
				t.Fatalf("empty contraction produced %g, want positive zero", value)
			}
		}
	}
}

func Test_Gemm_BiasValidation(t *testing.T) {
	var tests = []string{"uninitialized", "integer", "incompatible", "extra rank", "enlarges output", "aliases output", "shallow output alias"}
	for _, name := range tests {
		t.Run(name, func(t *testing.T) {
			var (
				left, right, bias, output *tensor.Tensor
				zero, shallow             tensor.Tensor
				before                    = []float32{99, 99, 99, 99}
				values                    []float32
				err                       error
			)
			if left, err = tensor.New(tensor.Float32, []int64{2, 2}); err != nil {
				t.Fatal(err)
			}
			if right, err = left.Clone(); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.FromFloat32([]int64{2, 2}, before); err != nil {
				t.Fatal(err)
			}
			switch name {
			case "uninitialized":
				bias = &zero
			case "integer":
				bias, err = tensor.New(tensor.Int64, []int64{2})
			case "incompatible":
				bias, err = tensor.New(tensor.Float32, []int64{3})
			case "extra rank":
				bias, err = tensor.New(tensor.Float32, []int64{1, 2, 2})
			case "enlarges output":
				if left, err = tensor.New(tensor.Float32, []int64{1, 2}); err != nil {
					t.Fatal(err)
				}
				if output, err = tensor.FromFloat32([]int64{1, 2}, before[:2]); err != nil {
					t.Fatal(err)
				}
				before = before[:2]
				bias, err = tensor.New(tensor.Float32, []int64{2, 2})
			case "aliases output":
				bias = output
			case "shallow output alias":
				shallow = *output
				bias = &shallow
			}
			if err != nil {
				t.Fatal(err)
			}
			// Zero Beta cannot bypass validation of a supplied bias.
			if err = tensor.GemmInto(left, right, bias, output, tensor.GemmConfig{Alpha: 1}); err == nil {
				t.Fatal("invalid bias accepted with zero Beta")
			}
			if values, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(values, before) {
				t.Fatalf("destination changed on bias error: %v", values)
			}
		})
	}
}

func Test_MatMul_EmptyUnusedBatchSizes(t *testing.T) {
	var (
		maxHostIndex        = int64(int(^uint(0) >> 1))
		shape               = []int64{maxHostIndex, maxHostIndex, 0, 0}
		left, right, output *tensor.Tensor
		err                 error
	)
	if left, err = tensor.New(tensor.Float32, shape); err != nil {
		t.Fatal(err)
	}
	if right, err = tensor.New(tensor.Float32, []int64{1, 1, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if output, err = tensor.New(tensor.Float32, shape); err != nil {
		t.Fatal(err)
	}
	if err = tensor.MatMulInto(left, right, output); err != nil {
		t.Fatal(err)
	}
}

func Test_Products_ExistingMatrixKernels(t *testing.T) {
	var tests = []struct {
		name   string
		config tensor.GemmConfig
		run    func(*matrix.Matrix, *matrix.Matrix, *matrix.Matrix) error
	}{
		{name: "normal", config: tensor.GemmConfig{Alpha: 1}, run: (*matrix.Matrix).MatMulInto},
		{name: "left transpose", config: tensor.GemmConfig{Alpha: 1, TransposeA: true}, run: (*matrix.Matrix).MatMulLeftTransposeInto},
		{name: "right transpose", config: tensor.GemmConfig{Alpha: 1, TransposeB: true}, run: (*matrix.Matrix).MatMulRightTransposeInto},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				left, right, output     *tensor.Tensor
				leftMatrix, rightMatrix *matrix.Matrix
				outputMatrix            *matrix.Matrix
				leftValues              = []float32{1, 2, 3, 4}
				rightValues             = []float32{5, 6, 7, 8}
				got, want               []float32
				err                     error
			)
			if left, err = tensor.FromFloat32([]int64{2, 2}, leftValues); err != nil {
				t.Fatal(err)
			}
			if right, err = tensor.FromFloat32([]int64{2, 2}, rightValues); err != nil {
				t.Fatal(err)
			}
			if output, err = tensor.New(tensor.Float32, []int64{2, 2}); err != nil {
				t.Fatal(err)
			}
			if leftMatrix, err = matrix.FromSlice(2, 2, leftValues); err != nil {
				t.Fatal(err)
			}
			if rightMatrix, err = matrix.FromSlice(2, 2, rightValues); err != nil {
				t.Fatal(err)
			}
			if outputMatrix, err = matrix.New(2, 2); err != nil {
				t.Fatal(err)
			}
			if err = tt.run(leftMatrix, rightMatrix, outputMatrix); err != nil {
				t.Fatal(err)
			}
			if err = tensor.GemmInto(left, right, nil, output, tt.config); err != nil {
				t.Fatal(err)
			}
			if got, err = output.Float32Values(); err != nil {
				t.Fatal(err)
			}
			if want, err = outputMatrix.Values(); err != nil {
				t.Fatal(err)
			}
			checkProductValues(t, got, want)
		})
	}
}

func Test_Products_DynamicWeightsAndConcurrency(t *testing.T) {
	var (
		left, weights, changed, first, second *tensor.Tensor
		before                                []float32
		values                                []float32
		workers                               sync.WaitGroup
		results                               = make(chan error, 4)
		err                                   error
	)
	if left, err = tensor.FromFloat32([]int64{2, 3}, []float32{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatal(err)
	}
	if weights, err = tensor.FromFloat32([]int64{3, 2}, []float32{7, 8, 9, 10, 11, 12}); err != nil {
		t.Fatal(err)
	}
	if changed, err = tensor.FromFloat32([]int64{3, 2}, []float32{1, 1, 1, 1, 1, 1}); err != nil {
		t.Fatal(err)
	}
	if first, err = tensor.New(tensor.Float32, []int64{2, 2}); err != nil {
		t.Fatal(err)
	}
	if second, err = first.Clone(); err != nil {
		t.Fatal(err)
	}
	if err = tensor.MatMulInto(left, weights, first); err != nil {
		t.Fatal(err)
	}
	if before, err = first.Float32Values(); err != nil {
		t.Fatal(err)
	}
	checkProductValues(t, before, []float32{58, 64, 139, 154})
	if err = tensor.GemmInto(left, changed, nil, second, tensor.GemmConfig{Alpha: 1}); err != nil {
		t.Fatal(err)
	}
	if values, err = second.Float32Values(); err != nil {
		t.Fatal(err)
	}
	checkProductValues(t, values, []float32{6, 6, 15, 15})
	if values, err = first.Float32Values(); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(before, values) {
		t.Fatal("later call changed earlier output")
	}
	for range 4 {
		workers.Add(1)
		go func() {
			var (
				output    *tensor.Tensor
				got       []float32
				workerErr error
			)
			defer workers.Done()
			if output, workerErr = tensor.New(tensor.Float32, []int64{2, 2}); workerErr == nil {
				workerErr = tensor.MatMulInto(left, weights, output)
			}
			if workerErr == nil {
				workerErr = tensor.GemmInto(left, weights, nil, output, tensor.GemmConfig{Alpha: 1})
			}
			if workerErr == nil {
				got, workerErr = output.Float32Values()
			}
			if workerErr == nil && !slices.Equal(got, []float32{58, 64, 139, 154}) {
				workerErr = fmt.Errorf("concurrent product=%v", got)
			}
			results <- workerErr
		}()
	}
	workers.Wait()
	close(results)
	for result := range results {
		if result != nil {
			t.Fatal(result)
		}
	}
}

func checkProductValues(t *testing.T, got, want []float32) {
	var (
		index    int
		actual   float64
		expected float64
	)

	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("value count=%d, want %d", len(got), len(want))
	}
	for index = range got {
		actual, expected = float64(got[index]), float64(want[index])
		if math.IsNaN(expected) {
			if !math.IsNaN(actual) {
				t.Fatalf("value %d=%g, want NaN", index, actual)
			}
			continue
		}
		if math.IsInf(expected, 0) {
			if actual != expected {
				t.Fatalf("value %d=%g, want %g", index, actual, expected)
			}
			continue
		}
		if math.IsNaN(actual) || math.IsInf(actual, 0) || math.Abs(actual-expected) > 1e-5+1e-5*math.Abs(expected) {
			t.Fatalf("value %d=%g, want %g within atol=rtol=1e-5", index, actual, expected)
		}
	}
}
