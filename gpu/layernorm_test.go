package gpu

import (
	"context"
	"errors"
	"math"
	"testing"
)

func Test_Scope_LayerNorm(t *testing.T) {
	t.Run("valid_with_and_without_bias_and_dtypes", func(t *testing.T) {
		var (
			backend     = &fakeBackend{}
			dev         = newDevice(backend, ^uint64(0))
			ctx         = context.Background()
			weightTypes = []DType{Float32, Float16, BFloat16}
			biasTypes   = []DType{Float32, Float16, BFloat16}
		)
		for _, shape := range [][]int64{{8}, {2, 8}, {2, 3, 8}, {2, 2, 3, 8}} {
			var (
				input = mustNewTensor(t, dev, shape...)
				dst   = mustNewTensor(t, dev, shape...)
			)
			requireError(t, input.Upload(make([]float32, input.state.storage.count)), nil)
			for _, wt := range weightTypes {
				var weight = mustNewTensorWithDType(t, dev, wt, 8)
				requireError(t, weight.UploadFloat32As(make([]float32, 8)), nil)

				// bias == nil
				var scope, err = dev.NewScope()
				requireError(t, err, nil)
				requireError(t, scope.LayerNorm(input, weight, nil, dst, 1e-5), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)

				// bias != nil (including mixed dtypes)
				for _, bt := range biasTypes {
					var bias = mustNewTensorWithDType(t, dev, bt, 8)
					requireError(t, bias.UploadFloat32As(make([]float32, 8)), nil)
					scope, err = dev.NewScope()
					requireError(t, err, nil)
					requireError(t, scope.LayerNorm(input, weight, bias, dst, 1e-5), nil)
					requireError(t, scope.Submit(ctx), nil)
					requireError(t, scope.Wait(ctx), nil)
					requireError(t, scope.Close(), nil)
					requireError(t, bias.Close(), nil)
				}

				// weight and bias sharing the same read-only tensor is allowed.
				scope, err = dev.NewScope()
				requireError(t, err, nil)
				requireError(t, scope.LayerNorm(input, weight, weight, dst, 1e-5), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)

				requireError(t, weight.Close(), nil)
			}
			requireError(t, input.Close(), nil)
			requireError(t, dst.Close(), nil)
		}
		if backend.layerNorms.Load() == 0 {
			t.Fatalf("expected fakeBackend.layerNorms > 0")
		}
		requireError(t, dev.Close(), nil)
	})

	t.Run("validation_errors", func(t *testing.T) {
		var (
			dev        = newDevice(&fakeBackend{}, ^uint64(0))
			otherDev   = newDevice(&fakeBackend{}, ^uint64(0))
			input      = mustNewTensor(t, dev, 2, 32)
			dst        = mustNewTensor(t, dev, 2, 32)
			wrongDst   = mustNewTensor(t, dev, 2, 16)
			weight     = mustNewTensor(t, dev, 32)
			wrongW     = mustNewTensor(t, dev, 16)
			rank2W     = mustNewTensor(t, dev, 1, 32)
			bias       = mustNewTensor(t, dev, 32)
			wrongB     = mustNewTensor(t, dev, 16)
			rank2B     = mustNewTensor(t, dev, 1, 32)
			f16In      = mustNewTensorWithDType(t, dev, Float16, 2, 32)
			f16Dst     = mustNewTensorWithDType(t, dev, Float16, 2, 32)
			q8W        = mustNewTensorWithDType(t, dev, Q8_0, 1, 32)
			q4W        = mustNewTensorWithDType(t, dev, Q4_0, 1, 32)
			uninitIn   = mustNewTensor(t, dev, 2, 32)
			uninitW    = mustNewTensor(t, dev, 32)
			uninitB    = mustNewTensor(t, dev, 32)
			otherIn    = mustNewTensor(t, otherDev, 2, 32)
			closedIn   = mustNewTensor(t, dev, 2, 32)
			closedB    = mustNewTensor(t, dev, 32)
			inputView  *Tensor
			weightView *Tensor
			biasView   *Tensor
			err        error
		)
		defer func() {
			for _, tensor := range []*Tensor{
				input, dst, wrongDst, weight, wrongW, rank2W, bias, wrongB, rank2B,
				f16In, f16Dst, q8W, q4W, uninitIn, uninitW, uninitB, otherIn,
				inputView, weightView, biasView,
			} {
				if tensor != nil {
					_ = tensor.Close()
				}
			}
			_ = dev.Close()
			_ = otherDev.Close()
		}()
		requireError(t, input.Upload(make([]float32, 64)), nil)
		requireError(t, weight.Upload(make([]float32, 32)), nil)
		requireError(t, bias.Upload(make([]float32, 32)), nil)
		requireError(t, f16In.UploadFloat32As(make([]float32, 64)), nil)
		requireError(t, otherIn.Upload(make([]float32, 64)), nil)
		requireError(t, closedIn.Upload(make([]float32, 64)), nil)
		requireError(t, closedB.Upload(make([]float32, 32)), nil)
		requireError(t, closedIn.Close(), nil)
		requireError(t, closedB.Close(), nil)

		inputView, err = input.View([]int64{2, 32})
		requireError(t, err, nil)
		weightView, err = weight.View([]int64{1, 32})
		requireError(t, err, nil)
		biasView, err = bias.View([]int64{1, 32})
		requireError(t, err, nil)
		var (
			input1D  = mustNewTensor(t, dev, 32)
			dst1D    = mustNewTensor(t, dev, 32)
			wView1D  *Tensor
			bView1D  *Tensor
			largeIn  = mustNewTensor(t, dev, 1<<17, 1<<16)
			largeDst = mustNewTensor(t, dev, 1<<17, 1<<16)
		)
		defer func() {
			for _, tensor := range []*Tensor{input1D, dst1D, wView1D, bView1D, largeIn, largeDst} {
				if tensor != nil {
					_ = tensor.Close()
				}
			}
		}()
		requireError(t, input1D.Upload(make([]float32, 32)), nil)
		wView1D, err = weight.View([]int64{32})
		requireError(t, err, nil)
		bView1D, err = bias.View([]int64{32})
		requireError(t, err, nil)

		type testcase struct {
			name string
			in   *Tensor
			w    *Tensor
			b    *Tensor
			dst  *Tensor
			eps  float32
			want error
		}
		var tests = []testcase{
			{name: "nil_input", in: nil, w: weight, b: bias, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "nil_weight", in: input, w: nil, b: bias, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "nil_destination", in: input, w: weight, b: bias, dst: nil, eps: 1e-5, want: ErrInvalid},
			{name: "shape_mismatch_dst", in: input, w: weight, b: bias, dst: wrongDst, eps: 1e-5, want: ErrInvalid},
			{name: "weight_length_mismatch", in: input, w: wrongW, b: nil, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "weight_rank2", in: input, w: rank2W, b: nil, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "bias_length_mismatch", in: input, w: weight, b: wrongB, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "bias_rank2", in: input, w: weight, b: rank2B, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "non_f32_input", in: f16In, w: weight, b: nil, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "non_f32_destination", in: input, w: weight, b: nil, dst: f16Dst, eps: 1e-5, want: ErrInvalid},
			{name: "quantized_q8_weight", in: input, w: q8W, b: nil, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "quantized_q4_weight", in: input, w: q4W, b: nil, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "zero_epsilon", in: input, w: weight, b: nil, dst: dst, eps: 0, want: ErrInvalid},
			{name: "negative_epsilon", in: input, w: weight, b: nil, dst: dst, eps: -1e-5, want: ErrInvalid},
			{name: "nan_epsilon", in: input, w: weight, b: nil, dst: dst, eps: float32(math.NaN()), want: ErrInvalid},
			{name: "inf_epsilon", in: input, w: weight, b: nil, dst: dst, eps: float32(math.Inf(1)), want: ErrInvalid},
			{name: "neg_inf_epsilon", in: input, w: weight, b: nil, dst: dst, eps: float32(math.Inf(-1)), want: ErrInvalid},
			{name: "exceeds_uint32_count", in: largeIn, w: weight, b: nil, dst: largeDst, eps: 1e-5, want: ErrInvalid},
			{name: "alias_input_direct", in: input, w: weight, b: nil, dst: input, eps: 1e-5, want: ErrInvalid},
			{name: "alias_input_view", in: input, w: weight, b: nil, dst: inputView, eps: 1e-5, want: ErrInvalid},
			{name: "alias_weight_view", in: input1D, w: weight, b: nil, dst: wView1D, eps: 1e-5, want: ErrInvalid},
			{name: "alias_bias_view", in: input1D, w: weight, b: bias, dst: bView1D, eps: 1e-5, want: ErrInvalid},
			{name: "uninitialized_input", in: uninitIn, w: weight, b: bias, dst: dst, eps: 1e-5, want: ErrUninitialized},
			{name: "uninitialized_weight", in: input, w: uninitW, b: bias, dst: dst, eps: 1e-5, want: ErrUninitialized},
			{name: "uninitialized_bias", in: input, w: weight, b: uninitB, dst: dst, eps: 1e-5, want: ErrUninitialized},
			{name: "cross_device_input", in: otherIn, w: weight, b: nil, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "closed_input", in: closedIn, w: weight, b: nil, dst: dst, eps: 1e-5, want: ErrClosed},
			{name: "closed_bias", in: input, w: weight, b: closedB, dst: dst, eps: 1e-5, want: ErrClosed},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var scope, scopeErr = dev.NewScope()
				requireError(t, scopeErr, nil)
				defer scope.Close()
				requireError(t, scope.LayerNorm(tt.in, tt.w, tt.b, tt.dst, tt.eps), tt.want)
			})
		}
	})

	t.Run("encode_error_poisons_scope", func(t *testing.T) {
		var (
			dev        = newDevice(&fakeBackend{}, ^uint64(0))
			input      = mustNewTensor(t, dev, 2, 8)
			weight     = mustNewTensor(t, dev, 8)
			dst        = mustNewTensor(t, dev, 2, 8)
			scope, err = dev.NewScope()
			boom       = errors.New("encode failure")
		)
		requireError(t, err, nil)
		requireError(t, input.Upload(make([]float32, 16)), nil)
		requireError(t, weight.Upload(make([]float32, 8)), nil)
		scope.state.commands.(*fakeCommandBuffer).encodeErr = boom
		requireError(t, scope.LayerNorm(input, weight, nil, dst, 1e-5), boom)
		requireError(t, scope.Submit(context.Background()), ErrInvalid)
		requireError(t, scope.Close(), nil)
		requireError(t, input.Close(), nil)
		requireError(t, weight.Close(), nil)
		requireError(t, dst.Close(), nil)
		requireError(t, dev.Close(), nil)
	})
}

func mustNewTensor(t *testing.T, dev *Device, shape ...int64) (out *Tensor) {
	t.Helper()
	var err error
	out, err = dev.NewFloat32(shape)
	requireError(t, err, nil)
	return out
}

func mustNewTensorWithDType(t *testing.T, dev *Device, dtype DType, shape ...int64) (out *Tensor) {
	t.Helper()
	var err error
	out, err = dev.NewTensorWithDType(shape, dtype)
	requireError(t, err, nil)
	return out
}
