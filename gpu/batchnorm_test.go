package gpu

import (
	"context"
	"math"
	"testing"
)

func Test_Scope_BatchNorm(t *testing.T) {
	t.Run("valid_ranks_and_shared_params", func(t *testing.T) {
		var (
			backend = &fakeBackend{}
			dev     = newDevice(backend, ^uint64(0))
			ctx     = context.Background()
			shapes  = [][]int64{
				{2, 8},
				{2, 8, 5},
				{2, 8, 4, 4},
				{2, 8, 2, 3, 2},
			}
		)
		for _, shape := range shapes {
			var (
				input    = mustNewTensor(t, dev, shape...)
				dst      = mustNewTensor(t, dev, shape...)
				scale    = mustNewTensor(t, dev, 8)
				bias     = mustNewTensor(t, dev, 8)
				mean     = mustNewTensor(t, dev, 8)
				variance = mustNewTensor(t, dev, 8)
			)
			requireError(t, input.Upload(make([]float32, input.state.storage.count)), nil)
			requireError(t, scale.Upload(make([]float32, 8)), nil)
			requireError(t, bias.Upload(make([]float32, 8)), nil)
			requireError(t, mean.Upload(make([]float32, 8)), nil)
			requireError(t, variance.Upload(make([]float32, 8)), nil)

			for _, eps := range []float32{0, 1e-5} {
				var scope, err = dev.NewScope()
				requireError(t, err, nil)
				requireError(t, scope.BatchNorm(input, scale, bias, mean, variance, dst, eps), nil)
				requireError(t, scope.Submit(ctx), nil)
				requireError(t, scope.Wait(ctx), nil)
				requireError(t, scope.Close(), nil)
			}

			// Shared read-only parameter tensors are allowed.
			var scope, err = dev.NewScope()
			requireError(t, err, nil)
			requireError(t, scope.BatchNorm(input, scale, scale, mean, mean, dst, 1e-5), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)

			for _, tensor := range []*Tensor{input, dst, scale, bias, mean, variance} {
				requireError(t, tensor.Close(), nil)
			}
		}
		if got := backend.batchNorms.Load(); got != int32(len(shapes)*3) {
			t.Fatalf("expected %d batchNorm dispatches, got %d", len(shapes)*3, got)
		}
		requireError(t, dev.Close(), nil)
	})

	t.Run("validation_errors", func(t *testing.T) {
		var (
			dev         = newDevice(&fakeBackend{}, ^uint64(0))
			otherDev    = newDevice(&fakeBackend{}, ^uint64(0))
			input       = mustNewTensor(t, dev, 2, 4, 3, 3)
			dst         = mustNewTensor(t, dev, 2, 4, 3, 3)
			wrongDst    = mustNewTensor(t, dev, 2, 4, 3, 2)
			rank1In     = mustNewTensor(t, dev, 4)
			rank1Dst    = mustNewTensor(t, dev, 4)
			scale       = mustNewTensor(t, dev, 4)
			bias        = mustNewTensor(t, dev, 4)
			mean        = mustNewTensor(t, dev, 4)
			variance    = mustNewTensor(t, dev, 4)
			wrongParam  = mustNewTensor(t, dev, 3)
			rank2Param  = mustNewTensor(t, dev, 1, 4)
			f16In       = mustNewTensorWithDType(t, dev, Float16, 2, 4, 3, 3)
			f16Dst      = mustNewTensorWithDType(t, dev, Float16, 2, 4, 3, 3)
			f16Param    = mustNewTensorWithDType(t, dev, Float16, 4)
			uninitIn    = mustNewTensor(t, dev, 2, 4, 3, 3)
			uninitParam = mustNewTensor(t, dev, 4)
			otherIn     = mustNewTensor(t, otherDev, 2, 4, 3, 3)
			closedIn    = mustNewTensor(t, dev, 2, 4, 3, 3)
			closedParam = mustNewTensor(t, dev, 4)
			largeIn     = mustNewTensor(t, dev, 1<<17, 1<<16)
			largeDst    = mustNewTensor(t, dev, 1<<17, 1<<16)
			largeParam  = mustNewTensor(t, dev, 1<<16)
			inputView   *Tensor
			scaleView2D *Tensor
			biasView2D  *Tensor
			meanView2D  *Tensor
			varView2D   *Tensor
			in2D        = mustNewTensor(t, dev, 1, 4)
			err         error
		)
		defer func() {
			for _, tensor := range []*Tensor{
				input, dst, wrongDst, rank1In, rank1Dst, scale, bias, mean, variance,
				wrongParam, rank2Param, f16In, f16Dst, f16Param, uninitIn, uninitParam,
				otherIn, largeIn, largeDst, largeParam, inputView, scaleView2D,
				biasView2D, meanView2D, varView2D, in2D,
			} {
				if tensor != nil {
					_ = tensor.Close()
				}
			}
			_ = dev.Close()
			_ = otherDev.Close()
		}()

		requireError(t, input.Upload(make([]float32, 72)), nil)
		requireError(t, rank1In.Upload(make([]float32, 4)), nil)
		requireError(t, scale.Upload(make([]float32, 4)), nil)
		requireError(t, bias.Upload(make([]float32, 4)), nil)
		requireError(t, mean.Upload(make([]float32, 4)), nil)
		requireError(t, variance.Upload(make([]float32, 4)), nil)
		requireError(t, wrongParam.Upload(make([]float32, 3)), nil)
		requireError(t, rank2Param.Upload(make([]float32, 4)), nil)
		requireError(t, f16In.UploadFloat32As(make([]float32, 72)), nil)
		requireError(t, f16Param.UploadFloat32As(make([]float32, 4)), nil)
		requireError(t, otherIn.Upload(make([]float32, 72)), nil)
		requireError(t, closedIn.Upload(make([]float32, 72)), nil)
		requireError(t, closedParam.Upload(make([]float32, 4)), nil)
		requireError(t, closedIn.Close(), nil)
		requireError(t, closedParam.Close(), nil)
		requireError(t, in2D.Upload(make([]float32, 4)), nil)

		inputView, err = input.View([]int64{2, 4, 3, 3})
		requireError(t, err, nil)
		scaleView2D, err = scale.View([]int64{1, 4})
		requireError(t, err, nil)
		biasView2D, err = bias.View([]int64{1, 4})
		requireError(t, err, nil)
		meanView2D, err = mean.View([]int64{1, 4})
		requireError(t, err, nil)
		varView2D, err = variance.View([]int64{1, 4})
		requireError(t, err, nil)

		type testcase struct {
			name string
			in   *Tensor
			sc   *Tensor
			bi   *Tensor
			mn   *Tensor
			vr   *Tensor
			dst  *Tensor
			eps  float32
			want error
		}
		var tests = []testcase{
			{name: "nil_input", in: nil, sc: scale, bi: bias, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "nil_scale", in: input, sc: nil, bi: bias, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "nil_bias", in: input, sc: scale, bi: nil, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "nil_mean", in: input, sc: scale, bi: bias, mn: nil, vr: variance, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "nil_variance", in: input, sc: scale, bi: bias, mn: mean, vr: nil, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "nil_destination", in: input, sc: scale, bi: bias, mn: mean, vr: variance, dst: nil, eps: 1e-5, want: ErrInvalid},
			{name: "rank1_input", in: rank1In, sc: scale, bi: bias, mn: mean, vr: variance, dst: rank1Dst, eps: 1e-5, want: ErrInvalid},
			{name: "dst_shape_mismatch", in: input, sc: scale, bi: bias, mn: mean, vr: variance, dst: wrongDst, eps: 1e-5, want: ErrInvalid},
			{name: "scale_shape_mismatch", in: input, sc: wrongParam, bi: bias, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "bias_rank2", in: input, sc: scale, bi: rank2Param, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "mean_shape_mismatch", in: input, sc: scale, bi: bias, mn: wrongParam, vr: variance, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "variance_shape_mismatch", in: input, sc: scale, bi: bias, mn: mean, vr: wrongParam, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "negative_epsilon", in: input, sc: scale, bi: bias, mn: mean, vr: variance, dst: dst, eps: -1e-5, want: ErrInvalid},
			{name: "nan_epsilon", in: input, sc: scale, bi: bias, mn: mean, vr: variance, dst: dst, eps: float32(math.NaN()), want: ErrInvalid},
			{name: "pos_inf_epsilon", in: input, sc: scale, bi: bias, mn: mean, vr: variance, dst: dst, eps: float32(math.Inf(1)), want: ErrInvalid},
			{name: "neg_inf_epsilon", in: input, sc: scale, bi: bias, mn: mean, vr: variance, dst: dst, eps: float32(math.Inf(-1)), want: ErrInvalid},
			{name: "f16_input", in: f16In, sc: scale, bi: bias, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "f16_dst", in: input, sc: scale, bi: bias, mn: mean, vr: variance, dst: f16Dst, eps: 1e-5, want: ErrInvalid},
			{name: "f16_scale", in: input, sc: f16Param, bi: bias, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "f16_bias", in: input, sc: scale, bi: f16Param, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "f16_mean", in: input, sc: scale, bi: bias, mn: f16Param, vr: variance, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "f16_variance", in: input, sc: scale, bi: bias, mn: mean, vr: f16Param, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "direct_alias_input", in: input, sc: scale, bi: bias, mn: mean, vr: variance, dst: input, eps: 1e-5, want: ErrInvalid},
			{name: "view_alias_input", in: input, sc: scale, bi: bias, mn: mean, vr: variance, dst: inputView, eps: 1e-5, want: ErrInvalid},
			{name: "view_alias_scale", in: in2D, sc: scale, bi: bias, mn: mean, vr: variance, dst: scaleView2D, eps: 1e-5, want: ErrInvalid},
			{name: "view_alias_bias", in: in2D, sc: scale, bi: bias, mn: mean, vr: variance, dst: biasView2D, eps: 1e-5, want: ErrInvalid},
			{name: "view_alias_mean", in: in2D, sc: scale, bi: bias, mn: mean, vr: variance, dst: meanView2D, eps: 1e-5, want: ErrInvalid},
			{name: "view_alias_variance", in: in2D, sc: scale, bi: bias, mn: mean, vr: variance, dst: varView2D, eps: 1e-5, want: ErrInvalid},
			{name: "uninit_input", in: uninitIn, sc: scale, bi: bias, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrUninitialized},
			{name: "uninit_scale", in: input, sc: uninitParam, bi: bias, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrUninitialized},
			{name: "other_device", in: otherIn, sc: scale, bi: bias, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrInvalid},
			{name: "closed_input", in: closedIn, sc: scale, bi: bias, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrClosed},
			{name: "closed_param", in: input, sc: closedParam, bi: bias, mn: mean, vr: variance, dst: dst, eps: 1e-5, want: ErrClosed},
			{name: "exceeds_uint32", in: largeIn, sc: largeParam, bi: largeParam, mn: largeParam, vr: largeParam, dst: largeDst, eps: 1e-5, want: ErrInvalid},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				var scope, scopeErr = dev.NewScope()
				requireError(t, scopeErr, nil)
				defer scope.Close()
				requireError(t, scope.BatchNorm(tc.in, tc.sc, tc.bi, tc.mn, tc.vr, tc.dst, tc.eps), tc.want)
			})
		}
	})
}
