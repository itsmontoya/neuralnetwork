package gpu

import (
	"context"
	"math"
	"slices"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_Conv2DShape(t *testing.T) {
	t.Run("valid_matches_tensor", func(t *testing.T) {
		type testcase struct {
			name    string
			input   []int64
			weights []int64
			config  Conv2DConfig
			want    []int64
		}
		var tests = []testcase{
			{
				name:    "1x1_pointwise",
				input:   []int64{4, 256, 56, 56},
				weights: []int64{64, 256, 1, 1},
				config:  Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1},
				want:    []int64{4, 64, 56, 56},
			},
			{
				name:    "3x3_same_pad1",
				input:   []int64{2, 64, 28, 28},
				weights: []int64{128, 64, 3, 3},
				config:  Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1},
				want:    []int64{2, 128, 28, 28},
			},
			{
				name:    "7x7_stride2_pad3_resnet_stem",
				input:   []int64{8, 3, 224, 224},
				weights: []int64{64, 3, 7, 7},
				config:  Conv2DConfig{Strides: [2]int64{2, 2}, Pads: [4]int64{3, 3, 3, 3}, Dilations: [2]int64{1, 1}, Groups: 1},
				want:    []int64{8, 64, 112, 112},
			},
			{
				name:    "asymmetric_pads_and_dilations",
				input:   []int64{2, 6, 11, 13},
				weights: []int64{10, 3, 3, 2},
				config:  Conv2DConfig{Strides: [2]int64{2, 1}, Pads: [4]int64{1, 2, 0, 1}, Dilations: [2]int64{2, 3}, Groups: 2},
				want:    []int64{2, 10, 4, 13},
			},
			{
				name:    "depthwise",
				input:   []int64{3, 16, 14, 14},
				weights: []int64{16, 1, 3, 3},
				config:  Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 16},
				want:    []int64{3, 16, 14, 14},
			},
			{
				name:    "vit_patch_embed",
				input:   []int64{4, 3, 224, 224},
				weights: []int64{768, 3, 16, 16},
				config:  Conv2DConfig{Strides: [2]int64{16, 16}, Dilations: [2]int64{1, 1}, Groups: 1},
				want:    []int64{4, 768, 14, 14},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var got, err = Conv2DShape(tt.input, tt.weights, tt.config)
				requireError(t, err, nil)
				if !slices.Equal(got, tt.want) {
					t.Fatalf("Conv2DShape(%v, %v, %+v) = %v, want %v", tt.input, tt.weights, tt.config, got, tt.want)
				}
				var refCfg = tensor.Conv2DConfig{
					Strides:   tt.config.Strides,
					Dilations: tt.config.Dilations,
					Pads:      tt.config.Pads,
					Groups:    tt.config.Groups,
				}
				var refShape, refErr = tensor.Conv2DShape(tt.input, tt.weights, refCfg)
				requireError(t, refErr, nil)
				if !slices.Equal(got, refShape) {
					t.Fatalf("Conv2DShape = %v differs from tensor.Conv2DShape = %v", got, refShape)
				}
			})
		}
	})

	t.Run("invalid", func(t *testing.T) {
		var validCfg = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		type testcase struct {
			name    string
			input   []int64
			weights []int64
			config  Conv2DConfig
		}
		var tests = []testcase{
			{name: "input_rank3", input: []int64{1, 3, 8}, weights: []int64{4, 3, 3, 3}, config: validCfg},
			{name: "weights_rank3", input: []int64{1, 3, 8, 8}, weights: []int64{4, 3, 3}, config: validCfg},
			{name: "zero_input_dim", input: []int64{1, 3, 0, 8}, weights: []int64{4, 3, 3, 3}, config: validCfg},
			{name: "negative_weight_dim", input: []int64{1, 3, 8, 8}, weights: []int64{4, 3, -1, 3}, config: validCfg},
			{name: "zero_groups", input: []int64{1, 3, 8, 8}, weights: []int64{4, 3, 3, 3}, config: Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 0}},
			{name: "negative_groups", input: []int64{1, 3, 8, 8}, weights: []int64{4, 3, 3, 3}, config: Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: -1}},
			{name: "in_channels_not_divisible", input: []int64{1, 5, 8, 8}, weights: []int64{4, 2, 3, 3}, config: Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 2}},
			{name: "out_channels_not_divisible", input: []int64{1, 4, 8, 8}, weights: []int64{5, 2, 3, 3}, config: Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 2}},
			{name: "weights_in_channels_mismatch", input: []int64{1, 4, 8, 8}, weights: []int64{4, 3, 3, 3}, config: Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Groups: 2}},
			{name: "zero_stride", input: []int64{1, 3, 8, 8}, weights: []int64{4, 3, 3, 3}, config: Conv2DConfig{Strides: [2]int64{0, 1}, Dilations: [2]int64{1, 1}, Groups: 1}},
			{name: "zero_dilation", input: []int64{1, 3, 8, 8}, weights: []int64{4, 3, 3, 3}, config: Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 0}, Groups: 1}},
			{name: "negative_pad", input: []int64{1, 3, 8, 8}, weights: []int64{4, 3, 3, 3}, config: Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{0, 0, -1, 0}, Dilations: [2]int64{1, 1}, Groups: 1}},
			{name: "kernel_larger_than_input", input: []int64{1, 3, 2, 2}, weights: []int64{4, 3, 3, 3}, config: validCfg},
			{name: "input_exceeds_uint32", input: []int64{1, 1, 1 << 16, 1 << 16}, weights: []int64{1, 1, 1, 1}, config: validCfg},
			{name: "output_exceeds_uint32", input: []int64{1, 1, 1 << 15, 1 << 15}, weights: []int64{8, 1, 1, 1}, config: validCfg},
			{name: "pad_overflow", input: []int64{1, 1, 8, 8}, weights: []int64{1, 1, 1, 1}, config: Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{math.MaxInt64, 0, 1, 0}, Dilations: [2]int64{1, 1}, Groups: 1}},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var _, err = Conv2DShape(tt.input, tt.weights, tt.config)
				requireError(t, err, ErrInvalid)
			})
		}
	})
}

func Test_Scope_Conv2D_Validation(t *testing.T) {
	var (
		backend    = &fakeBackend{}
		dev        = newDevice(backend, 1<<30)
		otherDev   = newDevice(&fakeBackend{}, 1<<30)
		ctx        = context.Background()
		cfg        = Conv2DConfig{Strides: [2]int64{1, 1}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1}, Groups: 1}
		in         = requireTensor(t, dev, 2, 4, 4, 4)
		wF32       = requireTensor(t, dev, 6, 4, 3, 3)
		wF16       = requireTensorWithDType(t, dev, Float16, 6, 4, 3, 3)
		wBF16      = requireTensorWithDType(t, dev, BFloat16, 6, 4, 3, 3)
		bF32       = requireTensor(t, dev, 6)
		bF16       = requireTensorWithDType(t, dev, Float16, 6)
		bBF16      = requireTensorWithDType(t, dev, BFloat16, 6)
		out        = requireTensor(t, dev, 2, 6, 4, 4)
		wrongOut   = requireTensor(t, dev, 2, 6, 3, 3)
		wrongBias  = requireTensor(t, dev, 5)
		outF16     = requireTensorWithDType(t, dev, Float16, 2, 6, 4, 4)
		inF16      = requireTensorWithDType(t, dev, Float16, 2, 4, 4, 4)
		otherIn    = requireTensor(t, otherDev, 2, 4, 4, 4)
		uninitIn   = requireTensor(t, dev, 2, 4, 4, 4)
		wQ8        = requireTensorWithDType(t, dev, Q8_0, 6, 32)
		bQ4        = requireTensorWithDType(t, dev, Q4_0, 1, 32)
		inAlias, _ = in.View([]int64{2, 4, 4, 4})
		wAlias, _  = wF32.View([]int64{6, 4, 3, 3})
	)
	defer dev.Close()
	defer otherDev.Close()
	defer inAlias.Close()
	defer wAlias.Close()

	requireError(t, in.Upload(make([]float32, 2*4*4*4)), nil)
	requireError(t, wF32.Upload(make([]float32, 6*4*3*3)), nil)
	requireError(t, wF16.UploadFloat32As(make([]float32, 6*4*3*3)), nil)
	requireError(t, wBF16.UploadFloat32As(make([]float32, 6*4*3*3)), nil)
	requireError(t, bF32.Upload(make([]float32, 6)), nil)
	requireError(t, bF16.UploadFloat32As(make([]float32, 6)), nil)
	requireError(t, bBF16.UploadFloat32As(make([]float32, 6)), nil)
	requireError(t, inF16.UploadFloat32As(make([]float32, 2*4*4*4)), nil)
	requireError(t, otherIn.Upload(make([]float32, 2*4*4*4)), nil)
	requireError(t, wrongBias.Upload(make([]float32, 5)), nil)
	requireError(t, wQ8.UploadFloat32As(make([]float32, 6*32)), nil)
	requireError(t, bQ4.UploadFloat32As(make([]float32, 32)), nil)

	// Valid combinations: FP32/FP16/BF16 weights x nil/FP32/FP16/BF16 bias.
	var validWeights = []*Tensor{wF32, wF16, wBF16}
	var validBiases = []*Tensor{nil, bF32, bF16, bBF16}
	for _, w := range validWeights {
		for _, b := range validBiases {
			var scope, err = dev.NewScope()
			requireError(t, err, nil)
			requireError(t, scope.Conv2D(in, w, b, out, cfg), nil)
			requireError(t, scope.Submit(ctx), nil)
			requireError(t, scope.Wait(ctx), nil)
			requireError(t, scope.Close(), nil)
		}
	}
	if backend.conv2Ds.Load() != int32(len(validWeights)*len(validBiases)) {
		t.Fatalf("encoded conv2D count = %d, want %d", backend.conv2Ds.Load(), len(validWeights)*len(validBiases))
	}

	// Alias via Tensor.View (matching output shape).
	var sameIn = requireTensor(t, dev, 1, 2, 4, 4)
	var sameW = requireTensor(t, dev, 2, 2, 3, 3)
	requireError(t, sameIn.Upload(make([]float32, 1*2*4*4)), nil)
	requireError(t, sameW.Upload(make([]float32, 2*2*3*3)), nil)
	var sameInView, vErr = sameIn.View([]int64{1, 2, 4, 4})
	requireError(t, vErr, nil)
	defer sameInView.Close()

	type errCase struct {
		name string
		run  func(scope *Scope) error
		want error
	}
	var errCases = []errCase{
		{name: "nil_input", run: func(s *Scope) error { return s.Conv2D(nil, wF32, bF32, out, cfg) }, want: ErrInvalid},
		{name: "nil_weights", run: func(s *Scope) error { return s.Conv2D(in, nil, bF32, out, cfg) }, want: ErrInvalid},
		{name: "nil_destination", run: func(s *Scope) error { return s.Conv2D(in, wF32, bF32, nil, cfg) }, want: ErrInvalid},
		{name: "non_f32_input", run: func(s *Scope) error { return s.Conv2D(inF16, wF32, bF32, out, cfg) }, want: ErrInvalid},
		{name: "non_f32_destination", run: func(s *Scope) error { return s.Conv2D(in, wF32, bF32, outF16, cfg) }, want: ErrInvalid},
		{name: "quantized_weights", run: func(s *Scope) error { return s.Conv2D(in, wQ8, nil, out, cfg) }, want: ErrInvalid},
		{name: "quantized_bias", run: func(s *Scope) error { return s.Conv2D(in, wF32, bQ4, out, cfg) }, want: ErrInvalid},
		{name: "wrong_destination_shape", run: func(s *Scope) error { return s.Conv2D(in, wF32, bF32, wrongOut, cfg) }, want: ErrInvalid},
		{name: "wrong_bias_shape", run: func(s *Scope) error { return s.Conv2D(in, wF32, wrongBias, out, cfg) }, want: ErrInvalid},
		{name: "invalid_config", run: func(s *Scope) error { return s.Conv2D(in, wF32, bF32, out, Conv2DConfig{}) }, want: ErrInvalid},
		{name: "direct_input_alias", run: func(s *Scope) error { return s.Conv2D(sameIn, sameW, nil, sameIn, cfg) }, want: ErrInvalid},
		{name: "view_input_alias", run: func(s *Scope) error { return s.Conv2D(sameIn, sameW, nil, sameInView, cfg) }, want: ErrInvalid},
		{name: "cross_device_input", run: func(s *Scope) error { return s.Conv2D(otherIn, wF32, bF32, out, cfg) }, want: ErrInvalid},
		{name: "uninitialized_input", run: func(s *Scope) error { return s.Conv2D(uninitIn, wF32, bF32, out, cfg) }, want: ErrUninitialized},
	}
	for _, tc := range errCases {
		t.Run(tc.name, func(t *testing.T) {
			var scope, err = dev.NewScope()
			requireError(t, err, nil)
			defer scope.Close()
			requireError(t, tc.run(scope), tc.want)
		})
	}
}
