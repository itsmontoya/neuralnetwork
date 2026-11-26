package gpu

import (
	"context"
	"math"
	"slices"
	"testing"

	"github.com/itsmontoya/neuralnetwork/tensor"
)

func Test_Pool2DShape(t *testing.T) {
	t.Run("valid_matches_tensor", func(t *testing.T) {
		type testcase struct {
			name   string
			input  []int64
			config Pool2DConfig
			want   []int64
		}
		var tests = []testcase{
			{
				name:  "2x2_stride2",
				input: []int64{2, 4, 8, 8},
				config: Pool2DConfig{
					Kernel: [2]int64{2, 2}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1},
				},
				want: []int64{2, 4, 4, 4},
			},
			{
				name:  "3x3_stride2_pad1_resnet_stem",
				input: []int64{32, 64, 112, 112},
				config: Pool2DConfig{
					Kernel: [2]int64{3, 3}, Strides: [2]int64{2, 2}, Pads: [4]int64{1, 1, 1, 1}, Dilations: [2]int64{1, 1},
				},
				want: []int64{32, 64, 56, 56},
			},
			{
				name:  "odd_floor",
				input: []int64{1, 3, 7, 7},
				config: Pool2DConfig{
					Kernel: [2]int64{2, 2}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1},
				},
				want: []int64{1, 3, 3, 3},
			},
			{
				name:  "odd_ceil",
				input: []int64{1, 3, 7, 7},
				config: Pool2DConfig{
					Kernel: [2]int64{2, 2}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1}, CeilMode: true,
				},
				want: []int64{1, 3, 4, 4},
			},
			{
				name:  "ceil_trailing_window_clamped",
				input: []int64{1, 1, 4, 4},
				config: Pool2DConfig{
					Kernel: [2]int64{3, 3}, Strides: [2]int64{2, 2}, Pads: [4]int64{0, 0, 1, 1}, Dilations: [2]int64{1, 1}, CeilMode: true,
				},
				want: []int64{1, 1, 2, 2},
			},
			{
				name:  "ceil_padded_smaller_than_effective_within_stride",
				input: []int64{1, 2, 2, 2},
				config: Pool2DConfig{
					Kernel: [2]int64{3, 3}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1}, CeilMode: true,
				},
				want: []int64{1, 2, 1, 1},
			},
			{
				name:  "asymmetric_pads_and_dilations",
				input: []int64{2, 5, 9, 10},
				config: Pool2DConfig{
					Kernel: [2]int64{3, 2}, Strides: [2]int64{2, 1}, Pads: [4]int64{1, 2, 0, 1}, Dilations: [2]int64{2, 3},
				},
				want: []int64{2, 5, 3, 10},
			},
			{
				name:  "global_pool",
				input: []int64{32, 512, 7, 7},
				config: Pool2DConfig{
					Kernel: [2]int64{7, 7}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1},
				},
				want: []int64{32, 512, 1, 1},
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				var got, err = Pool2DShape(tc.input, tc.config)
				requireError(t, err, nil)
				if !slices.Equal(got, tc.want) {
					t.Fatalf("Pool2DShape(%v) = %v, want %v", tc.input, got, tc.want)
				}
				var refCfg = tensor.Pool2DConfig{
					Kernel:          tc.config.Kernel,
					Strides:         tc.config.Strides,
					Dilations:       tc.config.Dilations,
					Pads:            tc.config.Pads,
					CeilMode:        tc.config.CeilMode,
					CountIncludePad: tc.config.CountIncludePad,
				}
				var ref, refErr = tensor.Pool2DShape(tc.input, refCfg)
				requireError(t, refErr, nil)
				if !slices.Equal(got, ref) {
					t.Fatalf("Pool2DShape(%v) = %v, tensor.Pool2DShape = %v", tc.input, got, ref)
				}
			})
		}
	})

	t.Run("invalid_errors", func(t *testing.T) {
		type testcase struct {
			name   string
			input  []int64
			config Pool2DConfig
		}
		var validCfg = Pool2DConfig{
			Kernel: [2]int64{2, 2}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1},
		}
		var tests = []testcase{
			{name: "rank3", input: []int64{1, 2, 4}, config: validCfg},
			{name: "rank5", input: []int64{1, 2, 4, 4, 1}, config: validCfg},
			{name: "zero_dim", input: []int64{1, 0, 4, 4}, config: validCfg},
			{name: "negative_dim", input: []int64{1, 2, -4, 4}, config: validCfg},
			{name: "zero_config", input: []int64{1, 2, 4, 4}, config: Pool2DConfig{}},
			{name: "zero_kernel", input: []int64{1, 2, 4, 4}, config: Pool2DConfig{Kernel: [2]int64{0, 2}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}}},
			{name: "zero_stride", input: []int64{1, 2, 4, 4}, config: Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{1, 0}, Dilations: [2]int64{1, 1}}},
			{name: "zero_dilation", input: []int64{1, 2, 4, 4}, config: Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{1, 1}, Dilations: [2]int64{0, 1}}},
			{name: "negative_pad", input: []int64{1, 2, 4, 4}, config: Pool2DConfig{Kernel: [2]int64{2, 2}, Strides: [2]int64{1, 1}, Pads: [4]int64{0, -1, 0, 0}, Dilations: [2]int64{1, 1}}},
			{name: "nonpositive_floor_output", input: []int64{1, 2, 2, 2}, config: Pool2DConfig{Kernel: [2]int64{3, 3}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}}},
			{name: "nonpositive_ceil_output", input: []int64{1, 2, 2, 2}, config: Pool2DConfig{Kernel: [2]int64{5, 5}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1}, CeilMode: true}},
			{name: "overflow_effective", input: []int64{1, 2, 4, 4}, config: Pool2DConfig{Kernel: [2]int64{math.MaxInt64, 2}, Strides: [2]int64{1, 1}, Dilations: [2]int64{2, 1}}},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				var _, err = Pool2DShape(tc.input, tc.config)
				requireError(t, err, ErrInvalid)
			})
		}
	})
}

func Test_Scope_Pool2D(t *testing.T) {
	t.Run("valid_max_and_avg", func(t *testing.T) {
		var (
			backend = &fakeBackend{}
			dev     = newDevice(backend, ^uint64(0))
			ctx     = context.Background()
			input   = mustNewTensor(t, dev, 2, 4, 6, 6)
			dst     = mustNewTensor(t, dev, 2, 4, 3, 3)
			cfg     = Pool2DConfig{
				Kernel: [2]int64{2, 2}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1},
			}
		)
		requireError(t, input.Upload(make([]float32, input.state.storage.count)), nil)

		var scope, err = dev.NewScope()
		requireError(t, err, nil)
		requireError(t, scope.MaxPool2D(input, dst, cfg), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)

		scope, err = dev.NewScope()
		requireError(t, err, nil)
		requireError(t, scope.AvgPool2D(input, dst, cfg), nil)
		requireError(t, scope.Submit(ctx), nil)
		requireError(t, scope.Wait(ctx), nil)
		requireError(t, scope.Close(), nil)

		if got := backend.pool2Ds.Load(); got != 2 {
			t.Fatalf("expected 2 Pool2D dispatches, got %d", got)
		}
		requireError(t, input.Close(), nil)
		requireError(t, dst.Close(), nil)
		requireError(t, dev.Close(), nil)
	})

	t.Run("validation_errors", func(t *testing.T) {
		var (
			dev       = newDevice(&fakeBackend{}, ^uint64(0))
			otherDev  = newDevice(&fakeBackend{}, ^uint64(0))
			input     = mustNewTensor(t, dev, 2, 4, 4, 4)
			sameDst   = mustNewTensor(t, dev, 2, 4, 4, 4)
			dst       = mustNewTensor(t, dev, 2, 4, 2, 2)
			wrongDst  = mustNewTensor(t, dev, 2, 4, 2, 3)
			rank3In   = mustNewTensor(t, dev, 2, 4, 4)
			f16In     = mustNewTensorWithDType(t, dev, Float16, 2, 4, 4, 4)
			f16Dst    = mustNewTensorWithDType(t, dev, Float16, 2, 4, 2, 2)
			uninitIn  = mustNewTensor(t, dev, 2, 4, 4, 4)
			otherIn   = mustNewTensor(t, otherDev, 2, 4, 4, 4)
			closedIn  = mustNewTensor(t, dev, 2, 4, 4, 4)
			largeIn   = mustNewTensor(t, dev, 1<<17, 1<<16, 1, 1)
			largeDst  = mustNewTensor(t, dev, 1<<17, 1<<16, 1, 1)
			inputView *Tensor
			err       error
			cfg       = Pool2DConfig{
				Kernel: [2]int64{2, 2}, Strides: [2]int64{2, 2}, Dilations: [2]int64{1, 1},
			}
			sameCfg = Pool2DConfig{
				Kernel: [2]int64{1, 1}, Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1},
			}
		)
		defer func() {
			for _, ten := range []*Tensor{
				input, sameDst, dst, wrongDst, rank3In, f16In, f16Dst,
				uninitIn, otherIn, largeIn, largeDst, inputView,
			} {
				if ten != nil {
					_ = ten.Close()
				}
			}
			_ = dev.Close()
			_ = otherDev.Close()
		}()

		requireError(t, input.Upload(make([]float32, 128)), nil)
		requireError(t, rank3In.Upload(make([]float32, 32)), nil)
		requireError(t, f16In.UploadFloat32As(make([]float32, 128)), nil)
		requireError(t, otherIn.Upload(make([]float32, 128)), nil)
		requireError(t, closedIn.Upload(make([]float32, 128)), nil)
		requireError(t, closedIn.Close(), nil)

		inputView, err = input.View([]int64{2, 4, 4, 4})
		requireError(t, err, nil)

		type testcase struct {
			name string
			in   *Tensor
			dst  *Tensor
			cfg  Pool2DConfig
			want error
		}
		var tests = []testcase{
			{name: "nil_input", in: nil, dst: dst, cfg: cfg, want: ErrInvalid},
			{name: "nil_dst", in: input, dst: nil, cfg: cfg, want: ErrInvalid},
			{name: "rank3_input", in: rank3In, dst: dst, cfg: cfg, want: ErrInvalid},
			{name: "invalid_config", in: input, dst: dst, cfg: Pool2DConfig{}, want: ErrInvalid},
			{name: "dst_shape_mismatch", in: input, dst: wrongDst, cfg: cfg, want: ErrInvalid},
			{name: "f16_input", in: f16In, dst: dst, cfg: cfg, want: ErrInvalid},
			{name: "f16_dst", in: input, dst: f16Dst, cfg: cfg, want: ErrInvalid},
			{name: "direct_alias", in: input, dst: input, cfg: sameCfg, want: ErrInvalid},
			{name: "view_alias", in: input, dst: inputView, cfg: sameCfg, want: ErrInvalid},
			{name: "uninit_input", in: uninitIn, dst: dst, cfg: cfg, want: ErrUninitialized},
			{name: "other_device", in: otherIn, dst: dst, cfg: cfg, want: ErrInvalid},
			{name: "closed_input", in: closedIn, dst: dst, cfg: cfg, want: ErrClosed},
			{name: "exceeds_uint32", in: largeIn, dst: largeDst, cfg: sameCfg, want: ErrInvalid},
			{
				name: "exceeds_int32_pad",
				in:   input,
				dst:  sameDst,
				cfg: Pool2DConfig{
					Kernel: [2]int64{1, 1}, Strides: [2]int64{1, 1}, Pads: [4]int64{math.MaxInt32, 0, 0, 0}, Dilations: [2]int64{1, 1},
				},
				want: ErrInvalid,
			},
		}
		for _, tc := range tests {
			t.Run("MaxPool2D/"+tc.name, func(t *testing.T) {
				var scope, scopeErr = dev.NewScope()
				requireError(t, scopeErr, nil)
				defer scope.Close()
				requireError(t, scope.MaxPool2D(tc.in, tc.dst, tc.cfg), tc.want)
			})
			t.Run("AvgPool2D/"+tc.name, func(t *testing.T) {
				var scope, scopeErr = dev.NewScope()
				requireError(t, scopeErr, nil)
				defer scope.Close()
				requireError(t, scope.AvgPool2D(tc.in, tc.dst, tc.cfg), tc.want)
			})
		}
	})
}
