package gpu

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"
)

func Test_DeviceNewTensorWithDTypeValidation(t *testing.T) {
	type testcase struct {
		name  string
		shape []int64
		dtype DType
	}

	var (
		b     fakeBackend
		d     = newDevice(&b, 128)
		tests = []testcase{
			{name: "zero dtype", shape: []int64{4}, dtype: 0},
			{name: "unknown dtype", shape: []int64{4}, dtype: DType(99)},
			{name: "empty shape float16", shape: []int64{}, dtype: Float16},
			{name: "negative dim bfloat16", shape: []int64{2, -3}, dtype: BFloat16},
			{name: "float16 over device limit", shape: []int64{65}, dtype: Float16},
			{name: "bfloat16 over device limit", shape: []int64{65}, dtype: BFloat16},
			{name: "float16 element overflow", shape: []int64{math.MaxInt64, 2}, dtype: Float16},
		}
	)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out, err = d.NewTensorWithDType(tt.shape, tt.dtype)
			if out != nil || !errors.Is(err, ErrInvalid) {
				t.Fatalf("NewTensorWithDType(%v, %v) = %v, %v, want nil and ErrInvalid", tt.shape, tt.dtype, out, err)
			}
		})
	}

	// Exact device limit for 2-byte dtypes: 128 / 2 = 64 elements succeeds.
	var f16, err16 = d.NewTensorWithDType([]int64{8, 8}, Float16)
	requireError(t, err16, nil)
	if f16.DType() != Float16 || f16.Bytes() != 128 {
		t.Fatalf("f16 metadata = (%v, %d), want (Float16, 128)", f16.DType(), f16.Bytes())
	}
	requireError(t, f16.Close(), nil)
	if f16.DType() != Float16 || f16.Bytes() != 128 {
		t.Fatalf("closed f16 lost metadata: (%v, %d)", f16.DType(), f16.Bytes())
	}

	var nilTensor *Tensor
	if nilTensor.DType().Valid() || nilTensor.Bytes() != 0 {
		t.Fatalf("nil tensor metadata = (%v, %d), want (invalid, 0)", nilTensor.DType(), nilTensor.Bytes())
	}

	// Host byte overflow check for 2-byte dtypes.
	var hugeDevice = newDevice(&b, math.MaxUint64)
	var _, hostErr = hugeDevice.NewTensorWithDType([]int64{int64(math.MaxInt/2) + 1}, BFloat16)
	requireError(t, hostErr, ErrInvalid)
	requireError(t, hugeDevice.Close(), nil)
	requireError(t, d.Close(), nil)
}

func Test_TensorDTypeTransfersAndValidation(t *testing.T) {
	var (
		b   fakeBackend
		d   = newDevice(&b, 1024)
		f32 = requireTensorWithDType(t, d, Float32, 3)
		f16 = requireTensorWithDType(t, d, Float16, 3)
		bf  = requireTensorWithDType(t, d, BFloat16, 3)
	)

	// Upload / DownloadInto reject non-Float32 tensors.
	requireError(t, f16.Upload([]float32{1, 2, 3}), ErrInvalid)
	requireError(t, bf.Upload([]float32{1, 2, 3}), ErrInvalid)
	requireError(t, f16.DownloadInto(make([]float32, 3)), ErrInvalid)
	requireError(t, bf.DownloadInto(make([]float32, 3)), ErrInvalid)

	// UploadBits / DownloadBitsInto reject Float32 tensors.
	requireError(t, f32.UploadBits([]uint16{1, 2, 3}), ErrInvalid)
	requireError(t, f32.DownloadBitsInto(make([]uint16, 3)), ErrInvalid)

	// Uninitialized and length checks for 16-bit tensors.
	requireError(t, f16.DownloadBitsInto(make([]uint16, 3)), ErrUninitialized)
	requireError(t, f16.DownloadFloat32Into(make([]float32, 3)), ErrUninitialized)
	requireError(t, f16.UploadBits([]uint16{1, 2}), ErrInvalid)
	requireError(t, f16.UploadFloat32As([]float32{1, 2}), ErrInvalid)

	// Raw bits round-trip for Float16 and BFloat16.
	var raw16 = []uint16{0x3c00, 0xc000, 0x0001}
	requireError(t, f16.UploadBits(raw16), nil)
	var got16 = make([]uint16, 3)
	requireError(t, f16.DownloadBitsInto(got16), nil)
	if !slices.Equal(got16, raw16) {
		t.Fatalf("Float16 bits = %x, want %x", got16, raw16)
	}
	var gotF32 = make([]float32, 3)
	requireError(t, f16.DownloadFloat32Into(gotF32), nil)
	var wantF16AsF32 = []float32{1.0, -2.0, Float16ToFloat32(0x0001)}
	if !slices.Equal(gotF32, wantF16AsF32) {
		t.Fatalf("Float16 as float32 = %v, want %v", gotF32, wantF16AsF32)
	}

	// UploadFloat32As / DownloadFloat32Into for BFloat16.
	var srcF32 = []float32{1.5, -3.25, 0.0}
	requireError(t, bf.UploadFloat32As(srcF32), nil)
	var gotBFBits = make([]uint16, 3)
	requireError(t, bf.DownloadBitsInto(gotBFBits), nil)
	var wantBFBits = []uint16{Float32ToBFloat16(1.5), Float32ToBFloat16(-3.25), 0x0000}
	if !slices.Equal(gotBFBits, wantBFBits) {
		t.Fatalf("BFloat16 bits = %x, want %x", gotBFBits, wantBFBits)
	}
	clear(gotF32)
	requireError(t, bf.DownloadFloat32Into(gotF32), nil)
	if !slices.Equal(gotF32, srcF32) {
		t.Fatalf("BFloat16 as float32 = %v, want %v", gotF32, srcF32)
	}

	// UploadFloat32As / DownloadFloat32Into also work for Float32.
	requireError(t, f32.UploadFloat32As(srcF32), nil)
	clear(gotF32)
	requireError(t, f32.DownloadFloat32Into(gotF32), nil)
	if !slices.Equal(gotF32, srcF32) {
		t.Fatalf("Float32 as float32 = %v, want %v", gotF32, srcF32)
	}

	// Closed state checks.
	requireError(t, f16.Close(), nil)
	requireError(t, f16.UploadBits(raw16), ErrClosed)
	requireError(t, f16.DownloadBitsInto(got16), ErrClosed)
	requireError(t, f16.UploadFloat32As(srcF32), ErrClosed)
	requireError(t, f16.DownloadFloat32Into(gotF32), ErrClosed)

	requireError(t, bf.Close(), nil)
	requireError(t, f32.Close(), nil)
	requireError(t, d.Close(), nil)
}

func Test_ScopeCopyAndOpDTypeValidation(t *testing.T) {
	var (
		ctx  = context.Background()
		b    fakeBackend
		d    = newDevice(&b, 4096)
		f32A = requireTensorWithDType(t, d, Float32, 2, 2)
		f32B = requireTensorWithDType(t, d, Float32, 2, 2)
		f32W = requireTensorWithDType(t, d, Float32, 2)
		f32C = requireTensorWithDType(t, d, Float32, 2, 1)
		f16A = requireTensorWithDType(t, d, Float16, 2, 2)
		f16B = requireTensorWithDType(t, d, Float16, 2, 2)
		f16W = requireTensorWithDType(t, d, Float16, 2)
		f16C = requireTensorWithDType(t, d, Float16, 2, 1)
		bfA  = requireTensorWithDType(t, d, BFloat16, 2, 2)
		bfB  = requireTensorWithDType(t, d, BFloat16, 2, 2)
	)

	requireError(t, f32A.Upload([]float32{1, 2, 3, 4}), nil)
	requireError(t, f32W.Upload([]float32{1, 1}), nil)
	requireError(t, f32C.Upload([]float32{1, 1}), nil)
	requireError(t, f16A.UploadBits([]uint16{0x3c00, 0x4000, 0x4200, 0x4400}), nil)
	requireError(t, f16W.UploadBits([]uint16{0x3c00, 0x3c00}), nil)
	requireError(t, f16C.UploadBits([]uint16{0x3c00, 0x3c00}), nil)
	requireError(t, bfA.UploadBits([]uint16{0x3f80, 0x4000, 0x4040, 0x4080}), nil)

	// Same-dtype Copy works for Float16 and BFloat16.
	var scope = requireScope(t, d)
	requireError(t, scope.Copy(f16A, f16B), nil)
	requireError(t, scope.Copy(bfA, bfB), nil)
	requireError(t, scope.Submit(ctx), nil)
	requireError(t, scope.Wait(ctx), nil)
	requireError(t, scope.Close(), nil)

	var got16 = make([]uint16, 4)
	requireError(t, f16B.DownloadBitsInto(got16), nil)
	if !slices.Equal(got16, []uint16{0x3c00, 0x4000, 0x4200, 0x4400}) {
		t.Fatalf("Float16 copy = %x", got16)
	}
	var gotBF = make([]uint16, 4)
	requireError(t, bfB.DownloadBitsInto(gotBF), nil)
	if !slices.Equal(gotBF, []uint16{0x3f80, 0x4000, 0x4040, 0x4080}) {
		t.Fatalf("BFloat16 copy = %x", gotBF)
	}

	var cache, cacheErr = NewKVCache(d, 4, 1, 2)
	requireError(t, cacheErr, nil)
	t.Cleanup(func() { requireError(t, cache.Close(), nil) })

	// Mismatched Copy and all FP32-only compute ops reject non-Float32 tensors with ErrInvalid.
	type opCase struct {
		name string
		call func(s *Scope) error
	}
	var opCases = []opCase{
		{name: "Copy f16->f32", call: func(s *Scope) error { return s.Copy(f16A, f32B) }},
		{name: "Copy f16->bf16", call: func(s *Scope) error { return s.Copy(f16A, bfB) }},
		{name: "Fill f16", call: func(s *Scope) error { return s.Fill(f16B, 1) }},
		{name: "Fill bf16", call: func(s *Scope) error { return s.Fill(bfB, 1) }},
		{name: "GatherRows f16 src", call: func(s *Scope) error { return s.GatherRows(f16A, []int64{0, 1}, f32B) }},
		{name: "GatherRows f16 dst", call: func(s *Scope) error { return s.GatherRows(f32A, []int64{0, 1}, f16B) }},
		{name: "Add f16", call: func(s *Scope) error { return s.Add(f16A, f32A, f32B) }},
		{name: "Mul bf16", call: func(s *Scope) error { return s.Mul(f32A, bfA, f32B) }},
		{name: "SiLU f16", call: func(s *Scope) error { return s.SiLU(f16A, f16B) }},
		{name: "SwiGLU bf16", call: func(s *Scope) error { return s.SwiGLU(f32A, f32A, bfB) }},
		{name: "MatMul f16 right", call: func(s *Scope) error { return s.MatMul(f32A, f16A, f32B) }},
		{name: "MatMulRightTranspose bf16 right", call: func(s *Scope) error { return s.MatMulRightTranspose(f32A, bfA, f32B) }},
		{name: "RMSNorm f16 weight", call: func(s *Scope) error { return s.RMSNorm(f32A, f16W, f32B, 1e-5) }},
		{name: "RMSNorm f16 input", call: func(s *Scope) error { return s.RMSNorm(f16A, f32W, f16B, 1e-5) }},
		{name: "RotatePairs f16 cos", call: func(s *Scope) error { return s.RotatePairs(f32A, f16C, f32C, f32B, 2, 0, 0) }},
		{name: "RotatePairs bf16 input", call: func(s *Scope) error { return s.RotatePairs(bfA, f32C, f32C, bfB, 2, 0, 0) }},
		{name: "AppendKV f16 keys", call: func(s *Scope) error { _, err := s.AppendKV(cache, f16A, f32A, 0); return err }},
		{name: "AppendKV bf16 values", call: func(s *Scope) error { _, err := s.AppendKV(cache, f32A, bfA, 0); return err }},
		{name: "CausalAttention f16 queries", call: func(s *Scope) error { return s.CausalAttention(f16A, cache, f16B, 0) }},
		{name: "CausalAttention bf16 dst", call: func(s *Scope) error { return s.CausalAttention(f32A, cache, bfB, 0) }},
	}

	for _, tt := range opCases {
		t.Run(tt.name, func(t *testing.T) {
			var s = requireScope(t, d)
			requireError(t, tt.call(s), ErrInvalid)
			requireError(t, s.Close(), nil)
		})
	}
}

func requireTensorWithDType(t testing.TB, d *Device, dtype DType, shape ...int64) (out *Tensor) {
	t.Helper()
	var tensor, err = d.NewTensorWithDType(shape, dtype)
	requireError(t, err, nil)
	t.Cleanup(func() { requireError(t, tensor.Close(), nil) })
	return tensor
}
