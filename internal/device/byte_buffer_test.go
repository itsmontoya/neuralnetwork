package device

import (
	"slices"
	"testing"
)

func Test_RuntimeByteBufferRoundTripAndCopy(t *testing.T) {
	var (
		backend = newTestBackend()
		runtime = newRuntime(backend)
		source  *Buffer
		target  *Buffer
		other   *Buffer
		scope   *Scope
		err     error
	)

	if _, err = runtime.NewByteBuffer(0); err == nil {
		t.Fatal("expected error for zero byte length")
	}

	if source, err = runtime.NewByteBuffer(6); err != nil {
		t.Fatalf("NewByteBuffer source returned error: %v", err)
	}
	defer source.Release()
	if source.Bytes() != 6 || source.Count() != 0 {
		t.Fatalf("source metadata = (bytes=%d, count=%d), want (6, 0)", source.Bytes(), source.Count())
	}

	if target, err = runtime.NewByteBuffer(6); err != nil {
		t.Fatalf("NewByteBuffer target returned error: %v", err)
	}
	defer target.Release()

	if other, err = runtime.NewByteBuffer(8); err != nil {
		t.Fatalf("NewByteBuffer other returned error: %v", err)
	}
	defer other.Release()

	var rawIn = []byte{0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc}
	var rawOut = make([]byte, 6)
	if err = source.UploadBytes(rawIn[:4]); err == nil {
		t.Fatal("expected error for short UploadBytes")
	}
	if err = source.UploadBytes(rawIn); err != nil {
		t.Fatalf("UploadBytes returned error: %v", err)
	}
	if err = source.DownloadBytes(rawOut[:4]); err == nil {
		t.Fatal("expected error for short DownloadBytes")
	}
	if err = source.DownloadBytes(rawOut); err != nil {
		t.Fatalf("DownloadBytes returned error: %v", err)
	}
	if !slices.Equal(rawOut, rawIn) {
		t.Fatalf("DownloadBytes = %x, want %x", rawOut, rawIn)
	}

	var u16In = []uint16{0x3c00, 0xbc00, 0x7bff}
	var u16Out = make([]uint16, 3)
	if err = source.UploadUint16(u16In[:2]); err == nil {
		t.Fatal("expected error for short UploadUint16")
	}
	if err = source.UploadUint16(u16In); err != nil {
		t.Fatalf("UploadUint16 returned error: %v", err)
	}
	if err = source.DownloadUint16(u16Out[:2]); err == nil {
		t.Fatal("expected error for short DownloadUint16")
	}
	if err = source.DownloadUint16(u16Out); err != nil {
		t.Fatalf("DownloadUint16 returned error: %v", err)
	}
	if !slices.Equal(u16Out, u16In) {
		t.Fatalf("DownloadUint16 = %x, want %x", u16Out, u16In)
	}

	if scope, err = runtime.NewScope(); err != nil {
		t.Fatalf("NewScope returned error: %v", err)
	}
	defer scope.Release()
	if err = scope.EncodeCopy(source, other); err == nil {
		t.Fatal("expected error copying buffers with different byte lengths")
	}
	if err = scope.EncodeCopy(source, target); err != nil {
		t.Fatalf("EncodeCopy returned error: %v", err)
	}
	if err = scope.Commit(); err != nil {
		t.Fatalf("Commit returned error: %v", err)
	}
	if err = scope.Wait(); err != nil {
		t.Fatalf("Wait returned error: %v", err)
	}

	clear(u16Out)
	if err = target.DownloadUint16(u16Out); err != nil {
		t.Fatalf("DownloadUint16 target returned error: %v", err)
	}
	if !slices.Equal(u16Out, u16In) {
		t.Fatalf("copied uint16 = %x, want %x", u16Out, u16In)
	}
}

func Test_ScopeEncodeMixedMatMulValidation(t *testing.T) {
	var (
		backend     = newTestBackend()
		runtime     = newRuntime(backend)
		left, err1  = runtime.NewBuffer(6)
		right, err2 = runtime.NewByteBuffer(24)
		wrong, err3 = runtime.NewBuffer(12)
		short, err4 = runtime.NewByteBuffer(20)
		out, err5   = runtime.NewBuffer(8)
		scope, err6 = runtime.NewScope()
	)
	for _, err := range []error{err1, err2, err3, err4, err5, err6} {
		if err != nil {
			t.Fatalf("setup failed: %v", err)
		}
	}
	defer left.Release()
	defer right.Release()
	defer wrong.Release()
	defer short.Release()
	defer out.Release()
	defer scope.Release()

	if err := scope.EncodeMatMul(left, right, out, 2, 3, 3, 4, 2, 4, OperationTensorMatMulFloat16); err != nil {
		t.Fatalf("expected valid FP16 matmul, got %v", err)
	}
	if err := scope.EncodeMatMul(left, right, out, 2, 3, 4, 3, 2, 4, OperationTensorMatMulRightTransposeFloat16); err != nil {
		t.Fatalf("expected valid FP16 right-transposed matmul, got %v", err)
	}
	if err := scope.EncodeMatMul(left, right, out, 2, 3, 3, 4, 2, 4, OperationTensorMatMulBFloat16); err != nil {
		t.Fatalf("expected valid BF16 matmul, got %v", err)
	}
	if err := scope.EncodeMatMul(left, right, out, 2, 3, 4, 3, 2, 4, OperationTensorMatMulRightTransposeBFloat16); err != nil {
		t.Fatalf("expected valid BF16 right-transposed matmul, got %v", err)
	}

	// FP32 right buffer, wrong byte length, and mismatched dimensions must fail.
	if err := scope.EncodeMatMul(left, wrong, out, 2, 3, 3, 4, 2, 4, OperationTensorMatMulFloat16); err == nil {
		t.Fatal("expected error when passing FP32 buffer as FP16 right operand")
	}
	if err := scope.EncodeMatMul(left, short, out, 2, 3, 3, 4, 2, 4, OperationTensorMatMulBFloat16); err == nil {
		t.Fatal("expected error when passing undersized byte buffer as BF16 right operand")
	}
	if err := scope.EncodeMatMul(left, right, out, 2, 3, 3, 4, 2, 4, OperationTensorMatMulRightTransposeFloat16); err == nil {
		t.Fatal("expected error on transposed inner dimension mismatch")
	}
	if err := scope.EncodeMatMul(left, right, out, 2, 3, 3, 4, 2, 4, OperationTensorMatMulRightTransposeBFloat16+1); err == nil {
		t.Fatal("expected error on unknown matmul operation")
	}
}
