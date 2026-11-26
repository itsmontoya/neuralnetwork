//go:build darwin && arm64 && cgo && metal && !purego && !race

package tensor

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/float32product"
)

func Test_MatrixProduct_MetalPacksOnARM64(t *testing.T) {
	const (
		inner = 128
		cols  = 256
	)

	var (
		values = make([]float32, cols*inner)
		weight *PreparedRightTranspose
		err    error
	)

	if !float32product.CanPackRightTranspose(inner, cols) {
		t.Fatalf("expected ARM64 right-transpose packing to be eligible for inner=%d cols=%d", inner, cols)
	}
	if weight, err = PrepareRightTranspose([]int64{cols, inner}, values); err != nil {
		t.Fatal(err)
	}
	if !weight.packed {
		t.Fatal("expected PrepareRightTranspose to pack weights on ARM64 under -tags=metal")
	}
}
