//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// Benchmark_TiedEmbeddingClassifierMetal measures tied Llama 3.2 1B
// [128256, 2048] GatherRows (M=1, M=29), MatMulRightTranspose (M=1), and
// combined single-scope GatherRows + MatMulRightTranspose across BFloat16,
// Q8_0, and Q4_0.
func Benchmark_TiedEmbeddingClassifierMetal(b *testing.B) {
	const (
		vocab int64 = 128256
		width int64 = 2048
	)

	var (
		dev   = requireMetalDevice(b)
		ctx   = context.Background()
		ids1  = []int64{42}
		ids29 = make([]int64, 29)
	)
	for i := range ids29 {
		ids29[i] = int64((i*4099 + 17) % int(vocab))
	}

	var (
		embed1  = requireTensor(b, dev, 1, width)
		embed29 = requireTensor(b, dev, 29, width)
		logits1 = requireTensor(b, dev, 1, vocab)
	)
	requireError(b, embed1.Upload(attentionValues(width, 301, 1)), nil)

	for _, dtype := range []DType{BFloat16, Q8_0, Q4_0} {
		var (
			weight = requireTensorWithDType(b, dev, dtype, vocab, width)
			raw    = make([]byte, weight.Bytes())
		)
		// Populate deterministic non-zero bytes so the benchmark exercises real data.
		for i := range raw {
			raw[i] = byte((i*31 + 7) & 0x7f)
		}
		requireError(b, weight.UploadBytes(raw), nil)
		var weightMB = float64(weight.Bytes()) / (1024 * 1024)

		b.Run(fmt.Sprintf("GatherRows_M1/%s", dtype), func(b *testing.B) {
			benchmarkScopeOp(b, dev, ctx, weightMB, func(scope *Scope) error {
				return scope.GatherRows(weight, ids1, embed1)
			})
		})
		b.Run(fmt.Sprintf("GatherRows_M29/%s", dtype), func(b *testing.B) {
			benchmarkScopeOp(b, dev, ctx, weightMB, func(scope *Scope) error {
				return scope.GatherRows(weight, ids29, embed29)
			})
		})
		b.Run(fmt.Sprintf("Classifier_M1/%s", dtype), func(b *testing.B) {
			benchmarkScopeOp(b, dev, ctx, weightMB, func(scope *Scope) error {
				return scope.MatMulRightTranspose(embed1, weight, logits1)
			})
		})
		b.Run(fmt.Sprintf("TiedGatherAndClassifier_M1/%s", dtype), func(b *testing.B) {
			benchmarkScopeOp(b, dev, ctx, weightMB, func(scope *Scope) error {
				if err := scope.GatherRows(weight, ids1, embed1); err != nil {
					return err
				}
				return scope.MatMulRightTranspose(embed1, weight, logits1)
			})
		})
	}
}

func benchmarkScopeOp(b *testing.B, dev *Device, ctx context.Context, weightMB float64, encode func(*Scope) error) {
	runOnce := func() (elapsed time.Duration) {
		var (
			start      = time.Now()
			scope, err = dev.NewScope()
		)
		requireError(b, err, nil)
		defer scope.Close()
		requireError(b, encode(scope), nil)
		requireError(b, scope.Submit(ctx), nil)
		requireError(b, scope.Wait(ctx), nil)
		requireError(b, scope.Close(), nil)
		return time.Since(start)
	}

	// Warm up pipeline compilation and GPU frequency scaling.
	for range 3 {
		_ = runOnce()
	}

	b.ReportAllocs()
	b.ResetTimer()
	var total time.Duration
	for range b.N {
		total += runOnce()
	}
	b.StopTimer()
	b.ReportMetric(float64(total.Nanoseconds())/float64(b.N)/1e6, "ms/op")
	b.ReportMetric(weightMB, "weight_MB")
}
