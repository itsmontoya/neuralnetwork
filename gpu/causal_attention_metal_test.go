//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"math"
	"sync"
	"testing"
)

func Test_MetalCausalAttentionParallelShapesAndWidths(t *testing.T) {
	type shapeCase struct {
		name     string
		rows     int64
		heads    int64
		kvHeads  int64
		headDim  int64
		history  int64
		position int64
	}

	var cases = []shapeCase{
		// Odd/small and >128 headDim coverage across MHA/GQA/MQA.
		{name: "MHA_D1_H7", rows: 1, heads: 4, kvHeads: 4, headDim: 1, history: 7, position: 6},
		{name: "MHA_D3_H32", rows: 1, heads: 4, kvHeads: 4, headDim: 3, history: 32, position: 31},
		{name: "GQA_D17_H33", rows: 6, heads: 6, kvHeads: 2, headDim: 17, history: 33, position: 27},
		{name: "MQA_D32_H38", rows: 1, heads: 8, kvHeads: 1, headDim: 32, history: 38, position: 37},
		{name: "GQA_D33_H127", rows: 1, heads: 6, kvHeads: 2, headDim: 33, history: 127, position: 126},
		{name: "GQA_32to8_D64_H1", rows: 1, heads: 32, kvHeads: 8, headDim: 64, history: 1, position: 0},
		{name: "GQA_32to8_D64_H2", rows: 1, heads: 32, kvHeads: 8, headDim: 64, history: 2, position: 1},
		{name: "GQA_32to8_D64_H38", rows: 1, heads: 32, kvHeads: 8, headDim: 64, history: 38, position: 37},
		{name: "GQA_32to8_D64_H128", rows: 1, heads: 32, kvHeads: 8, headDim: 64, history: 128, position: 127},
		{name: "GQA_32to8_D64_H129", rows: 1, heads: 32, kvHeads: 8, headDim: 64, history: 129, position: 128},
		{name: "GQA_32to8_D64_H256", rows: 1, heads: 32, kvHeads: 8, headDim: 64, history: 256, position: 255},
		{name: "GQA_32to8_D64_H512", rows: 1, heads: 32, kvHeads: 8, headDim: 64, history: 512, position: 511},
		{name: "GQA_32to8_D64_H1024", rows: 1, heads: 32, kvHeads: 8, headDim: 64, history: 1024, position: 1023},
		{name: "GQA_D65_H256", rows: 1, heads: 6, kvHeads: 2, headDim: 65, history: 256, position: 255},
		{name: "GQA_D128_H128", rows: 1, heads: 8, kvHeads: 2, headDim: 128, history: 128, position: 127},
		{name: "GQA_D128_H512", rows: 1, heads: 8, kvHeads: 2, headDim: 128, history: 512, position: 511},
		{name: "GQA_D129_H129", rows: 1, heads: 6, kvHeads: 2, headDim: 129, history: 129, position: 128},
		{name: "MHA_D256_H128", rows: 1, heads: 4, kvHeads: 4, headDim: 256, history: 128, position: 127},
		// Multi-row prefill shapes (M = 6, 27, 28, 29), including partial-stripe queryRow=0 when maxVisible >= 64.
		{name: "Prefill_M6_H6_D64", rows: 6, heads: 32, kvHeads: 8, headDim: 64, history: 6, position: 0},
		{name: "Prefill_M27_H27_D64", rows: 27, heads: 6, kvHeads: 2, headDim: 64, history: 27, position: 0},
		{name: "Prefill_M28_H28_D128", rows: 28, heads: 8, kvHeads: 2, headDim: 128, history: 28, position: 0},
		{name: "Prefill_M29_H29_D64", rows: 29, heads: 32, kvHeads: 8, headDim: 64, history: 29, position: 0},
		{name: "Prefill_M29_H129_D64_Pos0", rows: 29, heads: 8, kvHeads: 2, headDim: 64, history: 129, position: 0},
		{name: "Prefill_M65_H65_D64_PartialStripe", rows: 65, heads: 4, kvHeads: 2, headDim: 64, history: 65, position: 0},
	}

	var (
		device   = requireMetalDevice(t)
		native   = device.state.backend.(metalBackend).runtime
		baseline = native.ResourceSnapshot()
	)

	for _, tc := range cases {
		for _, kvDType := range []DType{Float32, Float16} {
			t.Run(fmt.Sprintf("%s/%s", tc.name, kvDType), func(t *testing.T) {
				var (
					queries    = requireTensor(t, device, tc.rows, tc.heads, tc.headDim)
					output     = requireTensor(t, device, tc.rows, tc.heads, tc.headDim)
					keys       = requireTensor(t, device, tc.history, tc.kvHeads, tc.headDim)
					values     = requireTensor(t, device, tc.history, tc.kvHeads, tc.headDim)
					cache, err = NewKVCacheWithDType(device, tc.history, tc.kvHeads, tc.headDim, kvDType)
				)
				requireError(t, err, nil)
				t.Cleanup(func() { requireError(t, cache.Close(), nil) })

				var (
					qData = attentionValues(tc.rows*tc.heads*tc.headDim, uint64(tc.history*31+tc.headDim*7+tc.rows), 0.5)
					kData = attentionValues(tc.history*tc.kvHeads*tc.headDim, uint64(tc.history*37+tc.headDim*11+tc.heads), 0.5)
					vData = attentionValues(tc.history*tc.kvHeads*tc.headDim, uint64(tc.history*41+tc.headDim*13+tc.kvHeads), 0.5)
				)
				requireError(t, queries.Upload(qData), nil)
				requireError(t, keys.Upload(kData), nil)
				requireError(t, values.Upload(vData), nil)

				var (
					seedScope = requireScope(t, device)
					update    = stageKV(t, seedScope, cache, keys, values, 0)
				)
				completeAttentionScope(t, seedScope)
				requireError(t, update.Publish(), nil)

				var attnScope = requireScope(t, device)
				requireError(t, attnScope.CausalAttention(queries, cache, output, tc.position), nil)
				completeAttentionScope(t, attnScope)

				var got = make([]float32, len(qData))
				requireError(t, output.DownloadInto(got), nil)

				var (
					effK             = dequantizeForRef(kData, kvDType)
					effV             = dequantizeForRef(vData, kvDType)
					want             = referenceAttention(testFP64(qData), testFP64(effK), testFP64(effV), tc.heads, tc.kvHeads, tc.headDim, tc.position)
					p50, p95, p99, m = fp64ErrorStats(got, want)
				)
				if m > 1e-5 {
					t.Fatalf("%s/%s max abs err = %g (p50=%g p95=%g p99=%g)", tc.name, kvDType, m, p50, p95, p99)
				}
				if tc.heads == 32 && tc.headDim == 64 && (tc.history == 38 || tc.history == 128 || tc.history == 512 || tc.history == 1024 || tc.rows == 29) {
					t.Logf("%s/%s FP64 abs err: p50=%g p95=%g p99=%g max=%g", tc.name, kvDType, p50, p95, p99, m)
				}
			})
		}
	}

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != baseline.LiveBuffers || after.LiveBufferBytes != baseline.LiveBufferBytes || after.LiveScopes != baseline.LiveScopes {
		t.Fatalf("parallel causal attention leaked resources: before %+v after %+v", baseline, after)
	}
}

func Test_MetalCausalAttentionOverflowGuardAndEmptyCache(t *testing.T) {
	var (
		device = requireMetalDevice(t)
		shapes = []struct {
			name    string
			history int64
			headDim int64
		}{
			{name: "single_simd_H7_D64", history: 7, headDim: 64},
			{name: "multi_simd_4_H128_D64", history: 128, headDim: 64},
			{name: "multi_simd_8_H512_D64", history: 512, headDim: 64},
			{name: "fallback_H70_D256", history: 70, headDim: 256},
		}
	)

	for _, tc := range shapes {
		for _, kvDType := range []DType{Float32, Float16} {
			t.Run(fmt.Sprintf("%s/%s", tc.name, kvDType), func(t *testing.T) {
				const (
					heads   int64 = 4
					kvHeads int64 = 2
				)
				var (
					queries    = requireTensor(t, device, 1, heads, tc.headDim)
					output     = requireTensor(t, device, 1, heads, tc.headDim)
					keys       = requireTensor(t, device, tc.history, kvHeads, tc.headDim)
					values     = requireTensor(t, device, tc.history, kvHeads, tc.headDim)
					cache, err = NewKVCacheWithDType(device, tc.history, kvHeads, tc.headDim, kvDType)
				)
				requireError(t, err, nil)
				t.Cleanup(func() { requireError(t, cache.Close(), nil) })

				// Empty cache yields exact zeros.
				var (
					qData      = make([]float32, heads*tc.headDim)
					kData      = make([]float32, tc.history*kvHeads*tc.headDim)
					vData      = attentionValues(tc.history*kvHeads*tc.headDim, 901, 2.0)
					emptyScope = requireScope(t, device)
					got        = make([]float32, len(qData))
				)
				for h := range heads {
					qData[h*tc.headDim] = 200
				}
				// Place extreme opposing scores across different SIMD stripes.
				for row := range tc.history {
					for kv := range kvHeads {
						var val float32 = -50
						if row == tc.history-1 {
							val = 150
						} else if row == 0 {
							val = 149
						}
						kData[(row*kvHeads+kv)*tc.headDim] = val
					}
				}
				requireError(t, queries.Upload(qData), nil)
				requireError(t, emptyScope.CausalAttention(queries, cache, output, 0), nil)
				completeAttentionScope(t, emptyScope)
				requireError(t, output.DownloadInto(got), nil)
				compareAttention(t, got, make([]float64, len(got)), 0)

				// Populate and run large-magnitude softmax overflow guard.
				requireError(t, keys.Upload(kData), nil)
				requireError(t, values.Upload(vData), nil)
				var (
					seedScope = requireScope(t, device)
					update    = stageKV(t, seedScope, cache, keys, values, 0)
				)
				completeAttentionScope(t, seedScope)
				requireError(t, update.Publish(), nil)

				var attnScope = requireScope(t, device)
				requireError(t, attnScope.CausalAttention(queries, cache, output, tc.history-1), nil)
				completeAttentionScope(t, attnScope)
				requireError(t, output.DownloadInto(got), nil)

				for i, v := range got {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatalf("%s/%s output[%d] = %v (non-finite)", tc.name, kvDType, i, v)
					}
				}
				var (
					effK = dequantizeForRef(kData, kvDType)
					effV = dequantizeForRef(vData, kvDType)
					want = referenceAttention(testFP64(qData), testFP64(effK), testFP64(effV), heads, kvHeads, tc.headDim, tc.history-1)
				)
				compareAttention(t, got, want, 1e-5)
			})
		}
	}
}

func Test_MetalCausalAttentionConcurrentStress(t *testing.T) {
	var (
		device   = requireMetalDevice(t)
		native   = device.state.backend.(metalBackend).runtime
		baseline = native.ResourceSnapshot()
		configs  = []struct {
			name    string
			history int64
			kvDType DType
		}{
			{name: "H38_FP32", history: 38, kvDType: Float32},
			{name: "H128_FP16", history: 128, kvDType: Float16},
			{name: "H512_FP32", history: 512, kvDType: Float32},
			{name: "H512_FP16", history: 512, kvDType: Float16},
		}
	)

	const (
		heads      int64 = 32
		kvHeads    int64 = 8
		headDim    int64 = 64
		workers          = 8
		iterations       = 8
		dispatches       = 16
	)

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			var (
				cache, err = NewKVCacheWithDType(device, cfg.history, kvHeads, headDim, cfg.kvDType)
				kData      = attentionValues(cfg.history*kvHeads*headDim, uint64(cfg.history*19+int64(cfg.kvDType)), 0.5)
				vData      = attentionValues(cfg.history*kvHeads*headDim, uint64(cfg.history*23+int64(cfg.kvDType)), 0.5)
				effK       = dequantizeForRef(kData, cfg.kvDType)
				effV       = dequantizeForRef(vData, cfg.kvDType)
				keys       = requireTensor(t, device, cfg.history, kvHeads, headDim)
				values     = requireTensor(t, device, cfg.history, kvHeads, headDim)
				seedScope  = requireScope(t, device)
			)
			requireError(t, err, nil)
			requireError(t, keys.Upload(kData), nil)
			requireError(t, values.Upload(vData), nil)
			var update = stageKV(t, seedScope, cache, keys, values, 0)
			completeAttentionScope(t, seedScope)
			requireError(t, update.Publish(), nil)
			requireError(t, keys.Close(), nil)
			requireError(t, values.Close(), nil)

			var (
				wg   sync.WaitGroup
				errs = make(chan error, workers)
			)
			for worker := range workers {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()
					var (
						qData   = attentionValues(heads*headDim, uint64(id+1)*131+uint64(cfg.history), 0.5)
						want    = referenceAttention(testFP64(qData), testFP64(effK), testFP64(effV), heads, kvHeads, headDim, cfg.history-1)
						got     = make([]float32, len(qData))
						queries *Tensor
						output  *Tensor
						wErr    error
					)
					if queries, wErr = device.NewFloat32([]int64{1, heads, headDim}); wErr != nil {
						errs <- fmt.Errorf("worker %d alloc queries: %w", id, wErr)
						return
					}
					defer func() { _ = queries.Close() }()
					if output, wErr = device.NewFloat32([]int64{1, heads, headDim}); wErr != nil {
						errs <- fmt.Errorf("worker %d alloc output: %w", id, wErr)
						return
					}
					defer func() { _ = output.Close() }()
					if wErr = queries.Upload(qData); wErr != nil {
						errs <- fmt.Errorf("worker %d upload: %w", id, wErr)
						return
					}
					for iter := range iterations {
						var scope *Scope
						if scope, wErr = device.NewScope(); wErr != nil {
							errs <- fmt.Errorf("worker %d iter %d scope: %w", id, iter, wErr)
							return
						}
						for range dispatches {
							if wErr = scope.CausalAttention(queries, cache, output, cfg.history-1); wErr != nil {
								_ = scope.Close()
								errs <- fmt.Errorf("worker %d iter %d encode: %w", id, iter, wErr)
								return
							}
						}
						if wErr = scope.Submit(context.Background()); wErr != nil {
							_ = scope.Close()
							errs <- fmt.Errorf("worker %d iter %d submit: %w", id, iter, wErr)
							return
						}
						if wErr = scope.Wait(context.Background()); wErr != nil {
							_ = scope.Close()
							errs <- fmt.Errorf("worker %d iter %d wait: %w", id, iter, wErr)
							return
						}
						if wErr = scope.Close(); wErr != nil {
							errs <- fmt.Errorf("worker %d iter %d close: %w", id, iter, wErr)
							return
						}
						if wErr = output.DownloadInto(got); wErr != nil {
							errs <- fmt.Errorf("worker %d iter %d download: %w", id, iter, wErr)
							return
						}
						var _, _, _, maxErr = fp64ErrorStats(got, want)
						if maxErr > 1e-5 {
							errs <- fmt.Errorf("worker %d iter %d max FP64 error %g exceeds 1e-5", id, iter, maxErr)
							return
						}
					}
				}(worker)
			}
			wg.Wait()
			close(errs)
			for wErr := range errs {
				t.Fatal(wErr)
			}
			requireError(t, cache.Close(), nil)
		})
	}

	var after = native.ResourceSnapshot()
	if after.LiveBuffers != baseline.LiveBuffers || after.LiveBufferBytes != baseline.LiveBufferBytes || after.LiveScopes != baseline.LiveScopes {
		t.Fatalf("concurrent causal attention stress leaked resources: before %+v after %+v", baseline, after)
	}
}
