//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"testing"
)

func Test_MetalKVCacheSingleBufferFootprint(t *testing.T) {
	const (
		capacity int64 = 4096
		kvHeads  int64 = 8
		headDim  int64 = 64
	)
	var (
		device = requireMetalDevice(t)
		native = device.state.backend.(metalBackend).runtime
	)
	for _, dtype := range []DType{Float32, Float16} {
		t.Run(dtype.String(), func(t *testing.T) {
			var (
				before    = native.ResourceSnapshot()
				cache, e  = NewKVCacheWithDType(device, capacity, kvHeads, headDim, dtype)
				wantBytes = uint64(2*capacity*kvHeads*headDim) * dtype.Bytes()
			)
			requireError(t, e, nil)
			var afterAlloc = native.ResourceSnapshot()
			if afterAlloc.LiveBuffers-before.LiveBuffers != 1 {
				t.Fatalf("%s KV cache allocated %d buffers, want 1", dtype, afterAlloc.LiveBuffers-before.LiveBuffers)
			}
			if afterAlloc.LiveBufferBytes-before.LiveBufferBytes != wantBytes {
				t.Fatalf("%s KV cache allocated %d bytes, want %d", dtype, afterAlloc.LiveBufferBytes-before.LiveBufferBytes, wantBytes)
			}
			requireError(t, cache.Close(), nil)
			var afterClose = native.ResourceSnapshot()
			if afterClose.LiveBuffers != before.LiveBuffers || afterClose.LiveBufferBytes != before.LiveBufferBytes {
				t.Fatalf("%s KV cache leaked resources: before %+v after %+v", dtype, before, afterClose)
			}
		})
	}
}

func Test_MetalKVCacheLongContextParity(t *testing.T) {
	const (
		heads   int64 = 8
		kvHeads int64 = 2
		headDim int64 = 32
	)
	var (
		device = requireMetalDevice(t)
		native = device.state.backend.(metalBackend).runtime
	)
	for _, history := range []int64{2048, 4096} {
		for _, dtype := range []DType{Float32, Float16} {
			t.Run(fmt.Sprintf("H%d/%s", history, dtype), func(t *testing.T) {
				var baseline = native.ResourceSnapshot()
				t.Run("exec", func(t *testing.T) {
					var (
						prefillRows = history - 2
						cache, e    = NewKVCacheWithDType(device, history, kvHeads, headDim, dtype)
						kPre        = requireTensor(t, device, prefillRows, kvHeads, headDim)
						vPre        = requireTensor(t, device, prefillRows, kvHeads, headDim)
						kStep1      = requireTensor(t, device, 1, kvHeads, headDim)
						vStep1      = requireTensor(t, device, 1, kvHeads, headDim)
						kStep2      = requireTensor(t, device, 1, kvHeads, headDim)
						vStep2      = requireTensor(t, device, 1, kvHeads, headDim)
						query       = requireTensor(t, device, 1, heads, headDim)
						output      = requireTensor(t, device, 1, heads, headDim)
						rowWidth    = kvHeads * headDim
						kPreData    = attentionValues(prefillRows*rowWidth, uint64(history+101), 0.25)
						vPreData    = attentionValues(prefillRows*rowWidth, uint64(history+102), 0.25)
						kStep1Data  = attentionValues(rowWidth, uint64(history+103), 0.25)
						vStep1Data  = attentionValues(rowWidth, uint64(history+104), 0.25)
						kStep2Data  = attentionValues(rowWidth, uint64(history+105), 0.25)
						vStep2Data  = attentionValues(rowWidth, uint64(history+106), 0.25)
						qData       = attentionValues(heads*headDim, uint64(history+107), 0.25)
					)
					requireError(t, e, nil)
					t.Cleanup(func() { requireError(t, cache.Close(), nil) })

					requireError(t, kPre.Upload(kPreData), nil)
					requireError(t, vPre.Upload(vPreData), nil)
					requireError(t, kStep1.Upload(kStep1Data), nil)
					requireError(t, vStep1.Upload(vStep1Data), nil)
					requireError(t, kStep2.Upload(kStep2Data), nil)
					requireError(t, vStep2.Upload(vStep2Data), nil)
					requireError(t, query.Upload(qData), nil)

					// Prefill [0..history-2).
					var s0 = requireScope(t, device)
					var u0 = stageKV(t, s0, cache, kPre, vPre, 0)
					completeAttentionScope(t, s0)
					requireError(t, u0.Publish(), nil)

					// Incremental append at position history-2.
					var s1 = requireScope(t, device)
					var u1 = stageKV(t, s1, cache, kStep1, vStep1, prefillRows)
					completeAttentionScope(t, s1)
					requireError(t, u1.Publish(), nil)

					// Incremental append at position history-1 + CausalAttention in one scope.
					var s2 = requireScope(t, device)
					var u2 = stageKV(t, s2, cache, kStep2, vStep2, prefillRows+1)
					requireError(t, s2.CausalAttention(query, cache, output, history-1), nil)
					completeAttentionScope(t, s2)
					requireError(t, u2.Publish(), nil)

					if cache.Len() != history {
						t.Fatalf("cache len = %d, want %d", cache.Len(), history)
					}

					var got = make([]float32, len(qData))
					requireError(t, output.DownloadInto(got), nil)

					var (
						kAll = slices.Concat(
							dequantizeForRef(kPreData, dtype),
							dequantizeForRef(kStep1Data, dtype),
							dequantizeForRef(kStep2Data, dtype),
						)
						vAll = slices.Concat(
							dequantizeForRef(vPreData, dtype),
							dequantizeForRef(vStep1Data, dtype),
							dequantizeForRef(vStep2Data, dtype),
						)
						want                 = referenceAttention(testFP64(qData), testFP64(kAll), testFP64(vAll), heads, kvHeads, headDim, history-1)
						p50, p95, p99, maxEr = fp64ErrorStats(got, want)
					)
					if maxEr > 1e-5 {
						t.Fatalf("H=%d %s max FP64 error = %g (p50=%g p95=%g p99=%g)", history, dtype, maxEr, p50, p95, p99)
					}
					t.Logf("Long-context KV H=%d %s FP64 abs err: p50=%g p95=%g p99=%g max=%g", history, dtype, p50, p95, p99, maxEr)
				})
				var after = native.ResourceSnapshot()
				if after.LiveBuffers != baseline.LiveBuffers || after.LiveBufferBytes != baseline.LiveBufferBytes || after.LiveScopes != baseline.LiveScopes {
					t.Fatalf("H=%d %s leaked resources: before %+v after %+v", history, dtype, baseline, after)
				}
			})
		}
	}
}

func Test_MetalKVCacheConcurrentStress(t *testing.T) {
	const (
		workers    = 8
		iterations = 8
		capacity   = 16
		heads      = 4
		kvHeads    = 2
		headDim    = 16
	)
	var (
		device   = requireMetalDevice(t)
		native   = device.state.backend.(metalBackend).runtime
		baseline = native.ResourceSnapshot()
		ctx      = context.Background()
	)
	for _, dtype := range []DType{Float32, Float16} {
		t.Run(dtype.String(), func(t *testing.T) {
			var (
				wg   sync.WaitGroup
				errs = make(chan error, workers)
			)
			for worker := range workers {
				wg.Add(1)
				go func(id int) {
					defer wg.Done()
					var (
						cache    *KVCache
						kPre     *Tensor
						vPre     *Tensor
						kDec     *Tensor
						vDec     *Tensor
						query    *Tensor
						output   *Tensor
						err      error
						rowWidth = int64(kvHeads * headDim)
					)
					if cache, err = NewKVCacheWithDType(device, capacity, kvHeads, headDim, dtype); err != nil {
						errs <- fmt.Errorf("worker %d cache: %w", id, err)
						return
					}
					defer func() { _ = cache.Close() }()
					if kPre, err = device.NewFloat32([]int64{4, kvHeads, headDim}); err != nil {
						errs <- err
						return
					}
					defer func() { _ = kPre.Close() }()
					if vPre, err = device.NewFloat32([]int64{4, kvHeads, headDim}); err != nil {
						errs <- err
						return
					}
					defer func() { _ = vPre.Close() }()
					if kDec, err = device.NewFloat32([]int64{1, kvHeads, headDim}); err != nil {
						errs <- err
						return
					}
					defer func() { _ = kDec.Close() }()
					if vDec, err = device.NewFloat32([]int64{1, kvHeads, headDim}); err != nil {
						errs <- err
						return
					}
					defer func() { _ = vDec.Close() }()
					if query, err = device.NewFloat32([]int64{1, heads, headDim}); err != nil {
						errs <- err
						return
					}
					defer func() { _ = query.Close() }()
					if output, err = device.NewFloat32([]int64{1, heads, headDim}); err != nil {
						errs <- err
						return
					}
					defer func() { _ = output.Close() }()

					var (
						kPreData = attentionValues(4*rowWidth, uint64(id*101+1), 0.5)
						vPreData = attentionValues(4*rowWidth, uint64(id*101+2), 0.5)
						kDecData = attentionValues(rowWidth, uint64(id*101+3), 0.5)
						vDecData = attentionValues(rowWidth, uint64(id*101+4), 0.5)
						qData    = attentionValues(int64(heads*headDim), uint64(id*101+5), 0.5)
						kAll     = slices.Concat(dequantizeForRef(kPreData, dtype), dequantizeForRef(kDecData, dtype))
						vAll     = slices.Concat(dequantizeForRef(vPreData, dtype), dequantizeForRef(vDecData, dtype))
						want     = referenceAttention(testFP64(qData), testFP64(kAll), testFP64(vAll), heads, kvHeads, headDim, 4)
						got      = make([]float32, len(qData))
					)
					if err = kPre.Upload(kPreData); err != nil {
						errs <- err
						return
					}
					if err = vPre.Upload(vPreData); err != nil {
						errs <- err
						return
					}
					if err = kDec.Upload(kDecData); err != nil {
						errs <- err
						return
					}
					if err = vDec.Upload(vDecData); err != nil {
						errs <- err
						return
					}
					if err = query.Upload(qData); err != nil {
						errs <- err
						return
					}

					for iter := range iterations {
						// Prefill 4 rows.
						var s0 *Scope
						if s0, err = device.NewScope(); err != nil {
							errs <- err
							return
						}
						var u0 *KVAppend
						if u0, err = s0.AppendKV(cache, kPre, vPre, 0); err != nil {
							_ = s0.Close()
							errs <- err
							return
						}
						if err = s0.Submit(ctx); err != nil {
							_ = s0.Close()
							_ = u0.Close()
							errs <- err
							return
						}
						if err = s0.Wait(ctx); err != nil {
							_ = s0.Close()
							_ = u0.Close()
							errs <- err
							return
						}
						if err = s0.Close(); err != nil {
							_ = u0.Close()
							errs <- err
							return
						}
						if err = u0.Publish(); err != nil {
							errs <- err
							return
						}

						// Abandoned append at position 4 (discarded before Publish).
						var abandon *Scope
						if abandon, err = device.NewScope(); err != nil {
							errs <- err
							return
						}
						var uAbandon *KVAppend
						if uAbandon, err = abandon.AppendKV(cache, kDec, vDec, 4); err != nil {
							_ = abandon.Close()
							errs <- err
							return
						}
						if err = abandon.Submit(ctx); err != nil {
							_ = abandon.Close()
							_ = uAbandon.Close()
							errs <- err
							return
						}
						if err = abandon.Wait(ctx); err != nil {
							_ = abandon.Close()
							_ = uAbandon.Close()
							errs <- err
							return
						}
						if err = abandon.Close(); err != nil {
							_ = uAbandon.Close()
							errs <- err
							return
						}
						if err = uAbandon.Close(); err != nil {
							errs <- err
							return
						}

						// Retry append at position 4 + CausalAttention.
						var s1 *Scope
						if s1, err = device.NewScope(); err != nil {
							errs <- err
							return
						}
						var u1 *KVAppend
						if u1, err = s1.AppendKV(cache, kDec, vDec, 4); err != nil {
							_ = s1.Close()
							errs <- err
							return
						}
						if err = s1.CausalAttention(query, cache, output, 4); err != nil {
							_ = s1.Close()
							_ = u1.Close()
							errs <- err
							return
						}
						if err = s1.Submit(ctx); err != nil {
							_ = s1.Close()
							_ = u1.Close()
							errs <- err
							return
						}
						if err = s1.Wait(ctx); err != nil {
							_ = s1.Close()
							_ = u1.Close()
							errs <- err
							return
						}
						if err = s1.Close(); err != nil {
							_ = u1.Close()
							errs <- err
							return
						}
						if err = u1.Publish(); err != nil {
							errs <- err
							return
						}
						if err = output.DownloadInto(got); err != nil {
							errs <- err
							return
						}
						var _, _, _, maxErr = fp64ErrorStats(got, want)
						if maxErr > 1e-5 {
							errs <- fmt.Errorf("worker %d iter %d %s max err %g > 1e-5", id, iter, dtype, maxErr)
							return
						}
						if err = cache.Reset(); err != nil {
							errs <- fmt.Errorf("worker %d iter %d reset: %w", id, iter, err)
							return
						}
					}
				}(worker)
			}
			wg.Wait()
			close(errs)
			for err := range errs {
				t.Fatal(err)
			}
		})
	}
	var after = native.ResourceSnapshot()
	if after.LiveBuffers != baseline.LiveBuffers || after.LiveBufferBytes != baseline.LiveBufferBytes || after.LiveScopes != baseline.LiveScopes {
		t.Fatalf("concurrent KV stress leaked resources: before %+v after %+v", baseline, after)
	}
}
