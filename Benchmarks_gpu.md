# GPU Benchmark Log

This file records benchmark evidence for the GPU acceleration branch. Every GPU
implementation session should append a dated section with exact commands, raw
output, and a short interpretation.

## Logging Template

````md
## Step N: Title

Captured on Month Day, Year.

### Commands

```sh
go test ...
```

### Raw Output

```text
...
```

### Interpretation

Short comparison of before and after results, including whether the step should
continue, change direction, or stop.
````

## Baseline: CPU Candidate Benchmarks

Captured on July 8, 2026.

### Environment

| Field | Value |
| --- | --- |
| OS | Darwin 23.5.0 |
| Architecture | arm64 |
| CPU | Apple M3 |
| go.mod Go version | 1.26.1 |
| Go toolchain | go1.26.5 darwin/arm64 |

### Commands

Focused GPU-candidate baseline:

```sh
go test ./matrix ./layer ./model -run '^$' -bench='(MatMul|Dense|SequentialTrainBatch_SyntheticDense|SequentialFit_SyntheticDense)' -benchmem
```

Comprehensive baseline also completed successfully:

```sh
go test ./matrix ./layer ./model ./loss ./optimizer ./data -run '^$' -bench=. -benchmem
```

The focused command is the primary comparison point for early GPU work because
it captures matrix multiplication, dense layer, and synthetic dense model paths.

### Summary

| Package | Benchmark | ns/op | B/op | allocs/op |
| --- | --- | ---: | ---: | ---: |
| `matrix` | `Benchmark_MatMul-8` | 150019 | 32816 | 2 |
| `matrix` | `Benchmark_MatMulInto-8` | 149942 | 0 | 0 |
| `matrix` | `Benchmark_MatMulLeftTransposeInto-8` | 149657 | 0 | 0 |
| `matrix` | `Benchmark_MatMulRightTransposeInto-8` | 161310 | 0 | 0 |
| `matrix` | `Benchmark_MatMulShapes/Large128x256x128-8` | 2488939 | 131120 | 2 |
| `matrix` | `Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8` | 3365983 | 0 | 0 |
| `layer` | `Benchmark_DenseForward_XOR-8` | 90.14 | 0 | 0 |
| `layer` | `Benchmark_DenseForward_MediumBatch-8` | 163792 | 0 | 0 |
| `layer` | `Benchmark_DenseBackward_XOR-8` | 147.1 | 0 | 0 |
| `layer` | `Benchmark_DenseBackward_MediumBatch-8` | 313680 | 0 | 0 |
| `model` | `Benchmark_SequentialTrainBatch_SyntheticDense-8` | 770740 | 147965 | 10 |
| `model` | `Benchmark_SequentialFit_SyntheticDense-8` | 1062330 | 634014 | 93 |

### Raw Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMulRightTransposeDotCandidate/Small2x2-8         	98413033	        12.04 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small4x4-8         	20434154	        59.34 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Medium64x64-8      	    7693	    155684 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Large128x256x128-8 	     332	   3601248 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven17x33x19-8   	  237910	      5036 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven63x65x31-8   	   15933	     75342 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMul-8                                            	    8005	    150019 ns/op	   32816 B/op	       2 allocs/op
Benchmark_MatMulInto-8                                        	    8054	    149942 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulLeftTransposeInto-8                           	    7912	    149657 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeInto-8                          	    7436	    161310 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulShapes/Small2x2-8                             	27938271	        42.07 ns/op	      80 B/op	       2 allocs/op
Benchmark_MatMulShapes/Small4x4-8                             	11491784	       103.8 ns/op	     176 B/op	       2 allocs/op
Benchmark_MatMulShapes/Medium64x64-8                          	    7947	    151077 ns/op	   32816 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8                     	     480	   2488939 ns/op	  131120 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven17x33x19-8                       	  171900	      6964 ns/op	    2736 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven63x65x31-8                       	   15476	     77619 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small2x2-8           	55752475	        21.31 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small4x4-8           	14820997	        80.54 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Medium64x64-8        	    7646	    156824 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8   	     354	   3365983 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven17x33x19-8     	  174900	      6863 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven63x65x31-8     	   15339	     77823 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	31.102s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/layer
cpu: Apple M3
Benchmark_DenseForward_XOR-8            	11521470	        90.14 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_MediumBatch-8    	    7064	    163792 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_XOR-8           	 8076812	       147.1 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_MediumBatch-8   	    3817	    313680 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/layer	6.053s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialTrainBatch_SyntheticDense-8   	    1315	    770740 ns/op	  147965 B/op	      10 allocs/op
Benchmark_SequentialFit_SyntheticDense-8          	    1123	   1062330 ns/op	  634014 B/op	      93 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/model	2.578s
```

### Interpretation

The relevant CPU paths are already allocation-efficient. Early GPU work should
focus on wall-clock wins for large matrix multiplication and dense training
shapes. Small shapes are too fast on CPU to justify WebGPU dispatch unless a
later design keeps data resident and fuses multiple operations.

## Step 1 Checklist Item 1: Focused CPU Baseline Rerun

Captured on July 8, 2026.

### Environment

| Field | Value |
| --- | --- |
| OS | Darwin 23.5.0 |
| Architecture | arm64 |
| CPU | Apple M3 |
| go.mod Go version | 1.26.1 |
| Go toolchain | go1.26.5 darwin/arm64 |

### Commands

```sh
go test ./matrix ./layer ./model -run '^$' -bench='(MatMul|Dense|SequentialTrainBatch_SyntheticDense|SequentialFit_SyntheticDense)' -benchmem
```

### Summary

| Package | Benchmark | ns/op | B/op | allocs/op |
| --- | --- | ---: | ---: | ---: |
| `matrix` | `Benchmark_MatMul-8` | 153864 | 32816 | 2 |
| `matrix` | `Benchmark_MatMulInto-8` | 148775 | 0 | 0 |
| `matrix` | `Benchmark_MatMulLeftTransposeInto-8` | 148896 | 0 | 0 |
| `matrix` | `Benchmark_MatMulRightTransposeInto-8` | 160876 | 0 | 0 |
| `matrix` | `Benchmark_MatMulShapes/Large128x256x128-8` | 2485526 | 131120 | 2 |
| `matrix` | `Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8` | 3388818 | 0 | 0 |
| `layer` | `Benchmark_DenseForward_XOR-8` | 93.55 | 0 | 0 |
| `layer` | `Benchmark_DenseForward_MediumBatch-8` | 167625 | 0 | 0 |
| `layer` | `Benchmark_DenseBackward_XOR-8` | 145.8 | 0 | 0 |
| `layer` | `Benchmark_DenseBackward_MediumBatch-8` | 314956 | 0 | 0 |
| `model` | `Benchmark_SequentialTrainBatch_SyntheticDense-8` | 773359 | 147961 | 10 |
| `model` | `Benchmark_SequentialFit_SyntheticDense-8` | 1059163 | 634013 | 93 |

### Raw Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMulRightTransposeDotCandidate/Small2x2-8         	99316678	        11.84 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small4x4-8         	19874596	        59.04 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Medium64x64-8      	    7719	    156294 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Large128x256x128-8 	     332	   3604595 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven17x33x19-8   	  232849	      5053 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven63x65x31-8   	   15838	     75508 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMul-8                                            	    8083	    153864 ns/op	   32816 B/op	       2 allocs/op
Benchmark_MatMulInto-8                                        	    8036	    148775 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulLeftTransposeInto-8                           	    8173	    148896 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeInto-8                          	    7473	    160876 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulShapes/Small2x2-8                             	28456276	        40.82 ns/op	      80 B/op	       2 allocs/op
Benchmark_MatMulShapes/Small4x4-8                             	11840595	       101.5 ns/op	     176 B/op	       2 allocs/op
Benchmark_MatMulShapes/Medium64x64-8                          	    8017	    150808 ns/op	   32816 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8                     	     482	   2485526 ns/op	  131120 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven17x33x19-8                       	  171794	      6956 ns/op	    2736 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven63x65x31-8                       	   15535	     77114 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small2x2-8           	55083456	        21.21 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small4x4-8           	14727524	        80.70 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Medium64x64-8        	    7628	    156900 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8   	     352	   3388818 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven17x33x19-8     	  172966	      6888 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven63x65x31-8     	   15422	     77654 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	31.072s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/layer
cpu: Apple M3
Benchmark_DenseForward_XOR-8            	13318743	        93.55 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_MediumBatch-8    	    7113	    167625 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_XOR-8           	 8117529	       145.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_MediumBatch-8   	    3804	    314956 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/layer	6.348s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialTrainBatch_SyntheticDense-8   	    1314	    773359 ns/op	  147961 B/op	      10 allocs/op
Benchmark_SequentialFit_SyntheticDense-8          	    1126	   1059163 ns/op	  634013 B/op	      93 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/model	2.581s
```

### Interpretation

The rerun remains consistent with the existing CPU candidate baseline. Matrix
destination variants and dense layer paths remain at zero steady-state
allocations, so the GPU experiment still needs to win on wall-clock time rather
than allocation reduction.

## Step 2: Direct Metal Matrix Multiplication

Captured on July 11, 2026.

### Environment

| Field | Value |
| --- | --- |
| OS | Darwin |
| Architecture | arm64 |
| CPU | Apple M3 |

### Commands

Correctness and build-tag checks:

```sh
go test ./...
go test ./... -tags=metal
go test ./matrix -tags=purego
go test ./matrix -tags='metal purego'
```

Focused benchmark samples:

```sh
go test ./matrix -tags=metal -run '^$' -bench='MatMul$|MatMulInto$|MatMulShapes/Large128x256x128$' -benchmem -benchtime=100ms
go test ./matrix -tags=metal -run '^$' -bench='DotProduct/Length(4096|65537)$' -benchmem -benchtime=100ms
go test ./matrix -tags=metal -run '^$' -bench='ElementwiseCandidates/(Large1024x1024)/(AddInto|MultiplyElementsInto|MultiplyScalarInPlace)/Active$' -benchmem -benchtime=100ms
go test ./matrix -run '^$' -bench='DotProduct/Length(4096|65537)$' -benchmem -benchtime=100ms
go test ./matrix -run '^$' -bench='MatMul$|MatMulInto$|MatMulShapes/Large128x256x128$' -benchmem -benchtime=100ms
go test ./matrix -run '^$' -bench='ElementwiseCandidates/(Medium256x256|Large1024x1024)/(AddInto|MultiplyElementsInto|MultiplyScalarInPlace)/Active$' -benchmem -benchtime=100ms
```

### Summary

| Benchmark | CPU/SIMD ns/op | `metal` ns/op | Interpretation |
| --- | ---: | ---: | --- |
| `Benchmark_MatMul-8` | 156478 | 169413 | Below Metal threshold; CPU path remains active. |
| `Benchmark_MatMulInto-8` | 154639 | 152871 | Below Metal threshold; CPU path remains active. |
| `Benchmark_MatMulShapes/Large128x256x128-8` | 2715264 | 460492 | Large shape uses Metal and is materially faster. |
| `Benchmark_DotProduct/Length4096-8` | 1130 | 4190 | `metal` build excludes SIMD; scalar fallback is slower than SIMD. |
| `Benchmark_DotProduct/Length65537-8` | 18587 | 72261 | `metal` build excludes SIMD; scalar fallback is slower than SIMD. |
| `Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8` | 233058 | 435876 | `metal` build uses scalar fallback for slice kernels. |
| `Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8` | 229687 | 440326 | `metal` build uses scalar fallback for slice kernels. |
| `Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8` | 140716 | 278138 | `metal` build uses scalar fallback for slice kernels. |

### Raw Output

Metal matrix multiplication:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMul-8         	     802	    169413 ns/op	   32823 B/op	       2 allocs/op
Benchmark_MatMulInto-8     	     742	    152871 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8         	     243	    460492 ns/op	  131120 B/op	       2 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	0.890s
```

Metal-tagged dot product scalar fallback:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_DotProduct/Length4096-8         	   25279	      4190 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8        	    1664	     72261 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	0.475s
```

Metal-tagged elementwise scalar fallback:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8         	     266	    435876 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8         	     273	    440326 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8        	     420	    278138 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	0.944s
```

CPU/SIMD comparison:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_DotProduct/Length4096-8         	  106303	      1130 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8        	    6505	     18587 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	0.790s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMul-8         	     778	    156478 ns/op	   32816 B/op	       2 allocs/op
Benchmark_MatMulInto-8     	     770	    154639 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8         	      44	   2715264 ns/op	  131120 B/op	       2 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	0.798s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Active-8         	   10544	     14148 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Active-8         	    8056	     13206 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Active-8        	   13231	      9576 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8                     	     476	    233058 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8        	     466	    229687 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8       	     847	    140716 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	1.207s
```

### Interpretation

Direct Metal access is useful for large matrix multiplication even with
`float64` to `float` boundary conversion. The sampled large
`128x256 * 256x128` shape improved from 2.715 ms/op to 0.460 ms/op.

Metal did not help dot product or elementwise kernels in this API shape. Those
operations are O(n), while the bridge must still convert and copy every value.
The `metal` build therefore excludes SIMD but uses scalar fallbacks for slice
kernels rather than dispatching them to GPU.

Because Metal rejected `double` in shader compilation, this implementation is an
explicit lower-precision path for large matrix multiplication. Public APIs still
use `float64`, but Metal compute is internally `float`.

## Step 3: Float32 Matrix Storage and Direct Metal Buffers

Captured on July 12, 2026.

### Environment

| Field | Value |
| --- | --- |
| OS | Darwin |
| Architecture | arm64 |
| CPU | Apple M3 |

This section supersedes the Step 2 interpretation after converting matrix
storage and public numeric APIs to `float32`.

### Commands

Correctness and build-tag checks:

```sh
go test ./...
go test ./... -tags=metal
go test ./matrix -tags=purego
go test ./matrix -tags='metal purego'
```

Focused benchmark samples:

```sh
go test ./matrix -run '^$' -bench='MatMul.*Shapes/Large128x256x128$' -benchmem -benchtime=500ms -count=3
go test ./matrix -tags=metal -run '^$' -bench='MatMul.*Shapes/Large128x256x128$' -benchmem -benchtime=500ms -count=3
go test ./matrix -tags=purego -run '^$' -bench='MatMul.*Shapes/Large128x256x128$' -benchmem -benchtime=500ms -count=3
go test ./matrix -run '^$' -bench='DotProduct/(Length4096|Length65537)$' -benchmem -benchtime=500ms -count=3
go test ./matrix -tags=purego -run '^$' -bench='DotProduct/(Length4096|Length65537)$' -benchmem -benchtime=500ms -count=3
go test ./matrix -run '^$' -bench='ElementwiseCandidates/Large1024x1024/(AddInto|SubtractInto|MultiplyElementsInto|AddScalarInto|MultiplyScalarInto|MultiplyScalarInPlace)/(Pure|Active)$' -benchmem -benchtime=500ms -count=3
```

### Summary

| Benchmark | Pure Go avg ns/op | SIMD avg ns/op | Metal avg ns/op | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `Benchmark_MatMulShapes/Large128x256x128-8` | 2433889 | 2401331 | 250808 | Metal is 9.6x faster than SIMD. |
| `Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8` | 2981280 | 2946228 | 278272 | Metal is 10.6x faster than SIMD. |
| `Benchmark_DotProduct/Length4096-8` | 3533 | 470 | n/a | SIMD is 7.5x faster than pure Go. |
| `Benchmark_DotProduct/Length65537-8` | 56989 | 7551 | n/a | SIMD is 7.5x faster than pure Go. |
| `Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8` | 407525 | 111576 | n/a | SIMD is 3.7x faster than pure Go. |
| `Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Active-8` | 418781 | 111300 | n/a | SIMD is 3.8x faster than pure Go. |
| `Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8` | 420411 | 111877 | n/a | SIMD is 3.8x faster than pure Go. |
| `Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Active-8` | 275740 | 73613 | n/a | SIMD is 3.7x faster than pure Go. |
| `Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Active-8` | 277930 | 73716 | n/a | SIMD is 3.8x faster than pure Go. |
| `Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8` | 278138 | 70518 | n/a | SIMD is 3.9x faster than pure Go. |

### Raw Output

Default SIMD matrix multiplication:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMulShapes/Large128x256x128-8    	     237	   2390595 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8    	     249	   2399690 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8    	     247	   2413709 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8         	     198	   2950196 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8         	     204	   2946825 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8         	     204	   2941662 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	5.445s
```

Metal matrix multiplication:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMulShapes/Large128x256x128-8    	    2475	    248374 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8    	    2433	    252089 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8    	    2370	    251961 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8         	    2290	    277941 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8         	    2148	    279050 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8         	    2144	    277826 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	4.848s
```

Pure Go matrix multiplication:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMulShapes/Large128x256x128-8    	     230	   2407651 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8    	     247	   2444372 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8    	     246	   2449645 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8         	     201	   2978981 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8         	     199	   2977714 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8         	     196	   2987144 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	5.384s
```

Default SIMD dot product:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_DotProduct/Length4096-8         	 1164004	       470.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4096-8         	 1277492	       470.3 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4096-8         	 1277731	       469.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8        	   80709	      7478 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8        	   79885	      7603 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8        	   78700	      7573 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	5.211s
```

Pure Go dot product:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_DotProduct/Length4096-8         	  168363	      3531 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4096-8         	  169690	      3536 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4096-8         	  168987	      3533 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8        	   10000	     56949 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8        	   10000	     57041 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8        	   10000	     56977 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	4.217s
```

Elementwise Pure and Active SIMD candidates:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Pure-8         	    1514	    404354 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Pure-8         	    1479	    407924 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Pure-8         	    1476	    410298 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8       	    5624	    111632 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8       	    5559	    111523 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8       	    5565	    111573 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Pure-8    	    1437	    420004 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Pure-8    	    1432	    417711 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Pure-8    	    1461	    418627 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Active-8  	    5448	    112401 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Active-8  	    5360	    109965 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Active-8  	    5542	    111533 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Pure-8         	    1431	    418833 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Pure-8         	    1453	    421952 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Pure-8         	    1435	    420448 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8       	    5313	    111518 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8       	    5374	    111684 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8       	    5350	    112428 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Pure-8                	    2215	    277918 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Pure-8                	    2168	    273235 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Pure-8                	    2211	    276067 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Active-8              	    7833	     73503 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Active-8              	    8264	     73528 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Active-8              	    8282	     73809 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Pure-8           	    2169	    278295 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Pure-8           	    2139	    277972 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Pure-8           	    2216	    277522 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Active-8         	    7407	     73574 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Active-8         	    8142	     74079 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Active-8         	    8096	     73495 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Pure-8        	    2166	    278202 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Pure-8        	    2188	    277796 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Pure-8        	    2162	    278416 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8      	    9128	     70802 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8      	    8588	     70516 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8      	    8582	     70235 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	23.310s
```

### Interpretation

Converting matrix storage to `float32` removes the old Metal boundary conversion
cost. Large matrix multiplication now dispatches direct f32 buffers and improved
from the Step 2 Metal sample of 460492 ns/op to about 250808 ns/op for
`128x256 * 256x128`.

Metal is now the clear winner for large matrix multiplication on this shape:
about 9.6x faster than default SIMD for standard matmul and 10.6x faster for
right-transposed matmul. SIMD remains valuable for CPU-only O(n) kernels:
f32 dot product is about 7.5x faster than pure Go, and large elementwise kernels
are about 3.7x to 3.9x faster than pure Go.

## Step 4: Hybrid CPU and Metal Baselines

Captured on July 21, 2026.

### Environment

| Field | Value |
| --- | --- |
| macOS version | 26.5.2 |
| Darwin kernel | 25.5.0 |
| Architecture | arm64 |
| CPU | Apple M3 |
| go.mod Go version | 1.26.1 |
| Go toolchain | go1.26.5 darwin/arm64 |
| cgo | enabled |

### Workloads

Every model benchmark uses the completion-target
`Dense -> ReLU -> Dense -> Softmax` graph. `TrainBatch` and `Fit` use
categorical cross entropy and SGD. The bounded `Fit` case runs one epoch with
one batch, followed by the existing full-dataset evaluation.

| Case | Batch | Input | Hidden | Classes | First dense work | Second dense work | Dispatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Small below threshold | 8 | 32 | 64 | 16 | 16,384 | 8,192 | CPU |
| Directly below threshold | 63 | 128 | 128 | 128 | 1,032,192 | 1,032,192 | CPU |
| At threshold | 64 | 128 | 128 | 128 | 1,048,576 | 1,048,576 | Metal when available |
| Large above threshold | 128 | 256 | 128 | 128 | 4,194,304 | 2,097,152 | Metal when available |

`ColdFirstUse` creates a fresh model and times the first requested operation;
setup is outside the timed region. The first eligible Metal case also includes
process-wide device, library, and pipeline initialization. Later cold-model
cases in the same benchmark process reuse that global initialization.
`Warmed` runs the operation once before measurement and reuses model scratch.

### Commands

Native build and correctness matrix:

```sh
go test ./...
go test ./... -tags=purego
go test ./... -tags=metal -count=1
go test ./... -tags='metal purego'
go test ./model -tags=metal -run '^Test_MetalBaseline' -v -count=1
```

Compile-time unsupported-platform and architecture fallbacks:

```sh
env CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go test ./... -tags=metal -run '^$' -exec=/usr/bin/true
env CGO_ENABLED=0 GOOS=linux GOARCH=386 go test ./... -tags=metal -run '^$' -exec=/usr/bin/true
env CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go test ./... -tags='metal purego' -run '^$' -exec=/usr/bin/true
```

Cold-model baselines:

```sh
go test ./model -run '^$' -bench='SequentialMetalBaseline/.*/.*/ColdFirstUse$' -benchmem -benchtime=1x -count=1
go test ./model -tags=metal -run '^$' -bench='SequentialMetalBaseline/.*/.*/ColdFirstUse$' -benchmem -benchtime=1x -count=1
```

Warmed steady-state baselines:

```sh
go test ./model -run '^$' -bench='SequentialMetalBaseline/.*/.*/Warmed$' -benchmem -benchtime=100ms -count=1
go test ./model -tags=metal -run '^$' -bench='SequentialMetalBaseline/.*/.*/Warmed$' -benchmem -benchtime=100ms -count=1
```

Hybrid CPU kernel controls:

```sh
go test ./matrix -run '^$' -bench='DotProduct/(Length4096|Length65537)$|ElementwiseCandidates/Large1024x1024/(AddInto|MultiplyElementsInto|MultiplyScalarInPlace)/Active$' -benchmem -benchtime=200ms -count=3
go test ./matrix -tags=metal -run '^$' -bench='DotProduct/(Length4096|Length65537)$|ElementwiseCandidates/Large1024x1024/(AddInto|MultiplyElementsInto|MultiplyScalarInPlace)/Active$' -benchmem -benchtime=200ms -count=3
go test ./matrix -tags='metal purego' -run '^$' -bench='DotProduct/(Length4096|Length65537)$|ElementwiseCandidates/Large1024x1024/(AddInto|MultiplyElementsInto|MultiplyScalarInPlace)/Active$' -benchmem -benchtime=200ms -count=1
```

Session verification:

```sh
go fmt ./...
go vet ./...
go test ./... -race
```

### Synchronous Transfer Counts

Private, opt-in test counters observe the existing synchronous bridge without
changing its `1 << 20` threshold. Each eligible multiplication creates two
input buffers and one result buffer, uploads both inputs, submits and waits for
one command, and downloads one result.

| Operation | Multiplications | Buffers | Input uploads | Result downloads | Commands | Waits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Predict` | 2 | 6 | 4 | 2 | 2 | 2 |
| `Backward` | 4 | 12 | 8 | 4 | 4 | 4 |
| `TrainBatch` | 6 | 18 | 12 | 6 | 6 | 6 |
| Bounded `Fit` | 8 | 24 | 16 | 8 | 8 | 8 |

The directly-below-threshold `Predict` records zero buffers, transfers,
commands, and waits. The counter integration test passed all five assertions
for each operation on the available Metal device.

### Summary

| Operation and shape | Default ns/op | `metal` ns/op | Metal speedup | Default allocations | Metal allocations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Predict, small | 16,484 | 17,026 | 0.97x | 0 | 0 |
| Predict, directly below | 1,271,927 | 1,297,257 | 0.98x | 0 | 0 |
| Predict, at threshold | 1,312,103 | 881,963 | 1.49x | 0 | 0 |
| Predict, large | 3,843,082 | 677,978 | 5.67x | 0 | 0 |
| Backward, small | 32,404 | 32,044 | 1.01x | 0 | 0 |
| Backward, directly below | 2,687,194 | 2,755,953 | 0.97x | 0 | 0 |
| Backward, at threshold | 2,785,703 | 910,740 | 3.06x | 0 | 0 |
| Backward, large | 8,185,317 | 1,124,772 | 7.28x | 0 | 0 |
| TrainBatch, small | 53,838 | 52,397 | 1.03x | 0 | 0 |
| TrainBatch, directly below | 4,094,610 | 4,054,574 | 1.01x | 0 | 0 |
| TrainBatch, at threshold | 4,166,698 | 1,618,061 | 2.58x | 0 | 0 |
| TrainBatch, large | 12,172,718 | 2,039,388 | 5.97x | 0 | 0 |
| Fit, small | 74,170 | 71,061 | 1.04x | 10 | 10 |
| Fit, directly below | 5,441,817 | 5,434,212 | 1.00x | 10 | 10 |
| Fit, at threshold | 5,476,875 | 2,184,204 | 2.51x | 10 | 10 |
| Fit, large | 16,256,714 | 2,772,700 | 5.86x | 10 | 10 |

### Raw Cold-Model Output

Default:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialMetalBaseline/Predict/SmallBelowThreshold/ColdFirstUse-8                 1       77916 ns/op     11136 B/op    16 allocs/op
Benchmark_SequentialMetalBaseline/Predict/DirectlyBelowThreshold/ColdFirstUse-8              1     3781333 ns/op    262528 B/op    16 allocs/op
Benchmark_SequentialMetalBaseline/Predict/AtThreshold/ColdFirstUse-8                         1     3676625 ns/op    262528 B/op    16 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/ColdFirstUse-8                 1     8674708 ns/op    590208 B/op    16 allocs/op
Benchmark_SequentialMetalBaseline/Backward/SmallBelowThreshold/ColdFirstUse-8                1       77458 ns/op     18208 B/op    12 allocs/op
Benchmark_SequentialMetalBaseline/Backward/DirectlyBelowThreshold/ColdFirstUse-8             1     4729209 ns/op    262432 B/op    12 allocs/op
Benchmark_SequentialMetalBaseline/Backward/AtThreshold/ColdFirstUse-8                        1     4262791 ns/op    262432 B/op    12 allocs/op
Benchmark_SequentialMetalBaseline/Backward/LargeAboveThreshold/ColdFirstUse-8                1    10735000 ns/op    524576 B/op    12 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/ColdFirstUse-8              1      108875 ns/op     29952 B/op    32 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/DirectlyBelowThreshold/ColdFirstUse-8           1     4941292 ns/op    557824 B/op    32 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/AtThreshold/ColdFirstUse-8                      1     4828291 ns/op    557824 B/op    32 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/ColdFirstUse-8              1    13252292 ns/op   1180416 B/op    32 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/ColdFirstUse-8                     1       89875 ns/op     33328 B/op    42 allocs/op
Benchmark_SequentialMetalBaseline/Fit/DirectlyBelowThreshold/ColdFirstUse-8                  1     5545916 ns/op    689648 B/op    42 allocs/op
Benchmark_SequentialMetalBaseline/Fit/AtThreshold/ColdFirstUse-8                             1     5530582 ns/op    689648 B/op    42 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/ColdFirstUse-8                     1    15665668 ns/op   1574896 B/op    42 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/model  0.316s
```

Metal:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialMetalBaseline/Predict/SmallBelowThreshold/ColdFirstUse-8                 1       86875 ns/op     11136 B/op    16 allocs/op
Benchmark_SequentialMetalBaseline/Predict/DirectlyBelowThreshold/ColdFirstUse-8              1     3744083 ns/op    262528 B/op    16 allocs/op
Benchmark_SequentialMetalBaseline/Predict/AtThreshold/ColdFirstUse-8                         1    46375541 ns/op    262528 B/op    16 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/ColdFirstUse-8                 1     1404542 ns/op    590208 B/op    16 allocs/op
Benchmark_SequentialMetalBaseline/Backward/SmallBelowThreshold/ColdFirstUse-8                1       49625 ns/op     18208 B/op    12 allocs/op
Benchmark_SequentialMetalBaseline/Backward/DirectlyBelowThreshold/ColdFirstUse-8             1     3365291 ns/op    262432 B/op    12 allocs/op
Benchmark_SequentialMetalBaseline/Backward/AtThreshold/ColdFirstUse-8                        1     1775624 ns/op    262432 B/op    12 allocs/op
Benchmark_SequentialMetalBaseline/Backward/LargeAboveThreshold/ColdFirstUse-8                1     2355833 ns/op    524576 B/op    12 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/ColdFirstUse-8              1      100917 ns/op     29952 B/op    32 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/DirectlyBelowThreshold/ColdFirstUse-8           1     5012584 ns/op    557840 B/op    33 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/AtThreshold/ColdFirstUse-8                      1     2725833 ns/op    557824 B/op    32 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/ColdFirstUse-8              1     7208084 ns/op   1180416 B/op    32 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/ColdFirstUse-8                     1      104208 ns/op     33328 B/op    42 allocs/op
Benchmark_SequentialMetalBaseline/Fit/DirectlyBelowThreshold/ColdFirstUse-8                  1     6431334 ns/op    689648 B/op    42 allocs/op
Benchmark_SequentialMetalBaseline/Fit/AtThreshold/ColdFirstUse-8                             1     3295999 ns/op    689648 B/op    42 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/ColdFirstUse-8                     1     4949125 ns/op   1574896 B/op    42 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/model  0.316s
```

### Raw Warmed Output

Default:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialMetalBaseline/Predict/SmallBelowThreshold/Warmed-8                 7099       16484 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/DirectlyBelowThreshold/Warmed-8                97     1271927 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/AtThreshold/Warmed-8                           97     1312103 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8                   30     3843082 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Backward/SmallBelowThreshold/Warmed-8                3903       32404 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Backward/DirectlyBelowThreshold/Warmed-8               45     2687194 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Backward/AtThreshold/Warmed-8                          45     2785703 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Backward/LargeAboveThreshold/Warmed-8                  13     8185317 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/Warmed-8              2332       53838 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/DirectlyBelowThreshold/Warmed-8             30     4094610 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/AtThreshold/Warmed-8                        27     4166698 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/Warmed-8                 9    12172718 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/Warmed-8                     1705       74170 ns/op      3376 B/op    10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/DirectlyBelowThreshold/Warmed-8                    21     5441817 ns/op    131824 B/op    10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/AtThreshold/Warmed-8                               20     5476875 ns/op    131824 B/op    10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/Warmed-8                        7    16256714 ns/op    394480 B/op    10 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/model  2.486s
```

Metal:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialMetalBaseline/Predict/SmallBelowThreshold/Warmed-8                 7110       17026 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/DirectlyBelowThreshold/Warmed-8                92     1297257 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/AtThreshold/Warmed-8                          135      881963 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8                  152      677978 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Backward/SmallBelowThreshold/Warmed-8                3849       32044 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Backward/DirectlyBelowThreshold/Warmed-8               48     2755953 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Backward/AtThreshold/Warmed-8                         129      910740 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Backward/LargeAboveThreshold/Warmed-8                 100     1124772 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/Warmed-8              2428       52397 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/DirectlyBelowThreshold/Warmed-8             30     4054574 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/AtThreshold/Warmed-8                        73     1618061 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/Warmed-8                60     2039388 ns/op         0 B/op     0 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/Warmed-8                     1800       71061 ns/op      3376 B/op    10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/DirectlyBelowThreshold/Warmed-8                    21     5434212 ns/op    131824 B/op    10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/AtThreshold/Warmed-8                               54     2184204 ns/op    131824 B/op    10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/Warmed-8                       44     2772700 ns/op    394480 B/op    10 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/model  2.872s
```

### Raw Hybrid CPU Kernel Output

The default and `metal` samples are three-run controls. `metal purego` confirms
that `purego` still selects the scalar implementations.

```text
Default:
Benchmark_DotProduct/Length4096-8                                  429384       470.5 ns/op       0 B/op   0 allocs/op
Benchmark_DotProduct/Length4096-8                                  459110       473.4 ns/op       0 B/op   0 allocs/op
Benchmark_DotProduct/Length4096-8                                  511426       469.2 ns/op       0 B/op   0 allocs/op
Benchmark_DotProduct/Length65537-8                                  32323        7435 ns/op       0 B/op   0 allocs/op
Benchmark_DotProduct/Length65537-8                                  32311        7465 ns/op       0 B/op   0 allocs/op
Benchmark_DotProduct/Length65537-8                                  32322        7455 ns/op       0 B/op   0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8      2234      112311 ns/op       0 B/op   0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8      2168      112473 ns/op       0 B/op   0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8      2233      112244 ns/op       0 B/op   0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8 2208 111418 ns/op 0 B/op 0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8 2160 111474 ns/op 0 B/op 0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8 2182 112287 ns/op 0 B/op 0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8 3561 71152 ns/op 0 B/op 0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8 3501 69918 ns/op 0 B/op 0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8 3420 70868 ns/op 0 B/op 0 allocs/op

Metal:
Benchmark_DotProduct/Length4096-8                                  429261       470.8 ns/op       0 B/op   0 allocs/op
Benchmark_DotProduct/Length4096-8                                  496650       470.9 ns/op       0 B/op   0 allocs/op
Benchmark_DotProduct/Length4096-8                                  510235       469.2 ns/op       0 B/op   0 allocs/op
Benchmark_DotProduct/Length65537-8                                  32347        7529 ns/op       0 B/op   0 allocs/op
Benchmark_DotProduct/Length65537-8                                  32347        7716 ns/op       0 B/op   0 allocs/op
Benchmark_DotProduct/Length65537-8                                  32336        7593 ns/op       0 B/op   0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8      2127      111814 ns/op       0 B/op   0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8      2193      111385 ns/op       0 B/op   0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8      2137      112758 ns/op       0 B/op   0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8 2206 112315 ns/op 0 B/op 0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8 2223 111487 ns/op 0 B/op 0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8 2179 112093 ns/op 0 B/op 0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8 3332 70643 ns/op 0 B/op 0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8 3619 71547 ns/op 0 B/op 0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8 3518 70315 ns/op 0 B/op 0 allocs/op

Metal purego:
Benchmark_DotProduct/Length4096-8                                   66452        3526 ns/op       0 B/op   0 allocs/op
Benchmark_DotProduct/Length65537-8                                   4161       56850 ns/op       0 B/op   0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8       598      409680 ns/op       0 B/op   0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8 598 413237 ns/op 0 B/op 0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8 925 262770 ns/op 0 B/op 0 allocs/op
```

### Interpretation

The directly-below-threshold controls remain on CPU and match default timing
and allocation behavior. The first threshold-eligible Metal prediction is
slower when it pays global initialization (46.38 ms versus 3.68 ms), which
confirms that cold dispatch must remain part of later eligibility work.

After warm-up, synchronous Metal is already materially faster for the complete
large graph despite transferring every multiplication: 5.67x for `Predict`,
7.28x for `Backward`, 5.97x for `TrainBatch`, and 5.86x for bounded `Fit`. The
same transfer counts also expose the residency opportunity: large
`TrainBatch` still creates 18 buffers, performs 12 input uploads and 6 result
downloads, and submits and waits for 6 separate commands.

The `metal` dot-product and elementwise controls now match default SIMD timing
and preserve zero allocations. The `metal purego` controls remain about 3.7x
to 7.5x slower, confirming that the hybrid build selects SIMD while `purego`
continues to select the scalar reference. No threshold or synchronous bridge
behavior changed in this session.

## Step 3: Add a Persistent Metal Runtime

Captured on July 22, 2026.

### Environment

| Field | Value |
| --- | --- |
| OS | macOS 26.5.2 (25F84) |
| Architecture | arm64 |
| CPU | Apple M3 |
| go.mod Go version | 1.26.1 |
| Go toolchain | go1.26.5 darwin/arm64 |
| CGO | enabled |
| Metal device | available |

### Commands

Focused runtime benchmark with ten samples:

```sh
go test ./internal/device -tags=metal -run '^$' -bench='^Benchmark_MetalRuntime' -benchmem -benchtime=200ms -count=10
```

The cold case allocates a 4 KiB buffer, uploads it, creates a scope, encodes a
fill, commits and waits, downloads it, and releases both resources. The warm
case retains the buffer and measures scope creation, fill encoding, commit,
wait, and scope release.

### Summary

| Case | Median ns/op | Range ns/op | B/op | allocs/op |
| --- | ---: | ---: | ---: | ---: |
| Cold buffer and scope | 111,608 | 106,037–115,983 | 128 | 2 |
| Warm buffer reuse | 106,519 | 100,942–108,694 | 64 | 1 |

The medians are the averages of the fifth and sixth sorted samples because
each case has ten measurements.

### Raw Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/internal/device
cpu: Apple M3
Benchmark_MetalRuntime/ColdBufferAndScope-8          2114  110939 ns/op  128 B/op  2 allocs/op
Benchmark_MetalRuntime/ColdBufferAndScope-8          2166  113779 ns/op  128 B/op  2 allocs/op
Benchmark_MetalRuntime/ColdBufferAndScope-8          2265  113389 ns/op  128 B/op  2 allocs/op
Benchmark_MetalRuntime/ColdBufferAndScope-8          2252  112475 ns/op  128 B/op  2 allocs/op
Benchmark_MetalRuntime/ColdBufferAndScope-8          2335  108342 ns/op  128 B/op  2 allocs/op
Benchmark_MetalRuntime/ColdBufferAndScope-8          2344  106037 ns/op  128 B/op  2 allocs/op
Benchmark_MetalRuntime/ColdBufferAndScope-8          2082  110723 ns/op  128 B/op  2 allocs/op
Benchmark_MetalRuntime/ColdBufferAndScope-8          2229  111545 ns/op  128 B/op  2 allocs/op
Benchmark_MetalRuntime/ColdBufferAndScope-8          2041  111671 ns/op  128 B/op  2 allocs/op
Benchmark_MetalRuntime/ColdBufferAndScope-8          2232  115983 ns/op  128 B/op  2 allocs/op
Benchmark_MetalRuntime/WarmBufferReuse-8             2331  106696 ns/op   64 B/op  1 allocs/op
Benchmark_MetalRuntime/WarmBufferReuse-8             2377  106355 ns/op   64 B/op  1 allocs/op
Benchmark_MetalRuntime/WarmBufferReuse-8             2097  100942 ns/op   64 B/op  1 allocs/op
Benchmark_MetalRuntime/WarmBufferReuse-8             2307  108694 ns/op   64 B/op  1 allocs/op
Benchmark_MetalRuntime/WarmBufferReuse-8             2350  107581 ns/op   64 B/op  1 allocs/op
Benchmark_MetalRuntime/WarmBufferReuse-8             2330  106683 ns/op   64 B/op  1 allocs/op
Benchmark_MetalRuntime/WarmBufferReuse-8             2317  103996 ns/op   64 B/op  1 allocs/op
Benchmark_MetalRuntime/WarmBufferReuse-8             2072  106918 ns/op   64 B/op  1 allocs/op
Benchmark_MetalRuntime/WarmBufferReuse-8             2096  106121 ns/op   64 B/op  1 allocs/op
Benchmark_MetalRuntime/WarmBufferReuse-8             2088  106204 ns/op   64 B/op  1 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/internal/device  5.830s
```

### Interpretation

Reusing the persistent buffer removes one Go allocation and 64 bytes per
operation. The median improvement is about 5%, while command creation,
submission, and synchronization still dominate a single tiny fill. This is a
runtime-primitives measurement, not an end-to-end resident-training claim. It
supports keeping buffers resident and batching several kernels into one scope
in later sections; the synchronous matrix adapter intentionally retains its
existing transfer and wait behavior for now.

## Step 4: Add Coherent Metal Matrix Residency

Captured on July 22, 2026.

### Environment

| Field | Value |
| --- | --- |
| OS | macOS 26.5.2 (25F84) |
| Architecture | arm64 |
| CPU | Apple M3 |
| go.mod Go version | 1.26.1 |
| Go toolchain | go1.26.5 darwin/arm64 |
| CGO | enabled |
| Metal device | available |

### Commands

CPU, `purego`, and Metal-tagged CPU-fallback controls with five samples:

```sh
go test ./matrix -run '^$' -bench='^Benchmark_(MatMulInto|AddInto|Values)$' -benchmem -benchtime=200ms -count=5
go test ./matrix -tags=purego -run '^$' -bench='^Benchmark_(MatMulInto|AddInto|Values)$' -benchmem -benchtime=200ms -count=5
go test ./matrix -tags=metal -run '^$' -bench='^Benchmark_(MatMulInto|AddInto|Values)$' -benchmem -benchtime=200ms -count=5
```

Warmed resident multiplication with and without an explicit host observation,
plus the existing allocating benchmark used by the synchronous baseline:

```sh
go test ./matrix -tags=metal -run '^$' -bench='^Benchmark_MetalMatrixResidency$' -benchmem -benchtime=200ms -count=10
go test ./matrix -tags=metal -run '^$' -bench='^Benchmark_MatMulShapes/Large128x256x128$' -benchmem -benchtime=200ms -count=10
```

### Summary

| Build/case | Benchmark | Median ns/op | B/op | allocs/op |
| --- | --- | ---: | ---: | ---: |
| Default | `MatMulInto` | 148,870 | 0 | 0 |
| `purego` | `MatMulInto` | 148,812 | 0 | 0 |
| Metal CPU fallback | `MatMulInto` | 149,132 | 0 | 0 |
| Default | `Values` | 13,752 | 262,147 | 1 |
| `purego` | `Values` | 14,271 | 262,145 | 1 |
| Metal CPU fallback | `Values` | 13,041 | 262,144 | 1 |
| Default | `AddInto` | 6,992 | 0 | 0 |
| `purego` | `AddInto` | 26,071 | 0 | 0 |
| Metal CPU fallback | `AddInto` | 7,010 | 0 | 0 |
| Resident Metal | Warmed unobserved | 221,176 | 128 | 2 |
| Resident Metal | Warmed observed | 230,491 | 128 | 2 |
| Resident allocating Metal | `Large128x256x128` | 247,051 | 65,888 | 5 |

Ten-sample medians average the fifth and sixth sorted samples. Five-sample
medians use the third sorted sample. Byte counts in the table use the median
sample when benchmark sink escape accounting varied by a few bytes.

### Raw CPU-Fallback Output

Default:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMulInto-8       1536  148870 ns/op       0 B/op  0 allocs/op
Benchmark_MatMulInto-8       1668  148778 ns/op       0 B/op  0 allocs/op
Benchmark_MatMulInto-8       1620  149384 ns/op       0 B/op  0 allocs/op
Benchmark_MatMulInto-8       1636  148299 ns/op       0 B/op  0 allocs/op
Benchmark_MatMulInto-8       1647  148988 ns/op       0 B/op  0 allocs/op
Benchmark_Values-8          19584   12495 ns/op  262144 B/op  1 allocs/op
Benchmark_Values-8          18250   13830 ns/op  262146 B/op  1 allocs/op
Benchmark_Values-8          17366   13661 ns/op  262147 B/op  1 allocs/op
Benchmark_Values-8          17702   14094 ns/op  262147 B/op  1 allocs/op
Benchmark_Values-8          17400   13752 ns/op  262147 B/op  1 allocs/op
Benchmark_AddInto-8         46666    7054 ns/op       0 B/op  0 allocs/op
Benchmark_AddInto-8         34542    6843 ns/op       0 B/op  0 allocs/op
Benchmark_AddInto-8         34452    6963 ns/op       0 B/op  0 allocs/op
Benchmark_AddInto-8         34446    6992 ns/op       0 B/op  0 allocs/op
Benchmark_AddInto-8         34455    6994 ns/op       0 B/op  0 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/matrix  5.427s
```

`purego`:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMulInto-8       1513  149061 ns/op       0 B/op  0 allocs/op
Benchmark_MatMulInto-8       1609  148283 ns/op       0 B/op  0 allocs/op
Benchmark_MatMulInto-8       1636  148812 ns/op       0 B/op  0 allocs/op
Benchmark_MatMulInto-8       1611  148227 ns/op       0 B/op  0 allocs/op
Benchmark_MatMulInto-8       1653  149332 ns/op       0 B/op  0 allocs/op
Benchmark_Values-8          15046   14131 ns/op  262146 B/op  1 allocs/op
Benchmark_Values-8          16995   14271 ns/op  262145 B/op  1 allocs/op
Benchmark_Values-8          14743   15336 ns/op  262144 B/op  1 allocs/op
Benchmark_Values-8          16284   14193 ns/op  262145 B/op  1 allocs/op
Benchmark_Values-8          16833   15875 ns/op  262144 B/op  1 allocs/op
Benchmark_AddInto-8          9225   26067 ns/op       0 B/op  0 allocs/op
Benchmark_AddInto-8          9188   26035 ns/op       0 B/op  0 allocs/op
Benchmark_AddInto-8          9756   26071 ns/op       0 B/op  0 allocs/op
Benchmark_AddInto-8          9584   26169 ns/op       0 B/op  0 allocs/op
Benchmark_AddInto-8          9115   26471 ns/op       0 B/op  0 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/matrix  4.806s
```

Metal-tagged CPU fallback:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMulInto-8       1524  148764 ns/op       0 B/op  0 allocs/op
Benchmark_MatMulInto-8       1646  149113 ns/op       0 B/op  0 allocs/op
Benchmark_MatMulInto-8       1578  150740 ns/op       0 B/op  0 allocs/op
Benchmark_MatMulInto-8       1627  149940 ns/op       0 B/op  0 allocs/op
Benchmark_MatMulInto-8       1623  149132 ns/op       0 B/op  0 allocs/op
Benchmark_Values-8          16849   12949 ns/op  262144 B/op  1 allocs/op
Benchmark_Values-8          18716   12925 ns/op  262144 B/op  1 allocs/op
Benchmark_Values-8          18436   13041 ns/op  262144 B/op  1 allocs/op
Benchmark_Values-8          18637   13485 ns/op  262147 B/op  1 allocs/op
Benchmark_Values-8          17908   13290 ns/op  262148 B/op  1 allocs/op
Benchmark_AddInto-8         33927    7004 ns/op       0 B/op  0 allocs/op
Benchmark_AddInto-8         34246    7010 ns/op       0 B/op  0 allocs/op
Benchmark_AddInto-8         34215    7099 ns/op       0 B/op  0 allocs/op
Benchmark_AddInto-8         34353    7008 ns/op       0 B/op  0 allocs/op
Benchmark_AddInto-8         34248    7069 ns/op       0 B/op  0 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/matrix  5.127s
```

### Raw Resident Metal Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MetalMatrixResidency/WarmedUnobserved-8   417   601761 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedUnobserved-8   955   215378 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedUnobserved-8  1156   211221 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedUnobserved-8  1154   215755 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedUnobserved-8  1095   223667 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedUnobserved-8  1114   227445 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedUnobserved-8  1039   226148 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedUnobserved-8  1112   224155 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedUnobserved-8  1142   218684 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedUnobserved-8  1094   217191 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedObserved-8    1062   231493 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedObserved-8    1113   225115 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedObserved-8    1059   227695 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedObserved-8    1028   232487 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedObserved-8    1052   232755 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedObserved-8    1053   227775 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedObserved-8    1046   232976 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedObserved-8    1075   232878 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedObserved-8    1023   229489 ns/op  128 B/op  2 allocs/op
Benchmark_MetalMatrixResidency/WarmedObserved-8    1095   225536 ns/op  128 B/op  2 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/matrix  5.761s
```

Existing allocating benchmark:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMulShapes/Large128x256x128-8  464  463098 ns/op  65889 B/op  5 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8  966  247387 ns/op  65888 B/op  5 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8  964  244410 ns/op  65888 B/op  5 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8  968  246971 ns/op  65889 B/op  5 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8  956  246890 ns/op  65888 B/op  5 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8  960  247939 ns/op  65888 B/op  5 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8  967  245684 ns/op  65888 B/op  5 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8  984  246462 ns/op  65888 B/op  5 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8  964  247400 ns/op  65888 B/op  5 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8  962  247131 ns/op  65888 B/op  5 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/matrix  3.096s
```

### Interpretation

The optional residency pointer grows `Matrix` to six machine words, but the
existing CPU destination operations remain allocation-free. Default
`MatMulInto` is within 0.1% of the recorded 148,775 ns/op baseline, and the
Metal-tagged below-threshold controls match default SIMD timing. `purego`
retains the same zero-allocation contracts and expected scalar elementwise
selection.

For the large resident multiplication, inputs upload only on first use and the
destination remains device-newer. An explicit `ValuesInto` boundary adds about
4.2% to the warmed median. The directly comparable allocating Metal benchmark
has a 247,051 ns/op median versus the recorded 250,808 ns/op synchronous
median; its three additional small Go allocations are the lazy residency
record, staging-buffer owner, and command-scope owner. Later batching and
buffer-pool tuning, not this coherence session, own those per-command costs.

## Step 5: Batch Metal Commands Across Sequential Execution

Captured on July 22, 2026.

### Environment

| Field | Value |
| --- | --- |
| OS | macOS 26.5.2 (25F84) |
| Architecture | arm64 |
| CPU | Apple M3 |
| go.mod Go version | 1.26.1 |
| Go toolchain | go1.26.5 darwin/arm64 |
| CGO | enabled |
| Metal device | available outside the filesystem sandbox |

### Commands

Focused dependent-command and CPU-fallback benchmark with ten samples:

```sh
GOCACHE=/tmp/neuralnetwork-go-cache go test ./matrix -tags=metal -run '^$' -bench='^Benchmark_MetalCommandBatch$' -benchmem -benchtime=200ms -count=10
```

The standalone control invokes the existing public matrix boundary twice. The
batched case binds the same inputs to one private outer execution and encodes
the two dependent multiplications before finishing. The fallback case inserts
a CPU scalar addition between the multiplications, which requires completion
and one result download before lazily uploading the CPU-written matrix.

### Summary

| Case | Median ns/op | Commands/op | Waits/op | Downloads/op | Uploads/op | B/op | allocs/op |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Two standalone multiplications | 436,040 | 2 | 2 | 0 | 0 | 1,216 | 22 |
| Two batched multiplications | 243,923 | 1 | 1 | 0 | 0 | 816 | 16 |
| CPU fallback boundary | 430,578 | 2 | 2 | 1 | 1 | 976 | 19 |

The medians average the fifth and sixth sorted samples. Inputs and parameters
were already resident by the measured benchmark passes, so the two directly
comparable cases report zero uploads. The CPU fallback intentionally writes a
new host revision on every operation and therefore reports one upload.

### Raw Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MetalCommandBatch/StandaloneTwoMatMuls-8          277   799207 ns/op  2.000 commands/op  0 downloads/op  0 uploads/op  2.000 waits/op  1216 B/op  22 allocs/op
Benchmark_MetalCommandBatch/StandaloneTwoMatMuls-8          414   510737 ns/op  2.000 commands/op  0 downloads/op  0 uploads/op  2.000 waits/op  1216 B/op  22 allocs/op
Benchmark_MetalCommandBatch/StandaloneTwoMatMuls-8          367   664707 ns/op  2.000 commands/op  0 downloads/op  0 uploads/op  2.000 waits/op  1216 B/op  22 allocs/op
Benchmark_MetalCommandBatch/StandaloneTwoMatMuls-8          579   438527 ns/op  2.000 commands/op  0 downloads/op  0 uploads/op  2.000 waits/op  1216 B/op  22 allocs/op
Benchmark_MetalCommandBatch/StandaloneTwoMatMuls-8          562   448539 ns/op  2.000 commands/op  0 downloads/op  0 uploads/op  2.000 waits/op  1216 B/op  22 allocs/op
Benchmark_MetalCommandBatch/StandaloneTwoMatMuls-8          561   427017 ns/op  2.000 commands/op  0 downloads/op  0 uploads/op  2.000 waits/op  1216 B/op  22 allocs/op
Benchmark_MetalCommandBatch/StandaloneTwoMatMuls-8          550   424839 ns/op  2.000 commands/op  0 downloads/op  0 uploads/op  2.000 waits/op  1216 B/op  22 allocs/op
Benchmark_MetalCommandBatch/StandaloneTwoMatMuls-8          564   433553 ns/op  2.000 commands/op  0 downloads/op  0 uploads/op  2.000 waits/op  1216 B/op  22 allocs/op
Benchmark_MetalCommandBatch/StandaloneTwoMatMuls-8          567   423142 ns/op  2.000 commands/op  0 downloads/op  0 uploads/op  2.000 waits/op  1216 B/op  22 allocs/op
Benchmark_MetalCommandBatch/StandaloneTwoMatMuls-8          564   424627 ns/op  2.000 commands/op  0 downloads/op  0 uploads/op  2.000 waits/op  1216 B/op  22 allocs/op
Benchmark_MetalCommandBatch/BatchedTwoMatMuls-8            1002   238740 ns/op  1.000 commands/op  0 downloads/op  0 uploads/op  1.000 waits/op   816 B/op  16 allocs/op
Benchmark_MetalCommandBatch/BatchedTwoMatMuls-8            1017   246078 ns/op  1.000 commands/op  0 downloads/op  0 uploads/op  1.000 waits/op   816 B/op  16 allocs/op
Benchmark_MetalCommandBatch/BatchedTwoMatMuls-8             984   324422 ns/op  1.000 commands/op  0 downloads/op  0 uploads/op  1.000 waits/op   816 B/op  16 allocs/op
Benchmark_MetalCommandBatch/BatchedTwoMatMuls-8             885   245250 ns/op  1.000 commands/op  0 downloads/op  0 uploads/op  1.000 waits/op   816 B/op  16 allocs/op
Benchmark_MetalCommandBatch/BatchedTwoMatMuls-8            1036   242595 ns/op  1.000 commands/op  0 downloads/op  0 uploads/op  1.000 waits/op   816 B/op  16 allocs/op
Benchmark_MetalCommandBatch/BatchedTwoMatMuls-8            1002   237408 ns/op  1.000 commands/op  0 downloads/op  0 uploads/op  1.000 waits/op   816 B/op  16 allocs/op
Benchmark_MetalCommandBatch/BatchedTwoMatMuls-8            1015   237943 ns/op  1.000 commands/op  0 downloads/op  0 uploads/op  1.000 waits/op   816 B/op  16 allocs/op
Benchmark_MetalCommandBatch/BatchedTwoMatMuls-8            1003   240494 ns/op  1.000 commands/op  0 downloads/op  0 uploads/op  1.000 waits/op   816 B/op  16 allocs/op
Benchmark_MetalCommandBatch/BatchedTwoMatMuls-8            1000   247585 ns/op  1.000 commands/op  0 downloads/op  0 uploads/op  1.000 waits/op   816 B/op  16 allocs/op
Benchmark_MetalCommandBatch/BatchedTwoMatMuls-8            1012   255704 ns/op  1.000 commands/op  0 downloads/op  0 uploads/op  1.000 waits/op   816 B/op  16 allocs/op
Benchmark_MetalCommandBatch/CPUFallbackBoundary-8           549   424577 ns/op  2.000 commands/op  1.000 downloads/op  1.000 uploads/op  2.000 waits/op   976 B/op  19 allocs/op
Benchmark_MetalCommandBatch/CPUFallbackBoundary-8           566   428482 ns/op  2.000 commands/op  1.000 downloads/op  1.000 uploads/op  2.000 waits/op   976 B/op  19 allocs/op
Benchmark_MetalCommandBatch/CPUFallbackBoundary-8           578   432153 ns/op  2.000 commands/op  1.000 downloads/op  1.000 uploads/op  2.000 waits/op   976 B/op  19 allocs/op
Benchmark_MetalCommandBatch/CPUFallbackBoundary-8           559   432545 ns/op  2.000 commands/op  1.000 downloads/op  1.000 uploads/op  2.000 waits/op   976 B/op  19 allocs/op
Benchmark_MetalCommandBatch/CPUFallbackBoundary-8           562   425656 ns/op  2.000 commands/op  1.000 downloads/op  1.000 uploads/op  2.000 waits/op   976 B/op  19 allocs/op
Benchmark_MetalCommandBatch/CPUFallbackBoundary-8           549   433857 ns/op  2.000 commands/op  1.000 downloads/op  1.000 uploads/op  2.000 waits/op   976 B/op  19 allocs/op
Benchmark_MetalCommandBatch/CPUFallbackBoundary-8           525   592308 ns/op  2.000 commands/op  1.000 downloads/op  1.000 uploads/op  2.000 waits/op   976 B/op  19 allocs/op
Benchmark_MetalCommandBatch/CPUFallbackBoundary-8           570   429395 ns/op  2.000 commands/op  1.000 downloads/op  1.000 uploads/op  2.000 waits/op   976 B/op  19 allocs/op
Benchmark_MetalCommandBatch/CPUFallbackBoundary-8           564   428477 ns/op  2.000 commands/op  1.000 downloads/op  1.000 uploads/op  2.000 waits/op   976 B/op  19 allocs/op
Benchmark_MetalCommandBatch/CPUFallbackBoundary-8           559   431760 ns/op  2.000 commands/op  1.000 downloads/op  1.000 uploads/op  2.000 waits/op   976 B/op  19 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/matrix  9.002s
```

### Interpretation

Batching the two dependent resident multiplications removes one command
submission, one wait, six Go allocations, and 400 bytes per operation. Its
median is about 44% lower than the two-call standalone boundary. This is a
synthetic supported-command-chain result, not an end-to-end dense prediction
or training claim; the non-multiplication kernels remain assigned to later
sections.

The explicit CPU boundary restores two submissions and waits and adds exactly
one download plus one lazy re-upload, demonstrating that fallback preserves
coherence without downloading unrelated resident inputs.

## Step 6: Keep Dense Inference Device-Resident

Captured on July 23, 2026.

### Environment

| Field | Value |
| --- | --- |
| OS | macOS 26.5.2 (25F84) |
| Architecture | arm64 |
| CPU | Apple M3 |
| go.mod Go version | 1.26.1 |
| Go toolchain | go1.26.5 darwin/arm64 |
| CGO | enabled |
| Metal device | available outside the filesystem sandbox |

### Workloads and Commands

The small workload is `16x32 -> Dense(32,64) -> ReLU -> Dense(64,10)
-> Softmax`. Both multiplications remain below the dispatch threshold. The
large workload is the representative `256x512 -> Dense(512,512) -> ReLU
-> Dense(512,64) -> Softmax` graph. Cold cases use a fresh logical model for
each prediction while reusing the process runtime after its first
initialization. Warmed cases reuse the model, parameters, matrix residency, and
layer scratch.

Ten default and Metal samples:

```sh
GOCACHE=/tmp/neuralnetwork-section6-go-cache go test ./model -run '^$' -bench='^Benchmark_SequentialResidentPredict/(Small|Large)/(ColdFirstUse|Warmed)$' -benchmem -benchtime=100ms -count=10
GOCACHE=/tmp/neuralnetwork-section6-go-cache go test ./model -tags=metal -run '^$' -bench='^Benchmark_SequentialResidentPredict/(Small|Large)/(ColdFirstUse|Warmed)$' -benchmem -benchtime=100ms -count=10
```

The longer command confirms the noisy large warmed Metal median:

```sh
GOCACHE=/tmp/neuralnetwork-section6-go-cache go test ./model -tags=metal -run '^$' -bench='^Benchmark_SequentialResidentPredict/Large/Warmed$' -benchmem -benchtime=1s -count=10
```

The historical Section 2 synchronous comparison uses its unchanged legacy
`128x256 -> Dense(256,128) -> ReLU -> Dense(128,128) -> Softmax` shape.
Current default and resident controls use:

```sh
GOCACHE=/tmp/neuralnetwork-section6-go-cache go test ./model -run '^$' -bench='^Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed$' -benchmem -benchtime=200ms -count=10
GOCACHE=/tmp/neuralnetwork-section6-go-cache go test ./model -tags=metal -run '^$' -bench='^Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed$' -benchmem -benchtime=200ms -count=10
```

### Summary

Ten-sample medians average the fifth and sixth sorted samples.

| Workload | Default median ns/op | Resident Metal median ns/op | Speedup | Resident transfers | Resident commands/waits | Default B/op, allocs | Resident B/op, allocs |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Small cold | 34,525 | 35,385 | 0.98x | 0 uploads, 0 downloads | 0 / 0 | 20,736, 16 | 23,104, 31 |
| Small warm | 31,359 | 30,406 | 1.03x | 0 uploads, 0 downloads | 0 / 0 | 0, 0 | 0, 0 |
| Large cold | 42,712,230 | 1,581,198 | 27.01x | 5 uploads / 1,706,240 bytes, 0 downloads | 1 / 1 | 2,818,432, 16 | 2,823,271, 73 |
| Large warm | 42,622,722 | 4,125,990 | 10.33x | 0 uploads / 0 bytes, 0 downloads / 0 bytes | 1 / 1 | 0, 0 | 960, 25 |

The longer large warmed Metal run produced a 4,127,368 ns/op median, within
0.04% of the short-run median. Its individual samples still ranged from
3,450,272 to 4,601,772 ns/op.

The legacy-shape comparison is:

| Implementation | Median or recorded ns/op | Transfers and commands |
| --- | ---: | --- |
| Historical Section 2 default | 3,843,082 | CPU |
| Historical Section 2 synchronous Metal | 677,978 | 4 uploads, 2 downloads, 2 commands, 2 waits |
| Current default control | 3,853,573 | CPU |
| Current resident Metal | 1,547,023 | 0 warmed uploads/downloads, 1 command, 1 wait |

The historical synchronous and current resident samples were captured on
different dates and are not treated as a tuning claim. They show that the
smaller legacy shape remains sensitive to staging allocation and GPU operating
state even though the resident path is 2.49x faster than its same-session
default control.

### Raw Default Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3070  35082 ns/op  20736 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3417  34627 ns/op  20737 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3496  34535 ns/op  20736 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3471  34780 ns/op  20736 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3525  34286 ns/op  20736 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3531  34262 ns/op  20736 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3489  34514 ns/op  20736 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3505  34397 ns/op  20736 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3468  34504 ns/op  20736 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3525  34545 ns/op  20736 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                3823  32374 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4273  32505 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4010  29957 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4065  32186 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                3948  31135 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                3992  31588 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4017  30157 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4026  31582 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                3932  30188 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4046  30212 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8             3  43038348 ns/op  2818432 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8             3  42474236 ns/op  2818437 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8             3  42253820 ns/op  2818437 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8             3  42410750 ns/op  2818432 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8             3  42662167 ns/op  2818432 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8             3  42826069 ns/op  2818506 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8             3  42762292 ns/op  2818432 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8             3  42553972 ns/op  2818432 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8             3  43085055 ns/op  2818432 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8             3  42818972 ns/op  2818432 B/op  16 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                   3  42583667 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                   3  42562333 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                   3  42674306 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                   3  42637597 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                   3  43020764 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                   3  42664667 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                   3  42607847 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                   3  42493236 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                   3  42584945 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                   3  42897333 ns/op        0 B/op   0 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/model  11.407s
```

### Raw Metal Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          2809  35793 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23104 B/op  31 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3190  35184 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23104 B/op  31 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3321  34917 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23104 B/op  31 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3274  35239 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23104 B/op  31 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3271  35599 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23104 B/op  31 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3291  35781 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23104 B/op  31 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3313  35422 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23104 B/op  31 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3331  35569 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23104 B/op  31 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3303  35348 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23104 B/op  31 allocs/op
Benchmark_SequentialResidentPredict/Small/ColdFirstUse-8          3336  35339 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23104 B/op  31 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4096  30411 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4129  30512 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                3866  30462 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4164  30401 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4161  30398 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4094  30354 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4142  30348 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4126  30410 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4166  30375 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Small/Warmed-8                4086  30572 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8            64  1600811 ns/op  13.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  1706240 upload-bytes/op  5.000 uploads/op  1.000 waits/op  2823271 B/op  73 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8            78  1569969 ns/op  13.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  1706240 upload-bytes/op  5.000 uploads/op  1.000 waits/op  2823271 B/op  73 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8            80  1603633 ns/op  13.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  1706240 upload-bytes/op  5.000 uploads/op  1.000 waits/op  2823339 B/op  73 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8            79  1579264 ns/op  13.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  1706240 upload-bytes/op  5.000 uploads/op  1.000 waits/op  2823272 B/op  73 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8            75  1576124 ns/op  13.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  1706240 upload-bytes/op  5.000 uploads/op  1.000 waits/op  2823275 B/op  73 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8            79  1552305 ns/op  13.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  1706240 upload-bytes/op  5.000 uploads/op  1.000 waits/op  2823273 B/op  73 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8            79  1583132 ns/op  13.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  1706240 upload-bytes/op  5.000 uploads/op  1.000 waits/op  2823271 B/op  73 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8            79  3521234 ns/op  13.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  1706240 upload-bytes/op  5.000 uploads/op  1.000 waits/op  2823272 B/op  73 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8            68  1596787 ns/op  13.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  1706240 upload-bytes/op  5.000 uploads/op  1.000 waits/op  2823270 B/op  73 allocs/op
Benchmark_SequentialResidentPredict/Large/ColdFirstUse-8            78  1565937 ns/op  13.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  1706240 upload-bytes/op  5.000 uploads/op  1.000 waits/op  2823271 B/op  73 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                 100  1100267 ns/op   8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op        0 upload-bytes/op  0 uploads/op  1.000 waits/op      960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                 100  1862844 ns/op   8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op        0 upload-bytes/op  0 uploads/op  1.000 waits/op      960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                  32  4360263 ns/op   8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op        0 upload-bytes/op  0 uploads/op  1.000 waits/op      960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                  31  4321859 ns/op   8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op        0 upload-bytes/op  0 uploads/op  1.000 waits/op      960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                  31  4490876 ns/op   8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op        0 upload-bytes/op  0 uploads/op  1.000 waits/op      960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                  32  4171798 ns/op   8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op        0 upload-bytes/op  0 uploads/op  1.000 waits/op      960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                  26  4835551 ns/op   8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op        0 upload-bytes/op  0 uploads/op  1.000 waits/op      960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                  75  1845369 ns/op   8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op        0 upload-bytes/op  0 uploads/op  1.000 waits/op      960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                  62  2589017 ns/op   8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op        0 upload-bytes/op  0 uploads/op  1.000 waits/op      960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8                  51  4080181 ns/op   8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op        0 upload-bytes/op  0 uploads/op  1.000 waits/op      960 B/op  25 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/model  9.633s
```

Longer warmed confirmation:

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialResidentPredict/Large/Warmed-8  613  3988034 ns/op  8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8  247  4601772 ns/op  8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8  284  4506562 ns/op  8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8  432  3450272 ns/op  8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8  286  4451659 ns/op  8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8  446  3740127 ns/op  8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8  496  3918201 ns/op  8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8  302  4391673 ns/op  8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8  616  3604161 ns/op  8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  960 B/op  25 allocs/op
Benchmark_SequentialResidentPredict/Large/Warmed-8  373  4266702 ns/op  8.000 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  960 B/op  25 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/model  22.057s
```

### Legacy-Shape Raw Output

Default:

```text
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  57  3826366 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  66  3826826 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  66  3864615 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  66  3818186 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  66  3853115 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  61  3854030 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  66  3869438 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  66  3873658 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  61  3843649 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  66  3859083 ns/op  0 B/op  0 allocs/op
```

Resident Metal:

```text
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  243  1178022 ns/op  960 B/op  25 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  152  1568839 ns/op  960 B/op  25 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  153  1893487 ns/op  960 B/op  25 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  145  1638577 ns/op  960 B/op  25 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  189  1519660 ns/op  960 B/op  25 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  152  1559439 ns/op  960 B/op  25 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  151  1455127 ns/op  960 B/op  25 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  136  1534606 ns/op  960 B/op  25 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  146  1575034 ns/op  960 B/op  25 allocs/op
Benchmark_SequentialMetalBaseline/Predict/LargeAboveThreshold/Warmed-8  184  1404730 ns/op  960 B/op  25 allocs/op
```

### Interpretation

The supported large graph now encodes both multiplications, both bias
additions, ReLU, Softmax, and four layer-cache copies into one command buffer.
Its first prediction uploads exactly the caller input and four parameters.
Warmed predictions upload and download nothing; every parameter buffer is
reused, and the returned Softmax matrix remains device-newer until observed.
The eight warmed buffer creations are failure-atomic destination staging for
four layer outputs and four caches, not operand re-uploads.

The small graph records no Metal buffer, transfer, command, wait, byte, or Go
allocation activity after warm-up. Its Metal-tag median is about 3% faster
than default in this sample, so transparent eligibility does not impose the
material small-workload regression defined by the design.

The representative large warmed graph is about 10.3x faster than default even
with the current naive multiplication shader. The shorter legacy graph is only
2.49x faster than its same-session default control and is slower than the
historical synchronous sample, which keeps staging allocation and dispatch
tuning explicitly assigned to Section 9.

Metal Performance Shaders remains outside the approved Section 1 contract. A
tiled custom multiplication kernel was not adopted here: the end-to-end large
result already materially favors Metal, while these samples do not isolate
matrix arithmetic from destination allocation and GPU operating-state
variance well enough to justify a second, more complex shader. The existing
naive kernel and all three multiplication variants remain the maintainable
baseline for the backward and hardening sections.

## Step 7: Keep Dense Backpropagation Device-Resident

Date: July 23, 2026

Environment:

```text
Hardware: Apple M3
OS: macOS 26.5.2 (25F84)
Architecture: arm64
Go: go1.26.5 darwin/arm64
CGO: enabled
Metal device: available
Power mode: not explicitly controlled
```

Commands:

```sh
go test ./model -run '^$' -bench='SequentialResidentBackward/(Small|Large)/(ColdFirstUse|Warmed)$' -benchmem -benchtime=100ms -count=10
go test ./model -tags=metal -run '^$' -bench='SequentialResidentBackward/(Small|Large)/(ColdFirstUse|Warmed)$' -benchmem -benchtime=100ms -count=10
```

The cold case constructs a fresh model, runs the required resident forward
pass outside the timer, and times its first backward call. Its Metal counters
intentionally include both preparation calls because recording begins before
setup. The warmed case performs one complete untimed backward call, then times
repeated accumulation into the same parameter gradients. Medians below are
calculated from the ten recorded `ns/op` samples.

### Summary

| Case | Default median ns/op | `metal` median ns/op | Comparison | Metal transfers | Metal commands/waits | Allocations |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Small cold | 67,271 | 69,492 | `metal` 1.03x slower | 0 uploads, 0 downloads | 0 / 0 | default 12; `metal` 23 |
| Small warmed | 56,974 | 58,929 | `metal` 1.03x slower | 0 uploads, 0 downloads | 0 / 0 | both 0 |
| Large cold | 102,185,417 | 4,321,493 | `metal` 23.65x faster | 10 uploads, 0 downloads | 2 / 2 | default 12; `metal` 62 |
| Large warmed | 104,215,333 | 4,254,676 | `metal` 24.49x faster | 0 uploads, 0 downloads | 1 / 1 | default 0; `metal` 31 |

The Small shape stays entirely on CPU/SIMD under the `metal` tag. Its warmed
median is about 3.4% and 2 microseconds slower than default, below the design's
material-regression boundary of both 10% and 5 microseconds.

The Large warmed backward pass encodes Softmax backward, both dense backward
chains, ReLU backward, four matrix multiplications, gradient additions, and
bias reductions in one command buffer. It creates ten failure-atomic staging
buffers per call, but performs no upload or download after warm-up. The cold
counter includes the preceding forward pass and records 28 buffers, ten
uploads totaling 2,953,728 bytes, two commands, two waits, and no downloads.
The large timing remains variable with GPU operating state, but every sampled
resident result is more than 17x faster than its corresponding default sample.

### Raw Default Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1742  67237 ns/op  22048 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1783  67065 ns/op  22048 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1782  67324 ns/op  22048 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1812  67383 ns/op  22048 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1796  67305 ns/op  22048 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1776  68169 ns/op  22048 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1794  67386 ns/op  22048 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1777  67214 ns/op  22048 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1780  66931 ns/op  22048 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1791  66584 ns/op  22048 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2086  57124 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2179  56587 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2047  56680 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2216  75265 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2074  57265 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2118  56655 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2228  56625 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2149  57281 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2109  56823 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2244  57316 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8       1  100067291 ns/op  2818336 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8       1  102789126 ns/op  2818336 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8       1  100327874 ns/op  2818336 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8       1  101719459 ns/op  2818336 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8       1  100091083 ns/op  2818336 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8       1  102094791 ns/op  2818336 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8       1  102375542 ns/op  2818336 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8       1  103792792 ns/op  2818336 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8       2  103013271 ns/op  2818336 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8       1  102276042 ns/op  2818336 B/op  12 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8             1  106793458 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8             2  105275750 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8             2   99456500 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8             2  105083125 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8             1  100493167 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8             1  104100500 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8             1  100171250 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8             1  104632667 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8             2   98997896 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8             1  104330166 ns/op        0 B/op   0 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/model  10.361s
```

### Raw Metal Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1513  69096 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23987 B/op  23 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1714  69628 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23984 B/op  23 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1776  69959 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23984 B/op  23 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1728  69741 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23984 B/op  23 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1744  69692 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23984 B/op  23 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1700  68154 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23984 B/op  23 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1768  69453 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23984 B/op  23 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1725  69398 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23984 B/op  23 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1726  68128 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23984 B/op  23 allocs/op
Benchmark_SequentialResidentBackward/Small/ColdFirstUse-8    1726  69531 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  23987 B/op  23 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2088  59124 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2122  59087 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2058  58534 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2028  58503 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2026  59212 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2112  58671 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2037  58770 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2042  76303 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2080  58581 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Small/Warmed-8          2089  59334 ns/op   0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8      39  2667041 ns/op  28.00 buffers/op  2.000 commands/op  0 download-bytes/op  0 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  2822802 B/op  62 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8      52  2647790 ns/op  28.00 buffers/op  2.000 commands/op  0 download-bytes/op  0 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  2822800 B/op  62 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8      55  2442030 ns/op  28.00 buffers/op  2.000 commands/op  0 download-bytes/op  0 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  2822800 B/op  62 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8      49  5608722 ns/op  28.00 buffers/op  2.000 commands/op  0 download-bytes/op  0 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  2822806 B/op  62 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8      25  4430265 ns/op  28.00 buffers/op  2.000 commands/op  0 download-bytes/op  0 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  2822808 B/op  62 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8      54  4212721 ns/op  28.00 buffers/op  2.000 commands/op  0 download-bytes/op  0 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  2822804 B/op  62 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8      22  5707746 ns/op  28.00 buffers/op  2.000 commands/op  0 download-bytes/op  0 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  2822800 B/op  62 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8      32  3984586 ns/op  28.00 buffers/op  2.000 commands/op  0 download-bytes/op  0 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  2822800 B/op  62 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8      52  4430571 ns/op  28.00 buffers/op  2.000 commands/op  0 download-bytes/op  0 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  2822800 B/op  62 allocs/op
Benchmark_SequentialResidentBackward/Large/ColdFirstUse-8      27  4774069 ns/op  28.00 buffers/op  2.000 commands/op  0 download-bytes/op  0 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  2822808 B/op  62 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8            24  5745196 ns/op  10.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  1184 B/op  31 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8            18  5745576 ns/op  10.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  1184 B/op  31 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8            24  4556337 ns/op  10.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  1184 B/op  31 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8            27  4342944 ns/op  10.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  1184 B/op  31 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8            28  5700754 ns/op  10.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  1184 B/op  31 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8            30  4166408 ns/op  10.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  1184 B/op  31 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8            36  4015096 ns/op  10.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  1184 B/op  31 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8            30  4097750 ns/op  10.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  1184 B/op  31 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8            32  4056811 ns/op  10.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  1184 B/op  31 allocs/op
Benchmark_SequentialResidentBackward/Large/Warmed-8            31  3621923 ns/op  10.00 buffers/op  1.000 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  1.000 waits/op  1184 B/op  31 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/model  10.442s
```

## Step 8: Complete Device-Resident Dense Training

Date: July 23, 2026

Environment:

```text
Hardware: Apple M3
OS: macOS 26.5.2 (25F84)
Architecture: arm64
Go: go1.26.5 darwin/arm64
CGO: enabled
Metal device: available
Power mode: not explicitly controlled
```

Commands:

```sh
go test ./model -run '^$' -bench='SequentialResidentTraining/(TrainBatch|Fit)/(Small|Large)/(ColdFirstUse|Warmed)$' -benchmem -benchtime=100ms -count=5
go test ./model -tags=metal -run '^$' -bench='SequentialResidentTraining/(TrainBatch|Fit)/(Small|Large)/(ColdFirstUse|Warmed)$' -benchmem -benchtime=100ms -count=5
go test ./model -run '^$' -bench='SequentialMetalBaseline/(TrainBatch|Fit)/(SmallBelowThreshold|LargeAboveThreshold)/Warmed$' -benchmem -benchtime=100ms -count=5
go test ./model -tags=metal -run '^$' -bench='SequentialMetalBaseline/(TrainBatch|Fit)/(SmallBelowThreshold|LargeAboveThreshold)/Warmed$' -benchmem -benchtime=100ms -count=5
```

The new training benchmark uses a small `16x32 -> 64 -> 10` graph and a
representative large `256x512 -> 512 -> 64` graph. `Fit` is bounded to one
epoch and one batch followed by the existing full-dataset evaluation.
Medians below are calculated from the five recorded `ns/op` samples.

### Summary

| Case | Default median ns/op | `metal` median ns/op | Comparison | Warm Metal transfers | Warm Metal commands/waits | Warm allocations |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| TrainBatch, small cold | 96,217 | 100,332 | `metal` 1.04x slower | none | none | default 35; `metal` 67 |
| TrainBatch, small warm | 91,962 | 94,720 | `metal` 1.03x slower | none | none | both 0 |
| TrainBatch, large cold | 142,315,583 | 6,189,551 | `metal` 23.0x faster | 10 uploads, one 20-byte download | 2 / 2 | default 35; `metal` 162 |
| TrainBatch, large warm | 143,655,375 | 5,380,140 | `metal` 26.7x faster | no uploads, one 20-byte download | 2 / 2 | default 0; `metal` 73 |
| Fit, small cold | 126,117 | 134,328 | `metal` 1.07x slower | none | none | default 45; `metal` 78 |
| Fit, small warm | 122,794 | 128,804 | `metal` 1.05x slower | none | none | default 10; `metal` 13 |
| Fit, large cold | 186,120,333 | 7,405,592 | `metal` 25.1x faster | 11 uploads, 65,556 downloaded bytes | 3 / 3 | default 45; `metal` 199 |
| Fit, large warm | 185,102,875 | 6,716,347 | `metal` 27.6x faster | 3 uploads, 65,556 downloaded bytes | 3 / 3 | default 10; `metal` 114 |

The warmed large `TrainBatch` performs no upload for unchanged input, target,
or parameters. It downloads one 20-byte buffer containing the scalar loss and
minimal categorical diagnostic state. Its 24 buffers are failure-atomic
destinations and parameter-update staging. The first command validates targets
and returns the loss; the second combines the loss gradient, backward pass,
stable-order SGD updates, and gradient reset.

The bounded warmed `Fit` adds one evaluation prediction and its required host
loss observation. Its three uploads are the CPU-built evaluation input and
target plus the training batch workspace rewritten for the next call. Metrics,
callbacks, schedules, validation, and early stopping therefore retain their
documented host behavior.

The same-shape Section 2 control gives the following direct historical view:

| Case | Current default median ns/op | Historical synchronous Metal ns/op | Resident Metal median ns/op |
| --- | ---: | ---: | ---: |
| TrainBatch, small warm | 52,616 | 52,397 | 57,361 |
| TrainBatch, large warm | 12,096,583 | 2,039,388 | 3,166,267 |
| Fit, small warm | 71,299 | 71,061 | 77,816 |
| Fit, large warm | 16,040,351 | 2,772,700 | 3,624,424 |

The larger representative workload shows the intended end-to-end residency
benefit, while the older medium-large shape is 1.55x slower for `TrainBatch`
and 1.31x slower for `Fit` than its historical synchronous Metal sample.
Shader, staging-allocation, and threshold tuning remain assigned to Section 9;
the resident path is retained because it is correct, eliminates supported
intermediate transfers, and materially improves the representative large
training workload.

### Raw Default Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialResidentTraining/TrainBatch/Small/ColdFirstUse-8  1132  96217 ns/op  43744 B/op  35 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/ColdFirstUse-8  1166  96803 ns/op  43748 B/op  35 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/ColdFirstUse-8  1251  95706 ns/op  43744 B/op  35 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/ColdFirstUse-8  1257  97096 ns/op  43744 B/op  35 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/ColdFirstUse-8  1214  95868 ns/op  43744 B/op  35 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/Warmed-8        1326  91879 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/Warmed-8        1362  92805 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/Warmed-8        1432  91868 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/Warmed-8        1353  92127 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/Warmed-8        1436  91962 ns/op      0 B/op   0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/ColdFirstUse-8     1  141505250 ns/op  5702624 B/op  35 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/ColdFirstUse-8     1  142315583 ns/op  5702624 B/op  35 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/ColdFirstUse-8     1  142341624 ns/op  5702624 B/op  35 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/ColdFirstUse-8     1  142831750 ns/op  5702624 B/op  35 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/ColdFirstUse-8     1  141431000 ns/op  5702624 B/op  35 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/Warmed-8           1  142393292 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/Warmed-8           1  143725416 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/Warmed-8           1  143655375 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/Warmed-8           1  142675417 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/Warmed-8           1  143999792 ns/op        0 B/op   0 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/ColdFirstUse-8          980  122860 ns/op  49488 B/op  45 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/ColdFirstUse-8          979  127013 ns/op  49488 B/op  45 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/ColdFirstUse-8          894  126665 ns/op  49488 B/op  45 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/ColdFirstUse-8          932  126066 ns/op  49488 B/op  45 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/ColdFirstUse-8          951  126117 ns/op  49488 B/op  45 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/Warmed-8                966  123763 ns/op   5744 B/op  10 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/Warmed-8                958  122794 ns/op   5744 B/op  10 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/Warmed-8                933  122187 ns/op   5744 B/op  10 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/Warmed-8                976  122614 ns/op   5744 B/op  10 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/Warmed-8               1017  123313 ns/op   5744 B/op  10 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/ColdFirstUse-8            1  185552917 ns/op  6884560 B/op  45 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/ColdFirstUse-8            1  192803041 ns/op  6884560 B/op  45 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/ColdFirstUse-8            1  186170333 ns/op  6884560 B/op  45 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/ColdFirstUse-8            1  184349166 ns/op  6884560 B/op  45 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/ColdFirstUse-8            1  186120333 ns/op  6884560 B/op  45 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/Warmed-8                  1  185133917 ns/op  1181936 B/op  10 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/Warmed-8                  1  186796417 ns/op  1181936 B/op  10 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/Warmed-8                  1  184645083 ns/op  1181936 B/op  10 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/Warmed-8                  1  184021625 ns/op  1181936 B/op  10 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/Warmed-8                  1  185102875 ns/op  1181936 B/op  10 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/model  8.572s
```

### Raw Resident Metal Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialResidentTraining/TrainBatch/Small/ColdFirstUse-8  1024  100209 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  49440 B/op  67 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/ColdFirstUse-8  1190  101374 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  49445 B/op  67 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/ColdFirstUse-8  1158  101769 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  49440 B/op  67 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/ColdFirstUse-8  1184  100234 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  49440 B/op  67 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/ColdFirstUse-8  1198  100332 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  49440 B/op  67 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/Warmed-8        1270  94720 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  0 B/op  0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/Warmed-8        1291  94675 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  0 B/op  0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/Warmed-8        1260  94978 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  0 B/op  0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/Warmed-8        1244  94951 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  0 B/op  0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Small/Warmed-8        1266  94694 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  0 B/op  0 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/ColdFirstUse-8  12  10237854 ns/op  34.00 buffers/op  2.000 commands/op  20.00 download-bytes/op  1.000 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  5713764 B/op  162 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/ColdFirstUse-8  16   6625057 ns/op  34.00 buffers/op  2.000 commands/op  20.00 download-bytes/op  1.000 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  5713295 B/op  162 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/ColdFirstUse-8  19   6166204 ns/op  34.00 buffers/op  2.000 commands/op  20.00 download-bytes/op  1.000 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  5713285 B/op  162 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/ColdFirstUse-8  19   6189551 ns/op  34.00 buffers/op  2.000 commands/op  20.00 download-bytes/op  1.000 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  5713285 B/op  162 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/ColdFirstUse-8  19   6096774 ns/op  34.00 buffers/op  2.000 commands/op  20.00 download-bytes/op  1.000 downloads/op  2953728 upload-bytes/op  10.00 uploads/op  2.000 waits/op  5713285 B/op  162 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/Warmed-8        20   5380140 ns/op  24.00 buffers/op  2.000 commands/op  20.00 download-bytes/op  1.000 downloads/op  0 upload-bytes/op  0 uploads/op  2.000 waits/op  2792 B/op  73 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/Warmed-8        22   5435712 ns/op  24.00 buffers/op  2.000 commands/op  20.00 download-bytes/op  1.000 downloads/op  0 upload-bytes/op  0 uploads/op  2.000 waits/op  2792 B/op  73 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/Warmed-8        22   5387485 ns/op  24.00 buffers/op  2.000 commands/op  20.00 download-bytes/op  1.000 downloads/op  0 upload-bytes/op  0 uploads/op  2.000 waits/op  2792 B/op  73 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/Warmed-8        22   5366218 ns/op  24.00 buffers/op  2.000 commands/op  20.00 download-bytes/op  1.000 downloads/op  0 upload-bytes/op  0 uploads/op  2.000 waits/op  2792 B/op  73 allocs/op
Benchmark_SequentialResidentTraining/TrainBatch/Large/Warmed-8        22   5320347 ns/op  24.00 buffers/op  2.000 commands/op  20.00 download-bytes/op  1.000 downloads/op  0 upload-bytes/op  0 uploads/op  2.000 waits/op  2792 B/op  73 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/ColdFirstUse-8         879  130280 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  55360 B/op  78 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/ColdFirstUse-8         898  132067 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  55360 B/op  78 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/ColdFirstUse-8         903  134594 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  55360 B/op  78 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/ColdFirstUse-8         882  134328 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  55360 B/op  78 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/ColdFirstUse-8         904  138282 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  55360 B/op  78 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/Warmed-8               937  129287 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  6272 B/op  13 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/Warmed-8               930  129044 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  6272 B/op  13 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/Warmed-8               925  128459 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  6272 B/op  13 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/Warmed-8               955  128804 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  6272 B/op  13 allocs/op
Benchmark_SequentialResidentTraining/Fit/Small/Warmed-8               946  128322 ns/op  0 buffers/op  0 commands/op  0 download-bytes/op  0 downloads/op  0 upload-bytes/op  0 uploads/op  0 waits/op  6272 B/op  13 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/ColdFirstUse-8         15  7758072 ns/op  43.00 buffers/op  3.000 commands/op  65556 download-bytes/op  2.000 downloads/op  3478016 upload-bytes/op  11.00 uploads/op  3.000 waits/op  6896460 B/op  199 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/ColdFirstUse-8         15  7347778 ns/op  43.00 buffers/op  3.000 commands/op  65556 download-bytes/op  2.000 downloads/op  3478016 upload-bytes/op  11.00 uploads/op  3.000 waits/op  6896416 B/op  199 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/ColdFirstUse-8         15  7409117 ns/op  43.00 buffers/op  3.000 commands/op  65556 download-bytes/op  2.000 downloads/op  3478016 upload-bytes/op  11.00 uploads/op  3.000 waits/op  6896445 B/op  199 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/ColdFirstUse-8         15  7275186 ns/op  43.00 buffers/op  3.000 commands/op  65556 download-bytes/op  2.000 downloads/op  3478016 upload-bytes/op  11.00 uploads/op  3.000 waits/op  6896430 B/op  199 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/ColdFirstUse-8         14  7405592 ns/op  43.00 buffers/op  3.000 commands/op  65556 download-bytes/op  2.000 downloads/op  3478016 upload-bytes/op  11.00 uploads/op  3.000 waits/op  6896416 B/op  199 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/Warmed-8               18  6716347 ns/op  35.00 buffers/op  3.000 commands/op  65556 download-bytes/op  2.000 downloads/op  1114112 upload-bytes/op  3.000 uploads/op  3.000 waits/op  1186408 B/op  114 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/Warmed-8               18  8013924 ns/op  35.00 buffers/op  3.000 commands/op  65556 download-bytes/op  2.000 downloads/op  1114112 upload-bytes/op  3.000 uploads/op  3.000 waits/op  1186408 B/op  114 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/Warmed-8               15  6704386 ns/op  35.00 buffers/op  3.000 commands/op  65556 download-bytes/op  2.000 downloads/op  1114112 upload-bytes/op  3.000 uploads/op  3.000 waits/op  1186408 B/op  114 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/Warmed-8               18  6787965 ns/op  35.00 buffers/op  3.000 commands/op  65556 download-bytes/op  2.000 downloads/op  1114112 upload-bytes/op  3.000 uploads/op  3.000 waits/op  1186408 B/op  114 allocs/op
Benchmark_SequentialResidentTraining/Fit/Large/Warmed-8               18  6664167 ns/op  35.00 buffers/op  3.000 commands/op  65556 download-bytes/op  2.000 downloads/op  1114112 upload-bytes/op  3.000 uploads/op  3.000 waits/op  1186408 B/op  114 allocs/op
PASS
ok  github.com/itsmontoya/neuralnetwork/model  7.088s
```

### Raw Same-Shape Warmed Controls

```text
Default:
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/Warmed-8  2323  52141 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/Warmed-8  2409  52909 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/Warmed-8  2412  52389 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/Warmed-8  2406  52616 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/Warmed-8  2380  53493 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/Warmed-8     9  12045792 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/Warmed-8     9  12096583 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/Warmed-8     9  12119685 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/Warmed-8     9  12030380 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/Warmed-8     9  13202519 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/Warmed-8  1672  71174 ns/op  3376 B/op  10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/Warmed-8  1688  71445 ns/op  3376 B/op  10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/Warmed-8  1683  70998 ns/op  3376 B/op  10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/Warmed-8  1746  71299 ns/op  3376 B/op  10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/Warmed-8  1677  71405 ns/op  3376 B/op  10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/Warmed-8     7  16040351 ns/op  394480 B/op  10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/Warmed-8     7  16207839 ns/op  394480 B/op  10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/Warmed-8     7  15924351 ns/op  394480 B/op  10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/Warmed-8     7  16240696 ns/op  394480 B/op  10 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/Warmed-8     7  15948631 ns/op  394480 B/op  10 allocs/op

Resident Metal:
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/Warmed-8  2017  57361 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/Warmed-8  2154  57274 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/Warmed-8  2162  57622 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/Warmed-8  2077  57535 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/SmallBelowThreshold/Warmed-8  2146  57346 ns/op  0 B/op  0 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/Warmed-8  22  5250426 ns/op  2792 B/op  73 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/Warmed-8  37  3166267 ns/op  2792 B/op  73 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/Warmed-8  38  3236002 ns/op  2792 B/op  73 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/Warmed-8  32  3158811 ns/op  2792 B/op  73 allocs/op
Benchmark_SequentialMetalBaseline/TrainBatch/LargeAboveThreshold/Warmed-8  38  3067213 ns/op  2792 B/op  73 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/Warmed-8  1647  77380 ns/op  3904 B/op  13 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/Warmed-8  1591  77816 ns/op  3904 B/op  13 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/Warmed-8  1527  78710 ns/op  3904 B/op  13 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/Warmed-8  1585  77241 ns/op  3904 B/op  13 allocs/op
Benchmark_SequentialMetalBaseline/Fit/SmallBelowThreshold/Warmed-8  1645  77894 ns/op  3904 B/op  13 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/Warmed-8  25  4216117 ns/op  398952 B/op  114 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/Warmed-8  33  3608015 ns/op  398955 B/op  114 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/Warmed-8  33  3576054 ns/op  398952 B/op  114 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/Warmed-8  34  3624424 ns/op  398952 B/op  114 allocs/op
Benchmark_SequentialMetalBaseline/Fit/LargeAboveThreshold/Warmed-8  33  3625922 ns/op  398952 B/op  114 allocs/op
PASS
```

## Step 9: Harden Metal Dispatch, Resources, and Performance

Captured on July 23, 2026.

### Environment

| Field | Value |
| --- | --- |
| Hardware | Apple M3 |
| OS | macOS 26.5.2 (25F84) |
| Architecture | arm64 |
| Go toolchain | go1.26.5 darwin/arm64 |
| cgo | enabled |
| Metal device | available |
| Power mode | not explicitly controlled |

### Commands

Final warmed slices:

```sh
go test ./model -run '^$' -bench='^Benchmark_SequentialResident(Predict|Backward)/(Small|Large)/Warmed$|^Benchmark_SequentialResidentTraining/(TrainBatch|Fit)/(Small|Large)/Warmed$' -benchtime=100ms -count=10 -benchmem
go test -tags=metal ./model -run '^$' -bench='^Benchmark_SequentialResident(Predict|Backward)/(Small|Large)/Warmed$|^Benchmark_SequentialResidentTraining/(TrainBatch|Fit)/(Small|Large)/Warmed$' -benchtime=100ms -count=10 -benchmem
```

Observed and dispatch-boundary cases:

```sh
go test ./model -run '^$' -bench='^Benchmark_SequentialResidentPredictObserved/ObservedBelowThreshold/Warmed$' -benchtime=100ms -count=10 -benchmem
go test -tags=metal ./model -run '^$' -bench='^Benchmark_SequentialResidentPredictObserved/ObservedBelowThreshold/Warmed$' -benchtime=100ms -count=10 -benchmem
go test ./model -run '^$' -bench='^Benchmark_SequentialResidentPredictObserved/(Large|WarmThreshold)/Warmed$' -benchtime=100ms -count=10 -benchmem
go test -tags=metal ./model -run '^$' -bench='^Benchmark_SequentialResidentPredictObserved/(Large|WarmThreshold)/Warmed$' -benchtime=100ms -count=10 -benchmem
```

Each loop starts a fresh `go test` process, so process runtime and pipelines
are cold for every sample:

```sh
for iteration in {1..10}; do go test ./model -run '^$' -bench='^Benchmark_SequentialMetalDispatch/Predict/ReadyThreshold/ColdFirstUse$' -benchtime=1x -count=1 -benchmem; done
for iteration in {1..10}; do go test -tags=metal ./model -run '^$' -bench='^Benchmark_SequentialMetalDispatch/Predict/ReadyThreshold/ColdFirstUse$' -benchtime=1x -count=1 -benchmem; done
for iteration in {1..10}; do go test ./model -run '^$' -bench='^Benchmark_SequentialResidentTraining/TrainBatch/Large/ColdFirstUse$' -benchtime=1x -count=1 -benchmem; done
for iteration in {1..10}; do go test -tags=metal ./model -run '^$' -bench='^Benchmark_SequentialResidentTraining/TrainBatch/Large/ColdFirstUse$' -benchtime=1x -count=1 -benchmem; done
```

Profiles and focused hardening gates:

```sh
go test ./model -run '^$' -bench='^Benchmark_SequentialResident(Predict|Backward)/Large/Warmed$|^Benchmark_SequentialResidentTraining/(TrainBatch|Fit)/Large/Warmed$' -benchtime=1s -count=1 -cpuprofile=/private/tmp/neuralnetwork-section9-final-default.cpu -memprofile=/private/tmp/neuralnetwork-section9-final-default.mem
go test -tags=metal ./model -run '^$' -bench='^Benchmark_SequentialResident(Predict|Backward)/Large/Warmed$|^Benchmark_SequentialResidentTraining/(TrainBatch|Fit)/Large/Warmed$' -benchtime=1s -count=1 -cpuprofile=/private/tmp/neuralnetwork-section9-final-metal.cpu -memprofile=/private/tmp/neuralnetwork-section9-final-metal.mem
go tool preprofile -i /private/tmp/neuralnetwork-section9-final-default.cpu -o /private/tmp/neuralnetwork-section9-final-default.pre
go tool preprofile -i /private/tmp/neuralnetwork-section9-final-metal.cpu -o /private/tmp/neuralnetwork-section9-final-metal.pre
go test -tags=metal ./model -run '^Test_SequentialResident(SteadyStateAllocationsAndResources|DistinctModelsRunConcurrently|LongRunningMixedStress)$' -count=1
go test -tags=metal ./... -count=1
```

### Summary

Medians are calculated from the ten raw `ns/op` samples below.

| Slice | Default median ns/op | `metal` median ns/op | Default allocs | Metal allocs | Comparison |
| --- | ---: | ---: | ---: | ---: | ---: |
| Predict, small | 30,236 | 30,247 | 0 | 0 | `metal` 11 ns slower |
| Predict, large | 42,835,153 | 4,365,861 | 0 | 25 | Metal 9.8x faster |
| Backward, small | 57,502 | 56,805 | 0 | 0 | `metal` 697 ns faster |
| Backward, large | 104,525,053 | 2,045,029 | 0 | 31 | Metal 51.1x faster |
| TrainBatch, small | 92,065 | 90,374 | 0 | 0 | `metal` 1,691 ns faster |
| TrainBatch, large | 143,308,708 | 7,717,370 | 0 | 73 | Metal 18.6x faster |
| Fit, small | 124,099 | 122,730 | 10 | 10 | `metal` 1,369 ns faster |
| Fit, large | 186,582,667 | 6,686,006 | 10 | 114 | Metal 27.9x faster |

Warm large Metal activity is invariant across the timed iterations:

| Slice | Buffers | Uploads / bytes | Downloads / bytes | Kernels | Commands / waits | Fallback barriers | Allocations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Predict | 8 | 0 / 0 | 0 / 0 | 10 | 1 / 1 | 0 | 25 |
| Backward | 10 | 0 / 0 | 0 / 0 | 12 | 1 / 1 | 0 | 31 |
| TrainBatch | 24 | 0 / 0 | 1 / 20 | 32 | 2 / 2 | 1 | 73 |
| Fit | 35 | 3 / 1,114,112 | 2 / 65,556 | 42 | 3 / 3 | 1 | 114 |

Focused allocation tests gate these values at 25, 31, and 73 allocations for
warm prediction, backward, and `TrainBatch`. The same tests require exact
kernel/command/transfer counts, no increase in live buffers, bytes, or scopes
after garbage collection, balanced created/released resources, and equal
submitted/completed command counts.

Dispatch medians:

| Case | Default median ns/op | `metal` median ns/op | Metal activity | Decision |
| --- | ---: | ---: | --- | --- |
| Observed `1 << 20`, ready process | 730,481 | 729,744 | none | Below the `1 << 22` warm cutoff; use CPU/SIMD. |
| Observed `1 << 22`, ready process | 5,345,624 | 1,125,654 | 8 buffers, 10 kernels, 1 command/wait, 131,072-byte download | Use Metal. |
| `1 << 22`, cold process | 11,665,271 | 11,801,438 | none | Below the `1 << 26` cold cutoff; do not initialize Metal. |
| Large TrainBatch, cold process | 163,451,980 | 58,347,021 | 34 buffers, 10 uploads / 2,953,728 bytes, 1 download / 20 bytes, 32 kernels, 2 commands/waits | Crosses `1 << 26`; initialize and use Metal. |

Before hardening, the observed `1 << 20` case measured about 867 microseconds
with Metal versus 730.9 microseconds on CPU and created eight buffers, ten
kernels, one command/wait, and a 4,096-byte download. The higher warm cutoff
removes that material regression. A separate fresh-process diagnostic at the
old `1 << 20` cutoff fell from 42,863,375 ns/op and 74 allocations when it
initialized Metal to a CPU/SIMD range of 2,470,000-3,460,000 ns/op and 16
allocations after cold preflight.

### Raw Timing Samples

Each list is the raw `ns/op` field in command output order.

```text
Default Predict/Small:
30329 30310 30211 30175 30118 30250 30358 30245 30226 30169
Metal Predict/Small:
30075 30336 30470 30245 30477 30209 30237 30223 30353 30248

Default Predict/Large:
42831792 43481583 42974653 42858667 42608819 42660847 42573097 42747153 42838514 42939264
Metal Predict/Large:
1955468 2708246 4134123 4401188 4394461 4770178 4337261 4033895 4773861 4668373

Default Backward/Small:
58520 59568 59027 56555 56605 58153 59177 56515 56851 56822
Metal Backward/Small:
57050 56641 57494 56568 56972 56795 56737 58352 56815 56608

Default Backward/Large:
104617292 104763730 104045833 104684250 105790208 104499938 104362146 103161542 104550167 104254708
Metal Backward/Large:
2055546 2046192 2051744 2041223 2063064 2042144 2043865 2042034 2365843 2040898

Default TrainBatch/Small:
91408 93458 94181 91513 92348 91454 91566 94342 91781 93026
Metal TrainBatch/Small:
90428 89868 90151 90881 90828 90394 90007 90387 90361 89985

Default TrainBatch/Large:
143372708 143910542 143394916 143244708 144022666 141831500 142692125 143187959 144570917 142234917
Metal TrainBatch/Large:
5486242 5353920 7330648 7125102 7018144 8104092 9588125 12185382 19242693 11364819

Default Fit/Small:
123437 123737 125963 128733 124025 124172 123531 126848 124387 123659
Metal Fit/Small:
122916 122349 123569 122865 123806 122048 122945 121944 122594 122440

Default Fit/Large:
185992792 186711500 190904042 185694709 190628166 187267375 186279458 186453834 187848833 186189500
Metal Fit/Large:
10697775 6756003 6666370 6624657 6695440 6647597 6676572 6609764 7981297 6840654

Default observed 1 << 20:
741522 730251 729437 730711 731859 728690 735483 731071 727170 727952
Metal observed 1 << 20:
757353 728531 737274 728132 735811 724664 730956 732780 722959 724389

Default observed 1 << 22:
5291909 5399682 5318532 5366538 5409581 5323667 5390236 5324710 5319587 5408609
Metal observed 1 << 22:
706253 1066719 1253602 1619725 1127827 1105105 1107445 1165410 1236527 1123481

Default cold 1 << 22:
5714875 5364084 13309083 13417749 12869125 9621250 10720874 13370375 12609667 7607709
Metal-tag cold 1 << 22:
12557958 8619458 5628167 12971209 7770292 10152958 12778458 12591666 12388917 11213959

Default cold large TrainBatch:
163155791 163160042 164074375 157043750 163930334 163712041 163636708 155698624 163267251 163697918
Metal cold large TrainBatch:
65400250 54169624 59158875 56712417 57535167 54019333 62046292 59702458 54917583 59767209
```

### Raw Profile Summary

Profile benchmark timings are excluded from comparisons because sampling
changes execution cost. `go tool preprofile` reports these leading stacks:

```text
Default:
matrix.matMulInto -> matrix.matMulIntoPure                         336
matrix.matMulRightTransposeInto -> matMulRightTransposeIntoPure   242
matrix.matMulLeftTransposeInto -> matMulLeftTransposeIntoPure     149
matrix.copyMatrixHost -> runtime.memmove                            3

Metal:
runtime._ExternalCode                                             147
nn_metal_buffer_new -> runtime.cgocall                             87
nn_metal_buffer_release -> runtime.cgocall                         17
nn_metal_scope_encode_copy -> runtime.cgocall                      10
nn_metal_scope_commit -> runtime.cgocall                            9
nn_metal_buffer_download -> runtime.cgocall                         4
runtime semaphore wait/wakeup                                      6
nn_metal_buffer_upload -> runtime.cgocall                           2
```

### Interpretation

The `1 << 26` cold and `1 << 22` ready thresholds are retained. They are
conservative process-level choices based on end-to-end initialization,
transfer, command, wait, and observation costs, not isolated multiplication
timing. Small and frequently observed `1 << 20` work stays on CPU/SIMD with no
material `metal`-tag regression; the exact ready cutoff is 4.75x faster even
when its result is observed.

No shader, elementwise, reduction, buffer-cache, or command-cap change is
justified by these results. The existing command boundaries are required by
the synchronous API and pre-update scalar loss, while an idle buffer cache
would weaken the zero-idle-resource bound without a measured end-to-end gain.
The final large `TrainBatch` and `Fit` results exceed the Section 1 gates by
wide margins, so device-resident dense training is retained.
