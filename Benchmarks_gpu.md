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
