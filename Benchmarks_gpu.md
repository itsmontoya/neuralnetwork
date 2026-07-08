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
