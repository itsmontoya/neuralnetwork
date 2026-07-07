# V1 Benchmarks

Captured on July 6, 2026.

## Baseline Command

```sh
go test ./matrix ./model -run '^$' -bench=. -benchmem
```

## Expanded Measurement Command

The expanded command was run after adding benchmark-only coverage and before
any production performance changes.

```sh
go test ./matrix ./layer ./model ./optimizer -run '^$' -bench=. -benchmem
```

## Environment

| Field | Value |
| --- | --- |
| OS | darwin |
| Architecture | arm64 |
| CPU | Apple M3 |
| Go version | go1.26.1 |

## Results

| Package | Benchmark | Iterations | ns/op | B/op | allocs/op |
| --- | --- | ---: | ---: | ---: | ---: |
| `matrix` | `Benchmark_MatMul-8` | 7863 | 152175 | 32816 | 2 |
| `matrix` | `Benchmark_MatMulInto-8` | 7764 | 151347 | 0 | 0 |
| `matrix` | `Benchmark_Clone-8` | 42266 | 29435 | 524337 | 2 |
| `matrix` | `Benchmark_Values-8` | 37258 | 28230 | 524288 | 1 |
| `matrix` | `Benchmark_Add-8` | 19386 | 61380 | 524336 | 2 |
| `matrix` | `Benchmark_AddInto-8` | 31903 | 37881 | 0 | 0 |
| `matrix` | `Benchmark_AddInPlace-8` | 31879 | 37539 | 0 | 0 |
| `matrix` | `Benchmark_AddScaledInPlace-8` | 38822 | 30940 | 0 | 0 |
| `matrix` | `Benchmark_Subtract-8` | 21842 | 59561 | 524336 | 2 |
| `matrix` | `Benchmark_MultiplyElements-8` | 19341 | 63504 | 524336 | 2 |
| `matrix` | `Benchmark_DivideElements-8` | 17095 | 70372 | 524336 | 2 |
| `matrix` | `Benchmark_AddScalar-8` | 23730 | 50638 | 524337 | 2 |
| `matrix` | `Benchmark_MultiplyScalar-8` | 20983 | 48706 | 524337 | 2 |
| `matrix` | `Benchmark_MultiplyScalarInPlace-8` | 66152 | 18382 | 0 | 0 |
| `matrix` | `Benchmark_DivideScalar-8` | 24816 | 51142 | 524337 | 2 |
| `matrix` | `Benchmark_Transpose-8` | 14635 | 82410 | 262192 | 2 |
| `matrix` | `Benchmark_TransposeInto-8` | 15693 | 76401 | 0 | 0 |
| `matrix` | `Benchmark_RowSums-8` | 21356 | 50810 | 2048 | 1 |
| `matrix` | `Benchmark_ColumnSums-8` | 41066 | 31108 | 2048 | 1 |
| `matrix` | `Benchmark_ColumnSumsInto-8` | 35754 | 33388 | 0 | 0 |
| `matrix` | `Benchmark_AddRowVectorInPlace-8` | 33020 | 36505 | 0 | 0 |
| `matrix` | `Benchmark_Apply-8` | 9895 | 122847 | 524336 | 2 |
| `layer` | `Benchmark_DenseForward_XOR-8` | 7876819 | 141.1 | 288 | 4 |
| `layer` | `Benchmark_DenseForward_MediumBatch-8` | 7555 | 161924 | 98400 | 4 |
| `layer` | `Benchmark_DenseBackward_XOR-8` | 3775456 | 317.4 | 528 | 10 |
| `layer` | `Benchmark_DenseBackward_MediumBatch-8` | 3698 | 325237 | 99056 | 10 |
| `model` | `Benchmark_SequentialTrainBatch_XOR-8` | 387598 | 3090 | 5056 | 102 |
| `model` | `Benchmark_SequentialFit_XOR-8` | 280663 | 4391 | 7672 | 149 |
| `model` | `Benchmark_SequentialTrainBatch_SyntheticDense-8` | 1446 | 829006 | 1050163 | 50 |
| `model` | `Benchmark_SequentialFit_SyntheticDense-8` | 1015 | 1193491 | 2171848 | 295 |
| `optimizer` | `Benchmark_SGDUpdate_SteadyState-8` | 278130 | 4290 | 0 | 0 |
| `optimizer` | `Benchmark_MomentumUpdate_SteadyState-8` | 169201 | 7090 | 0 | 0 |
| `optimizer` | `Benchmark_AdamUpdate_SteadyState-8` | 54430 | 21709 | 177184 | 44 |

## V2 Performance Targets

These targets are release gates for v2 performance work. Matrix benchmarks not
listed here remain diagnostic coverage and should not regress without an
explicit maintainer decision.

| Package | Benchmark | Baseline ns/op | Target ns/op | Baseline B/op | Target B/op | Baseline allocs/op | Target allocs/op |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matrix` | `Benchmark_MatMul-8` | 152175 | 129349 | 32816 | 32816 | 2 | 2 |
| `matrix` | `Benchmark_MatMulInto-8` | 151347 | 128645 | 0 | 0 | 0 | 0 |
| `matrix` | `Benchmark_AddInto-8` | 37881 | 32200 | 0 | 0 | 0 | 0 |
| `matrix` | `Benchmark_AddScaledInPlace-8` | 30940 | 26300 | 0 | 0 | 0 | 0 |
| `matrix` | `Benchmark_MultiplyScalarInPlace-8` | 18382 | 15600 | 0 | 0 | 0 | 0 |
| `matrix` | `Benchmark_TransposeInto-8` | 76401 | 64900 | 0 | 0 | 0 | 0 |
| `matrix` | `Benchmark_ColumnSumsInto-8` | 33388 | 28400 | 0 | 0 | 0 | 0 |
| `matrix` | `Benchmark_AddRowVectorInPlace-8` | 36505 | 31000 | 0 | 0 | 0 | 0 |
| `layer` | `Benchmark_DenseForward_XOR-8` | 141.1 | 113 | 288 | 144 | 4 | 2 |
| `layer` | `Benchmark_DenseForward_MediumBatch-8` | 161924 | 129539 | 98400 | 49200 | 4 | 2 |
| `layer` | `Benchmark_DenseBackward_XOR-8` | 317.4 | 222 | 528 | 264 | 10 | 5 |
| `layer` | `Benchmark_DenseBackward_MediumBatch-8` | 325237 | 227666 | 99056 | 49528 | 10 | 5 |
| `model` | `Benchmark_SequentialTrainBatch_XOR-8` | 3090 | 2470 | 5056 | 3540 | 102 | 76 |
| `model` | `Benchmark_SequentialFit_XOR-8` | 4391 | 3510 | 7672 | 5360 | 149 | 112 |
| `model` | `Benchmark_SequentialTrainBatch_SyntheticDense-8` | 829006 | 663205 | 1050163 | 630098 | 50 | 38 |
| `model` | `Benchmark_SequentialFit_SyntheticDense-8` | 1193491 | 954793 | 2171848 | 1303109 | 295 | 221 |
| `optimizer` | `Benchmark_SGDUpdate_SteadyState-8` | 4290 | 3860 | 0 | 0 | 0 | 0 |
| `optimizer` | `Benchmark_MomentumUpdate_SteadyState-8` | 7090 | 6380 | 0 | 0 | 0 | 0 |
| `optimizer` | `Benchmark_AdamUpdate_SteadyState-8` | 21709 | 15196 | 177184 | 0 | 44 | 0 |

## V2 Allocation Audit: Matrix Destination Variants

Captured on July 6, 2026.

### Command

```sh
go test ./matrix ./layer ./model ./optimizer -run '^$' -bench=. -benchmem
```

### Findings

The first allocation pass used benchmark output and code inspection. Memory
profiles were not needed because the allocation sources were direct constructor,
copy, or `Values` calls in the measured paths.

| Package | Benchmarks | Finding | Follow-up |
| --- | --- | --- | --- |
| `matrix` | `Benchmark_Subtract`, `Benchmark_MultiplyElements`, `Benchmark_DivideElements`, `Benchmark_AddScalar`, `Benchmark_MultiplyScalar`, `Benchmark_DivideScalar`, `Benchmark_Apply` | Each allocating operation created a result matrix and backing slice for every call, with no destination form available to callers. | Added matching `Into` methods and benchmarks. Existing allocating methods keep returning owned results. |
| `matrix` | `Benchmark_RowSums` | Row reductions allocated a result slice for every call, unlike column sums which already had `ColumnSumsInto`. | Added `RowSumsInto` using a `[Rows(), 1]` destination matrix. |
| `matrix` | `Benchmark_Clone`, `Benchmark_Values` | Allocations are part of the public ownership contract. | Keep as-is; use `CopyFrom` or caller-owned destinations where ownership is explicit. |
| `layer` | `Benchmark_DenseForward_*` | Allocations come from the matrix product result and the input cache clone. Bias addition already happens in-place. | Next dense-forward pass should add stable-shape scratch reuse while preserving output lifetime expectations. |
| `layer` | `Benchmark_DenseBackward_*` | Allocations come from input transpose, weight-gradient product, bias-gradient matrix, weight transpose, and input-gradient product. | Next dense-backward pass should use transpose-aware multiplication or direct accumulation and write bias gradients directly. |
| `model` | `Benchmark_SequentialTrainBatch_*`, `Benchmark_SequentialFit_*` | Allocation counts primarily inherit layer allocations, with additional batching and evaluation copies still to audit. | Defer until dense-layer and data ownership work is complete. |
| `optimizer` | `Benchmark_AdamUpdate_SteadyState` | Adam copies parameter values, gradients, and moment buffers out through `Values`, then copies updates back. | Defer until a direct owned-buffer update path is chosen. |

### Added Matrix Destination Benchmarks

| Benchmark | Iterations | ns/op | B/op | allocs/op |
| --- | ---: | ---: | ---: | ---: |
| `Benchmark_SubtractInto-8` | 31837 | 37605 | 0 | 0 |
| `Benchmark_MultiplyElementsInto-8` | 31791 | 37601 | 0 | 0 |
| `Benchmark_DivideElementsInto-8` | 26812 | 45125 | 0 | 0 |
| `Benchmark_AddScalarInto-8` | 44844 | 26729 | 0 | 0 |
| `Benchmark_MultiplyScalarInto-8` | 37149 | 34592 | 0 | 0 |
| `Benchmark_DivideScalarInto-8` | 44607 | 28223 | 0 | 0 |
| `Benchmark_RowSumsInto-8` | 27469 | 43906 | 0 | 0 |
| `Benchmark_ApplyInto-8` | 10000 | 103690 | 0 | 0 |

The new destination variants all report zero steady-state allocations. The
remaining layer, model, and Adam allocation counts are unchanged in this pass.

### Raw Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMul-8                  	    7708	    154633 ns/op	   32816 B/op	       2 allocs/op
Benchmark_MatMulInto-8              	    7946	    153078 ns/op	       0 B/op	       0 allocs/op
Benchmark_Clone-8                   	   42214	     28803 ns/op	  524337 B/op	       2 allocs/op
Benchmark_Values-8                  	   40246	     29021 ns/op	  524288 B/op	       1 allocs/op
Benchmark_Add-8                     	   21246	     58654 ns/op	  524336 B/op	       2 allocs/op
Benchmark_AddInto-8                 	   31614	     37822 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddInPlace-8              	   31695	     38061 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScaledInPlace-8        	   38628	     30993 ns/op	       0 B/op	       0 allocs/op
Benchmark_Subtract-8                	   20595	     56620 ns/op	  524336 B/op	       2 allocs/op
Benchmark_SubtractInto-8            	   31837	     37605 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyElements-8        	   21849	     56456 ns/op	  524336 B/op	       2 allocs/op
Benchmark_MultiplyElementsInto-8    	   31791	     37601 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideElements-8          	   18018	     68166 ns/op	  524336 B/op	       2 allocs/op
Benchmark_DivideElementsInto-8      	   26812	     45125 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScalar-8               	   20505	     56784 ns/op	  524336 B/op	       2 allocs/op
Benchmark_AddScalarInto-8           	   44844	     26729 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalar-8          	   25419	     56446 ns/op	  524337 B/op	       2 allocs/op
Benchmark_MultiplyScalarInto-8      	   37149	     34592 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInPlace-8   	   63949	     18807 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideScalar-8            	   24608	     50027 ns/op	  524338 B/op	       2 allocs/op
Benchmark_DivideScalarInto-8        	   44607	     28223 ns/op	       0 B/op	       0 allocs/op
Benchmark_Transpose-8               	   13678	     80920 ns/op	  262193 B/op	       2 allocs/op
Benchmark_TransposeInto-8           	   15678	     76039 ns/op	       0 B/op	       0 allocs/op
Benchmark_RowSums-8                 	   23072	     51643 ns/op	    2048 B/op	       1 allocs/op
Benchmark_RowSumsInto-8             	   27469	     43906 ns/op	       0 B/op	       0 allocs/op
Benchmark_ColumnSums-8              	   40813	     29351 ns/op	    2048 B/op	       1 allocs/op
Benchmark_ColumnSumsInto-8          	   35198	     33573 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddRowVectorInPlace-8     	   32590	     36794 ns/op	       0 B/op	       0 allocs/op
Benchmark_Apply-8                   	    9566	    121638 ns/op	  524336 B/op	       2 allocs/op
Benchmark_ApplyInto-8               	   10000	    103690 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	48.703s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/layer
cpu: Apple M3
Benchmark_DenseForward_XOR-8            	 7793666	       142.3 ns/op	     288 B/op	       4 allocs/op
Benchmark_DenseForward_MediumBatch-8    	    7483	    164037 ns/op	   98400 B/op	       4 allocs/op
Benchmark_DenseBackward_XOR-8           	 3749528	       321.6 ns/op	     528 B/op	      10 allocs/op
Benchmark_DenseBackward_MediumBatch-8   	    3628	    338486 ns/op	   99056 B/op	      10 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/layer	5.492s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialTrainBatch_XOR-8              	  388257	      3099 ns/op	    5056 B/op	     102 allocs/op
Benchmark_SequentialFit_XOR-8                     	  280002	      4337 ns/op	    7672 B/op	     149 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8   	    1430	    832642 ns/op	 1050162 B/op	      50 allocs/op
Benchmark_SequentialFit_SyntheticDense-8          	    1002	   1194729 ns/op	 2171847 B/op	     295 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/model	6.259s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/optimizer
cpu: Apple M3
Benchmark_SGDUpdate_SteadyState-8        	  278445	      4305 ns/op	       0 B/op	       0 allocs/op
Benchmark_MomentumUpdate_SteadyState-8   	  168194	      7160 ns/op	       0 B/op	       0 allocs/op
Benchmark_AdamUpdate_SteadyState-8       	   55034	     22023 ns/op	  177184 B/op	      44 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/optimizer	5.127s
```

## V2 Dense Allocation Reduction

Captured on July 6, 2026.

### Commands

```sh
go test ./layer -run '^$' -bench=Dense -benchmem
go test ./matrix -run '^$' -bench=MatMul -benchmem
go test ./matrix -run '^$' -bench=ColumnSums -benchmem
```

### Implementation Notes

Dense forward now retains stable-shape output and input-cache scratch matrices.
Dense backward now retains stable-shape gradient scratch matrices, computes
`input^T * outputGradient` and `outputGradient * weights^T` with transpose-aware
matrix helpers, and accumulates output-gradient column sums directly into the
bias gradient.

The transpose-aware kernels are private matrix helpers with narrow exported
`Into` wrappers because `layer.Dense` lives outside the `matrix` package. The
wrappers preserve destination shape validation and reject destination aliasing.

### Dense Before

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/layer
cpu: Apple M3
Benchmark_DenseForward_XOR-8            	 7653796	       138.3 ns/op	     288 B/op	       4 allocs/op
Benchmark_DenseForward_MediumBatch-8    	    7454	    161317 ns/op	   98400 B/op	       4 allocs/op
Benchmark_DenseBackward_XOR-8           	 3487857	       320.7 ns/op	     528 B/op	      10 allocs/op
Benchmark_DenseBackward_MediumBatch-8   	    3663	    328354 ns/op	   99056 B/op	      10 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/layer	5.453s
```

### Dense After

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/layer
cpu: Apple M3
Benchmark_DenseForward_XOR-8            	11071054	        95.21 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_MediumBatch-8    	    7483	    158685 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_XOR-8           	 7946544	       151.5 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_MediumBatch-8   	    3664	    324080 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/layer	6.255s
```

### Matrix Helper Results

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMul-8                     	    7797	    154799 ns/op	   32817 B/op	       2 allocs/op
Benchmark_MatMulInto-8                 	    7906	    153849 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulLeftTransposeInto-8    	    7748	    154676 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeInto-8   	    7452	    167003 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	5.862s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_ColumnSums-8                 	   39604	     29091 ns/op	    2048 B/op	       1 allocs/op
Benchmark_ColumnSumsInto-8             	   35931	     33499 ns/op	       0 B/op	       0 allocs/op
Benchmark_AccumulateColumnSumsInto-8   	   35869	     33413 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	4.743s
```

### Interpretation

The dense forward and backward hot paths now report zero steady-state
allocations for the measured stable-shape cases. Dense forward improves both
small and medium throughput. Dense backward improves the XOR-sized case
substantially and removes allocations from the medium case, but medium
throughput remains close to baseline and does not yet meet the v2 target.

## V2 Layer Scratch Reuse Review

Captured on July 6, 2026.

### Command

```sh
go test ./layer -run '^$' -bench='(Activation|Dropout|BatchNormalization)' -benchmem
```

### Implementation Notes

Activation layers now reuse their input-cache matrix for stable input shapes.
Elementwise activation helpers use matrix `Apply` and `MultiplyElementsInto`
instead of copying through temporary value slices.

Dropout now retains stable-shape output, mask, gradient, and value scratch
storage. Training mode still uses the caller-provided random source, and
evaluation mode still behaves as identity while ignoring any previous training
mask.

Batch normalization now retains stable-shape output, input-gradient, parameter
gradient, statistic, and value scratch storage. Running mean and variance are
updated in their existing matrices.

The matrix package added `ValuesInto` and `CopyValuesFrom` so callers can copy
through owned slices without exposing mutable matrix storage.

### Layer Scratch Before

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/layer
cpu: Apple M3
Benchmark_ActivationForward_MediumBatch-8                    	   19392	     60087 ns/op	  262241 B/op	       6 allocs/op
Benchmark_ActivationBackward_MediumBatch-8                   	   16794	     72203 ns/op	  262192 B/op	       5 allocs/op
Benchmark_DropoutForwardTraining_MediumBatch-8               	   12793	     92954 ns/op	  327776 B/op	       7 allocs/op
Benchmark_DropoutBackwardTraining_MediumBatch-8              	  171691	      6980 ns/op	   65584 B/op	       2 allocs/op
Benchmark_BatchNormalizationForwardTraining_MediumBatch-8    	   34942	     34183 ns/op	  334017 B/op	      21 allocs/op
Benchmark_BatchNormalizationBackwardTraining_MediumBatch-8   	   43413	     27957 ns/op	  264848 B/op	      12 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/layer	11.195s
```

### Layer Scratch After

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/layer
cpu: Apple M3
Benchmark_ActivationForward_MediumBatch-8                    	   22641	     52492 ns/op	   65584 B/op	       2 allocs/op
Benchmark_ActivationBackward_MediumBatch-8                   	   19569	     61749 ns/op	   65584 B/op	       2 allocs/op
Benchmark_DropoutForwardTraining_MediumBatch-8               	   13761	     84838 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutBackwardTraining_MediumBatch-8              	  244677	      5244 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationForwardTraining_MediumBatch-8    	   44822	     24538 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationBackwardTraining_MediumBatch-8   	   53437	     22271 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/layer	9.893s
```

### Interpretation

Dropout and batch normalization now report zero steady-state allocations for
the measured stable-shape training cases. Activation layers still allocate the
returned activation or gradient matrix, but the cache and helper-copy
allocations were removed.

## Raw Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMul-8       	    7794	    152069 ns/op	   32816 B/op	       2 allocs/op
Benchmark_MatMulInto-8   	    8097	    151347 ns/op	       0 B/op	       0 allocs/op
Benchmark_Add-8          	   21322	     56745 ns/op	  524337 B/op	       2 allocs/op
Benchmark_AddInto-8      	   32036	     37421 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	6.695s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialTrainBatch_XOR-8              	  393706	      3067 ns/op	    5056 B/op	     102 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8   	    1464	    817915 ns/op	 1050161 B/op	      50 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/model	3.568s
```

## Expanded Raw Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMul-8                  	    7863	    152175 ns/op	   32816 B/op	       2 allocs/op
Benchmark_MatMulInto-8              	    7764	    151347 ns/op	       0 B/op	       0 allocs/op
Benchmark_Clone-8                   	   42266	     29435 ns/op	  524337 B/op	       2 allocs/op
Benchmark_Values-8                  	   37258	     28230 ns/op	  524288 B/op	       1 allocs/op
Benchmark_Add-8                     	   19386	     61380 ns/op	  524336 B/op	       2 allocs/op
Benchmark_AddInto-8                 	   31903	     37881 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddInPlace-8              	   31879	     37539 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScaledInPlace-8        	   38822	     30940 ns/op	       0 B/op	       0 allocs/op
Benchmark_Subtract-8                	   21842	     59561 ns/op	  524336 B/op	       2 allocs/op
Benchmark_MultiplyElements-8        	   19341	     63504 ns/op	  524336 B/op	       2 allocs/op
Benchmark_DivideElements-8          	   17095	     70372 ns/op	  524336 B/op	       2 allocs/op
Benchmark_AddScalar-8               	   23730	     50638 ns/op	  524337 B/op	       2 allocs/op
Benchmark_MultiplyScalar-8          	   20983	     48706 ns/op	  524337 B/op	       2 allocs/op
Benchmark_MultiplyScalarInPlace-8   	   66152	     18382 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideScalar-8            	   24816	     51142 ns/op	  524337 B/op	       2 allocs/op
Benchmark_Transpose-8               	   14635	     82410 ns/op	  262192 B/op	       2 allocs/op
Benchmark_TransposeInto-8           	   15693	     76401 ns/op	       0 B/op	       0 allocs/op
Benchmark_RowSums-8                 	   21356	     50810 ns/op	    2048 B/op	       1 allocs/op
Benchmark_ColumnSums-8              	   41066	     31108 ns/op	    2048 B/op	       1 allocs/op
Benchmark_ColumnSumsInto-8          	   35754	     33388 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddRowVectorInPlace-8     	   33020	     36505 ns/op	       0 B/op	       0 allocs/op
Benchmark_Apply-8                   	    9895	    122847 ns/op	  524336 B/op	       2 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	37.391s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/layer
cpu: Apple M3
Benchmark_DenseForward_XOR-8            	 7876819	       141.1 ns/op	     288 B/op	       4 allocs/op
Benchmark_DenseForward_MediumBatch-8    	    7555	    161924 ns/op	   98400 B/op	       4 allocs/op
Benchmark_DenseBackward_XOR-8           	 3775456	       317.4 ns/op	     528 B/op	      10 allocs/op
Benchmark_DenseBackward_MediumBatch-8   	    3698	    325237 ns/op	   99056 B/op	      10 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/layer	5.435s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialTrainBatch_XOR-8              	  387598	      3090 ns/op	    5056 B/op	     102 allocs/op
Benchmark_SequentialFit_XOR-8                     	  280663	      4391 ns/op	    7672 B/op	     149 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8   	    1446	    829006 ns/op	 1050163 B/op	      50 allocs/op
Benchmark_SequentialFit_SyntheticDense-8          	    1015	   1193491 ns/op	 2171848 B/op	     295 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/model	6.195s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/optimizer
cpu: Apple M3
Benchmark_SGDUpdate_SteadyState-8        	  278130	      4290 ns/op	       0 B/op	       0 allocs/op
Benchmark_MomentumUpdate_SteadyState-8   	  169201	      7090 ns/op	       0 B/op	       0 allocs/op
Benchmark_AdamUpdate_SteadyState-8       	   54430	     21709 ns/op	  177184 B/op	      44 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/optimizer	5.046s
```

## V2 Loss Allocation Reduction

Captured on July 6, 2026.

### Commands

```sh
go test ./loss
go test ./loss -run '^$' -bench=. -benchmem
go test ./...
go test ./model -run '^$' -bench=Sequential -benchmem
```

### Allocation Audit

The loss allocation sources were direct code paths:

| Package | Path | Finding | Change |
| --- | --- | --- | --- |
| `loss` | `matrixValuePair` | Every `Value` and `Gradient` call copied predictions and targets through `Matrix.Values`, producing two temporary slices before any loss-specific work. | Replaced it with shape validation that does not copy values. |
| `loss` | `MeanSquaredError.Gradient` | Allocated a gradient slice, filled it, then copied it again through `matrix.FromSlice`. | Allocate the returned matrix once, write `predictions - targets` into it with `SubtractInto`, and scale in place. |
| `loss` | Cross-entropy gradients | Allocated prediction and target copies, a gradient slice, and a copied result matrix. | Allocate only the returned gradient matrix and fill it through matrix pair destination iteration. |
| `matrix` | Loss pair iteration | `At` and `Set` avoid storage exposure but validate on every element, which removed allocations at the cost of large timing regressions. | Added documented `Pairwise` and `PairwiseInto` helpers that validate once, keep storage private, and support the loss hot paths. |

The `loss.Loss` interface is unchanged. Matrix ownership remains unchanged:
constructors and `Values` still copy, and the new pair helpers pass values to
callbacks without exposing mutable matrix storage.

### Loss Before

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/loss
cpu: Apple M3
Benchmark_MeanSquaredErrorValue_Small-8                   	36446523	        33.10 ns/op	      64 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorValue_MediumBatch-8             	  308329	      3966 ns/op	   32768 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradient_Small-8                	17035028	        59.95 ns/op	     144 B/op	       4 allocs/op
Benchmark_MeanSquaredErrorGradient_MediumBatch-8          	  267528	      4432 ns/op	   65584 B/op	       5 allocs/op
Benchmark_BinaryCrossEntropyValue_Small-8                 	17457644	        68.84 ns/op	      64 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyValue_MediumBatch-8           	  991759	      1221 ns/op	    2048 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradient_Small-8              	18061419	        66.99 ns/op	     144 B/op	       4 allocs/op
Benchmark_BinaryCrossEntropyGradient_MediumBatch-8        	 2427573	       487.4 ns/op	    4144 B/op	       5 allocs/op
Benchmark_CategoricalCrossEntropyValue_Small-8            	14054724	        86.31 ns/op	     192 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyValue_MediumBatch-8      	  230536	      5216 ns/op	   32768 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradient_Small-8         	10818448	       112.1 ns/op	     432 B/op	       5 allocs/op
Benchmark_CategoricalCrossEntropyGradient_MediumBatch-8   	  195165	      5754 ns/op	   65584 B/op	       5 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/loss	16.459s
```

### Loss After

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/loss
cpu: Apple M3
Benchmark_MeanSquaredErrorValue_Small-8                   	60705260	        19.66 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorValue_MediumBatch-8             	  430102	      2801 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorGradient_Small-8                	22906390	        52.28 ns/op	      80 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradient_MediumBatch-8          	  486931	      2457 ns/op	   16432 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyValue_Small-8                 	19853510	        58.98 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyValue_MediumBatch-8           	  914274	      1319 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyGradient_Small-8              	16259473	        72.83 ns/op	      80 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradient_MediumBatch-8        	 1548943	       773.4 ns/op	    1072 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyValue_Small-8            	18233282	        65.33 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyValue_MediumBatch-8      	  241802	      4970 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyGradient_Small-8         	10244148	       115.0 ns/op	     144 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradient_MediumBatch-8   	  142140	      8383 ns/op	   16432 B/op	       2 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/loss	16.709s
```

### Model Loss Re-run

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialTrainBatch_XOR-8              	  468099	      2482 ns/op	    2296 B/op	      58 allocs/op
Benchmark_SequentialFit_XOR-8                     	  349948	      3548 ns/op	    3760 B/op	      88 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8   	    1502	    794022 ns/op	  147926 B/op	      10 allocs/op
Benchmark_SequentialFit_SyntheticDense-8          	    1042	   1131147 ns/op	 1015716 B/op	     141 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/model	6.121s
```

### Interpretation

Loss value paths now report zero allocations. Gradient paths now allocate only
the returned matrix, reducing medium-batch gradients from 5 allocations to 2
and cutting medium-batch gradient bytes by 50% to 75%.

Most measured loss paths are also faster after removing copies. Binary
cross-entropy medium value and cross-entropy gradients retain small timing
costs from validation and callback dispatch, but they remove the temporary
copies without changing loss validation, clamping, or the public loss API.

Model benchmarks that exercise losses improved allocation counts materially:
`SequentialTrainBatch_XOR` dropped from the v1 baseline of 102 allocs/op to 58,
and `SequentialTrainBatch_SyntheticDense` dropped from 50 allocs/op to 10.

## V2 Adam Allocation Reduction

Captured on July 6, 2026.

### Commands

```sh
go test ./optimizer -run '^$' -bench=Update -benchmem
go test ./...
go test ./model -run '^$' -bench='(XOR|Sequential)' -benchmem
```

### Allocation Audit

Adam's steady-state allocation sources were direct code paths:

| Package | Path | Finding | Change |
| --- | --- | --- | --- |
| `optimizer` | `parameterValues` | Copied parameter values and gradients through `Matrix.Values` on every Adam update. | Removed the helper from the Adam path and read owned matrices directly through a narrow matrix update helper. |
| `optimizer` | `matrixValues` | Copied first and second moment matrices through `Matrix.Values` before every update. | Adam now passes owned moment matrices to the matrix helper, which updates moment storage in place. |
| `optimizer` | `copyMatrixValues` | Rebuilt each updated values, first-moment, and second-moment matrix through `matrix.FromSlice`, then copied it back with `CopyFrom`. | Removed the helper from the Adam path; no temporary result matrices are created during steady-state updates. |
| `optimizer` | `stateFor` | The local `adamState` value escaped before the cache-hit return, causing one small heap allocation per parameter even when state was reused. | Moved state construction behind the map miss so cache hits return without allocation. |
| `matrix` | Adam elementwise loop | Existing public operations could not express the four-matrix Adam update without copies or per-element `At`/`Set` validation. | Added `AdamUpdateInPlace`, which validates shapes once, rejects aliasing, keeps matrix storage private, and updates values plus moment matrices in one matrix-owned loop. |

The optimizer API is unchanged. The only new public surface is the narrow
matrix-owned Adam update helper; it does not expose mutable matrix storage.

### Optimizer Before

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/optimizer
cpu: Apple M3
Benchmark_SGDUpdate_SteadyState-8        	  266364	      4310 ns/op	       0 B/op	       0 allocs/op
Benchmark_MomentumUpdate_SteadyState-8   	  169041	      7106 ns/op	       0 B/op	       0 allocs/op
Benchmark_AdamUpdate_SteadyState-8       	   54945	     21987 ns/op	  177184 B/op	      44 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/optimizer	5.163s
```

### Optimizer After

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/optimizer
cpu: Apple M3
Benchmark_SGDUpdate_SteadyState-8        	  279056	      4302 ns/op	       0 B/op	       0 allocs/op
Benchmark_MomentumUpdate_SteadyState-8   	  168210	      7088 ns/op	       0 B/op	       0 allocs/op
Benchmark_AdamUpdate_SteadyState-8       	  136090	      9056 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/optimizer	4.987s
```

### Model Adam Re-run

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialTrainBatch_XOR-8              	  711825	      2554 ns/op	     672 B/op	      14 allocs/op
Benchmark_SequentialFit_XOR-8                     	  370011	      2762 ns/op	    2136 B/op	      44 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8   	    1477	    816951 ns/op	  147930 B/op	      10 allocs/op
Benchmark_SequentialFit_SyntheticDense-8          	    1018	   1139725 ns/op	 1015718 B/op	     141 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/model	7.288s
```

### Interpretation

Adam steady-state updates now meet the v2 allocation target with zero `B/op`
and zero `allocs/op`, down from 177184 `B/op` and 44 `allocs/op` before the
change. Adam update time also improved from 21987 ns/op to 9056 ns/op in the
focused optimizer benchmark.

SGD and Momentum remain allocation-free. The model benchmarks that use Adam
also retain the lower post-loss allocation profile while improving the XOR fit
and training paths.

## V2 Data Batch Allocation Reduction

Captured on July 7, 2026.

### Commands

```sh
go test ./data
go test ./data -run '^$' -bench='(DatasetBatches|BatchInputs|BatchTargets)' -benchmem
go test ./...
go test ./model -run '^$' -bench=SequentialFit -benchmem
```

### Allocation Audit

The data allocation sources were direct copy paths:

| Package | Path | Finding | Change |
| --- | --- | --- | --- |
| `data` | `Dataset.Batches` | The result slice grew through append even though the batch count is known after batch-size validation. | Preallocated the batch slice to the exact expected capacity. |
| `data` | `matrixRows` | Each batch row selection copied the full source matrix through `Values`, allocated a row result slice, then copied again through `matrix.FromSlice`. | Replaced the helper with `Matrix.SelectRows`, which validates once and copies selected rows directly into the returned owned matrix without exposing storage. |
| `data` | `newBatch` | `Dataset.Batches` created owned row-selected matrices, then `newBatch` cloned both matrices again before storing them. | Narrowed the unexported batch constructor to store data-package-owned row-selected matrices directly. |
| `data` | `Batch.Inputs`, `Batch.Targets` | Accessors clone stored matrices on every call. | Kept unchanged because returning copies is the public copy-protection contract. |

`Matrix.SelectRows` is the only new public API. It returns copied row data in
the requested order, supports repeated indexes, and does not expose mutable
matrix storage. The helper is used by data row selection for both batching and
splitting.

### Data Before

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/data
cpu: Apple M3
Benchmark_DatasetBatches_Unshuffled-8   	    2514	    398742 ns/op	 6237646 B/op	     211 allocs/op
Benchmark_DatasetBatches_Shuffled-8     	    3048	    421041 ns/op	 6237673 B/op	     211 allocs/op
Benchmark_BatchInputs-8                 	 1000000	      1159 ns/op	   16432 B/op	       2 allocs/op
Benchmark_BatchTargets-8                	 3371464	       349.6 ns/op	    4144 B/op	       2 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/data	5.374s
```

### Data After

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/data
cpu: Apple M3
Benchmark_DatasetBatches_Unshuffled-8   	   41972	     42755 ns/op	  337796 B/op	      82 allocs/op
Benchmark_DatasetBatches_Shuffled-8     	   31246	     35303 ns/op	  337795 B/op	      82 allocs/op
Benchmark_BatchInputs-8                 	 1000000	      1296 ns/op	   16432 B/op	       2 allocs/op
Benchmark_BatchTargets-8                	 3455426	       360.7 ns/op	    4144 B/op	       2 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/data	7.184s
```

### Fit Re-run

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialFit_XOR-8              	  447300	      2600 ns/op	    1784 B/op	      37 allocs/op
Benchmark_SequentialFit_SyntheticDense-8   	    1015	   1137451 ns/op	  720420 B/op	     109 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/model	3.571s
```

### Interpretation

`Dataset.Batches` now avoids the repeated full-matrix source copies and the
second batch-storage clone. The unshuffled benchmark drops from 6237646 B/op
and 211 allocs/op to 337796 B/op and 82 allocs/op. The shuffled benchmark drops
from 6237673 B/op and 211 allocs/op to 337795 B/op and 82 allocs/op.

Batch accessor benchmarks remain at 2 allocations per call because those
methods intentionally return defensive matrix copies. Additional tests cover
`SelectRows` copy behavior and verify that mutating returned dataset or batch
matrices does not mutate stored samples.

The fit re-run reflects the lower batching cost: `SequentialFit_XOR` reports
1784 B/op and 37 allocs/op, and `SequentialFit_SyntheticDense` reports
720420 B/op and 109 allocs/op.
