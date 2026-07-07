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

## V2 Sequential Fit Allocation Reduction

Captured on July 7, 2026.

### Commands

```sh
go test ./model -run '^$' -bench=Sequential -benchmem
go test ./data ./model
go test ./...
go test ./model -run '^$' -bench=Sequential -benchmem
```

### Allocation Audit

The fit allocation sources were direct copy paths in model orchestration:

| Package | Path | Finding | Change |
| --- | --- | --- | --- |
| `model` | `validateFitDataset` | Fit validation cloned full dataset inputs and targets through `Dataset.Inputs` and `Dataset.Targets` before every fit call. | Added `Dataset.Validate` and changed fit validation to validate dataset-owned matrices without copying them. |
| `model` | `trainFitEpoch` | Each mini-batch already owned copied row-selected matrices, then `Batch.Inputs` and `Batch.Targets` cloned those matrices again before `TrainBatch`. | Added `Batch.InputsInto` and `Batch.TargetsInto`, then reused fit-owned scratch matrices for batch training inputs and targets. |
| `model` | `evaluateFitDataset` | Each training and validation evaluation cloned full dataset inputs and targets every epoch. | Added `Dataset.InputsInto` and `Dataset.TargetsInto`, then reused separate fit-owned scratch matrices for training and validation evaluation. |
| `data` | Copy-into accessors | Existing public accessors intentionally returned defensive copies, with no destination form available for repeated copy-protected reads. | Added destination accessors that copy into caller-owned matrices and keep dataset and batch storage private. |

Remaining allocations are inherited from batch construction, activation output
matrices, loss gradient matrices, optimizer/model bookkeeping, history growth,
and user callback or accuracy paths. `Dataset.Batches` still creates owned
mini-batch matrices each epoch so training order, shuffle behavior, and mutation
protection remain unchanged.

### Model Before

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialTrainBatch_XOR-8              	  734689	      1564 ns/op	     672 B/op	      14 allocs/op
Benchmark_SequentialFit_XOR-8                     	  505905	      2404 ns/op	    1784 B/op	      37 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8   	    1543	    783709 ns/op	  147919 B/op	      10 allocs/op
Benchmark_SequentialFit_SyntheticDense-8          	    1118	   1065715 ns/op	  720413 B/op	     109 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/model	5.856s
```

### Model After

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialTrainBatch_XOR-8              	  764514	      1551 ns/op	     672 B/op	      14 allocs/op
Benchmark_SequentialFit_XOR-8                     	  512886	      2345 ns/op	    1592 B/op	      33 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8   	    1560	    765376 ns/op	  147916 B/op	      10 allocs/op
Benchmark_SequentialFit_SyntheticDense-8          	    1144	   1044033 ns/op	  634013 B/op	      93 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/model	5.943s
```

### Interpretation

The fit-only scratch reuse leaves `SequentialTrainBatch` allocation counts
unchanged, as expected. `SequentialFit_XOR` drops from 1784 B/op and 37
allocs/op to 1592 B/op and 33 allocs/op. `SequentialFit_SyntheticDense` drops
from 720413 B/op and 109 allocs/op to 634013 B/op and 93 allocs/op.

The public copy-protection contracts remain intact: `Dataset.Inputs`,
`Dataset.Targets`, `Batch.Inputs`, and `Batch.Targets` still return defensive
copies. The new destination accessors copy into caller-owned matrices and do
not expose dataset or batch storage. Tests cover copy-into behavior, wrong-shape
errors, validation data, callback errors, early stopping, shuffle
reproducibility, and training-mode restoration after evaluation prediction,
loss, and accuracy errors.

## V2 Allocation Regression Checks

Captured on July 7, 2026.

### Commands

```sh
go test ./matrix ./layer ./loss ./optimizer ./data
go test ./...
go test ./matrix ./layer ./loss ./optimizer ./data ./model -run '^$' -bench=. -benchmem
```

### AllocsPerRun Coverage

Focused allocation tests were added only for counts tied to stable ownership or
steady-state scratch behavior:

| Package | Covered allocation count | Rationale |
| --- | --- | --- |
| `matrix` | Destination and in-place matrix operations, reduction destinations, `ValuesInto`, `CopyValuesFrom`, `Pairwise`, `PairwiseInto`, and `AdamUpdateInPlace` assert zero allocations. `Values` asserts at most 1 allocation. `Clone` and `SelectRows` assert at most 2 allocations. | Destination helpers mutate caller-owned storage. Copy-returning methods allocate owned results by contract. |
| `layer` | Dense forward/backward, dropout training forward/backward, and batch-normalization training forward/backward assert zero steady-state allocations after warm-up. | These paths intentionally retain stable-shape scratch storage. |
| `loss` | Mean squared error, binary cross entropy, and categorical cross entropy value paths assert zero allocations. Their gradient paths assert at most 2 allocations. | Value paths should not copy matrix values. Gradient paths allocate only the returned owned matrix. |
| `optimizer` | SGD, Momentum, and Adam update paths assert zero steady-state allocations after state warm-up. | Optimizer updates operate on owned parameter and state matrices. |
| `data` | `Dataset.Batches` asserts the small-shape batch construction ceiling, destination accessors assert zero allocations, and copy-returning accessors assert at most 2 allocations. | Batches still create owned mini-batch matrices, while destination accessors copy into caller-owned storage. |

### Benchmark-Only Rationale

Model-level `SequentialTrainBatch` and `SequentialFit` allocation counts remain
benchmark-only. They are composite orchestration measurements that include layer
outputs, activation-owned result matrices, loss gradients, data batching,
history growth, optimizer bookkeeping, and optional user callback or metric
paths. The current counts are deterministic enough for benchmark tracking, but
too broad for stable `AllocsPerRun` assertions.

Activation forward/backward allocation counts also remain benchmark-only. The
current allocation count is the returned owned matrix, while the v2 scratch
reuse work removed helper-copy allocations. A future activation destination API
would need its own allocation contract before test assertions are useful.

Allocating matrix arithmetic methods such as `Add`, `Subtract`, `MatMul`, and
`Apply` remain benchmark-only because their public behavior is to return owned
result matrices. The destination and in-place variants are the stable
allocation-free contracts covered by tests.

### Raw Benchmark Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMul-8                     	    8061	    146352 ns/op	   32816 B/op	       2 allocs/op
Benchmark_MatMulInto-8                 	    8179	    147650 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulLeftTransposeInto-8    	    8378	    147563 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeInto-8   	    7489	    160210 ns/op	       0 B/op	       0 allocs/op
Benchmark_Clone-8                      	   41601	     30140 ns/op	  524340 B/op	       2 allocs/op
Benchmark_Values-8                     	   37743	     28721 ns/op	  524288 B/op	       1 allocs/op
Benchmark_Add-8                        	   21734	     55032 ns/op	  524336 B/op	       2 allocs/op
Benchmark_AddInto-8                    	   32493	     36778 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddInPlace-8                 	   32482	     37217 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScaledInPlace-8           	   39408	     31231 ns/op	       0 B/op	       0 allocs/op
Benchmark_Subtract-8                   	   20536	     57162 ns/op	  524336 B/op	       2 allocs/op
Benchmark_SubtractInto-8               	   32426	     36908 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyElements-8           	   21898	     56346 ns/op	  524336 B/op	       2 allocs/op
Benchmark_MultiplyElementsInto-8       	   32445	     37004 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideElements-8             	   18868	     61161 ns/op	  524336 B/op	       2 allocs/op
Benchmark_DivideElementsInto-8         	   27428	     43843 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScalar-8                  	   26308	     46909 ns/op	  524337 B/op	       2 allocs/op
Benchmark_AddScalarInto-8              	   45715	     26249 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalar-8             	   24637	     48921 ns/op	  524337 B/op	       2 allocs/op
Benchmark_MultiplyScalarInto-8         	   45423	     26243 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInPlace-8      	   67434	     18459 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideScalar-8               	   24734	     48439 ns/op	  524337 B/op	       2 allocs/op
Benchmark_DivideScalarInto-8           	   45594	     26300 ns/op	       0 B/op	       0 allocs/op
Benchmark_Transpose-8                  	   15792	     75702 ns/op	  262192 B/op	       2 allocs/op
Benchmark_TransposeInto-8              	   16440	     72873 ns/op	       0 B/op	       0 allocs/op
Benchmark_RowSums-8                    	   22066	     51427 ns/op	    2048 B/op	       1 allocs/op
Benchmark_RowSumsInto-8                	   26503	     44660 ns/op	       0 B/op	       0 allocs/op
Benchmark_ColumnSums-8                 	   41713	     28746 ns/op	    2048 B/op	       1 allocs/op
Benchmark_ColumnSumsInto-8             	   36286	     32906 ns/op	       0 B/op	       0 allocs/op
Benchmark_AccumulateColumnSumsInto-8   	   36218	     32838 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddRowVectorInPlace-8        	   33300	     36190 ns/op	       0 B/op	       0 allocs/op
Benchmark_Apply-8                      	   10000	    116544 ns/op	  524337 B/op	       2 allocs/op
Benchmark_ApplyInto-8                  	   10000	    100434 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	51.752s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/layer
cpu: Apple M3
Benchmark_DenseForward_XOR-8                                 	11374191	        90.71 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_MediumBatch-8                         	    7696	    155500 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_XOR-8                                	 8020212	       147.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_MediumBatch-8                        	    3770	    317399 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForward_MediumBatch-8                    	   23013	     51669 ns/op	   65584 B/op	       2 allocs/op
Benchmark_ActivationBackward_MediumBatch-8                   	   19933	     60167 ns/op	   65584 B/op	       2 allocs/op
Benchmark_DropoutForwardTraining_MediumBatch-8               	   14576	     79776 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutBackwardTraining_MediumBatch-8              	  258130	      4654 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationForwardTraining_MediumBatch-8    	   49881	     23894 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationBackwardTraining_MediumBatch-8   	   54536	     22727 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/layer	14.766s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/loss
cpu: Apple M3
Benchmark_MeanSquaredErrorValue_Small-8                   	59971922	        19.71 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorValue_MediumBatch-8             	  436986	      2751 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorGradient_Small-8                	22806406	        50.69 ns/op	      80 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradient_MediumBatch-8          	  496056	      2387 ns/op	   16432 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyValue_Small-8                 	19938950	        59.09 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyValue_MediumBatch-8           	  927364	      1297 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyGradient_Small-8              	16538521	        70.70 ns/op	      80 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradient_MediumBatch-8        	 1575759	       752.6 ns/op	    1072 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyValue_Small-8            	18477705	        64.71 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyValue_MediumBatch-8      	  245866	      4873 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyGradient_Small-8         	10404255	       113.0 ns/op	     144 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradient_MediumBatch-8   	  132272	      8853 ns/op	   16432 B/op	       2 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/loss	16.547s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/optimizer
cpu: Apple M3
Benchmark_SGDUpdate_SteadyState-8        	  283598	      4238 ns/op	       0 B/op	       0 allocs/op
Benchmark_MomentumUpdate_SteadyState-8   	  169740	      6988 ns/op	       0 B/op	       0 allocs/op
Benchmark_AdamUpdate_SteadyState-8       	  139032	      8632 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/optimizer	4.970s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/data
cpu: Apple M3
Benchmark_DatasetBatches_Unshuffled-8   	   42610	     27545 ns/op	  337792 B/op	      82 allocs/op
Benchmark_DatasetBatches_Shuffled-8     	   38046	     31541 ns/op	  337792 B/op	      82 allocs/op
Benchmark_BatchInputs-8                 	 1000000	      1103 ns/op	   16432 B/op	       2 allocs/op
Benchmark_BatchTargets-8                	 3591530	       338.0 ns/op	    4144 B/op	       2 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/data	5.818s
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialTrainBatch_XOR-8              	  727297	      1646 ns/op	     672 B/op	      14 allocs/op
Benchmark_SequentialFit_XOR-8                     	  489129	      2480 ns/op	    1592 B/op	      33 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8   	    1504	    786640 ns/op	  147925 B/op	      10 allocs/op
Benchmark_SequentialFit_SyntheticDense-8          	    1088	   1141725 ns/op	  634016 B/op	      93 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/model	6.026s
```

### Interpretation

The stable allocation contracts now have focused package-local
`testing.AllocsPerRun` coverage. Matrix destination operations, dense/dropout/
batch-normalization steady-state paths, loss value paths, and optimizer updates
continue to report zero allocations. Copy-returning APIs continue to allocate
owned results as part of their mutation-protection contracts.

The remaining model and activation allocation counts are documented as
benchmark-only. They should continue to be watched in the benchmark history
rather than enforced through narrow allocation assertions.

## V2 SIMD Dot-Product Session

Captured on July 7, 2026.

The SIMD implementation approach is documented in [`docs/simd.md`](docs/simd.md).
This session added a private dot-product kernel boundary and benchmark coverage.
The active `arm64 && !purego` and `amd64 && !purego` entry points currently
route to the scalar reference until architecture-specific assembly can be
implemented and benchmarked. `arm64` is the primary development and measurement
path; `amd64` is also a supported target. Other architectures use the pure Go
fallback.

### Commands

```sh
go test ./matrix -run '^$' -bench=MatMul -benchmem
go test ./...
go test ./matrix -run '^$' -bench='(DotProduct|MatMul)' -benchmem
```

### Before Matrix Multiplication Benchmark

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_MatMul-8                     	    7984	    151200 ns/op	   32817 B/op	       2 allocs/op
Benchmark_MatMulInto-8                 	    8124	    147964 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulLeftTransposeInto-8    	    8020	    148776 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeInto-8   	    7540	    161584 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	6.072s
```

### Test Output

```text
ok  	github.com/itsmontoya/neuralnetwork/activation	(cached)
ok  	github.com/itsmontoya/neuralnetwork/data	(cached)
ok  	github.com/itsmontoya/neuralnetwork/examples/heart	(cached)
ok  	github.com/itsmontoya/neuralnetwork/examples/multiclass	(cached)
ok  	github.com/itsmontoya/neuralnetwork/examples/regression	(cached)
ok  	github.com/itsmontoya/neuralnetwork/examples/toycode	(cached)
ok  	github.com/itsmontoya/neuralnetwork/examples/xor	(cached)
ok  	github.com/itsmontoya/neuralnetwork/internal/testutil	(cached)
ok  	github.com/itsmontoya/neuralnetwork/layer	(cached)
ok  	github.com/itsmontoya/neuralnetwork/loss	(cached)
ok  	github.com/itsmontoya/neuralnetwork/matrix	0.231s
ok  	github.com/itsmontoya/neuralnetwork/metric	(cached)
ok  	github.com/itsmontoya/neuralnetwork/model	(cached)
ok  	github.com/itsmontoya/neuralnetwork/optimizer	(cached)
```

### Dot Product and Matrix Shape Benchmark Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_DotProduct/Length1-8                 	1000000000	         0.7463 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length2-8                 	802354690	         1.493 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length3-8                 	638188676	         1.865 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4-8                 	575595699	         2.171 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length5-8                 	523495780	         2.316 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length31-8                	95975046	        12.55 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length33-8                	90624741	        13.13 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length64-8                	33879961	        35.32 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length257-8               	 5999739	       200.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4096-8              	  298564	      4020 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4099-8              	  298164	      4029 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8             	   18268	     65323 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small2x2-8         	100000000	        12.25 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small4x4-8         	21444686	        56.73 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Medium64x64-8      	    7672	    155951 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Large128x256x128-8 	     332	   3606813 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven17x33x19-8   	  234943	      5050 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven63x65x31-8   	   15837	     75506 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMul-8                                            	    7910	    150485 ns/op	   32816 B/op	       2 allocs/op
Benchmark_MatMulInto-8                                        	    8012	    149653 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulLeftTransposeInto-8                           	    7983	    149943 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeInto-8                          	    7430	    161655 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulShapes/Small2x2-8                             	27131545	        41.81 ns/op	      80 B/op	       2 allocs/op
Benchmark_MatMulShapes/Small4x4-8                             	11417338	       103.7 ns/op	     176 B/op	       2 allocs/op
Benchmark_MatMulShapes/Medium64x64-8                          	    7906	    157169 ns/op	   32816 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8                     	     471	   2501597 ns/op	  131120 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven17x33x19-8                       	  171226	      6976 ns/op	    2736 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven63x65x31-8                       	   15445	     77652 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small2x2-8           	54865818	        21.65 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small4x4-8           	14661009	        80.94 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Medium64x64-8        	    7676	    158114 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8   	     354	   3363504 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven17x33x19-8     	  173622	      6894 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven63x65x31-8     	   15408	     77961 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	46.314s
```

### Interpretation

The private dot-product boundary reports zero allocations across the small,
medium, large, and uneven lengths defined in `docs/simd.md`. Correctness tests
compare the active kernel to the pure Go reference for empty input, sizes below
and at the expected vector width, multi-vector lengths, scalar tails, uneven
lengths, and `Inf`/`NaN` inputs.

No SIMD assembly was integrated in this session. The checked-in `arm64` and
`amd64` architecture entry points deliberately use the scalar reference until
each architecture has benchmark evidence proving an assembly kernel is stable.
The arm64 path is no longer treated as fallback-only; it is the first target for
future kernel implementation because this repository's current evidence is
captured on `darwin/arm64`.

Production matrix multiplication remains unchanged. A test-only
right-transpose candidate that calls the scalar dot-product kernel was faster
for `2x2`, `4x4`, and `17x33 * 33x19`, approximately even for `64x64` and
`63x65 * 65x31`, and slower for `128x256 * 256x128`. That mixed scalar-fallback
result is not stable enough to integrate without real arm64 SIMD evidence, so
the existing pure Go multiplication loops remain the fallback for all public
matrix methods.

## V2 SIMD Elementwise Session

Captured on July 7, 2026.

This session added private elementwise kernel boundaries for the candidate
operations defined in [`docs/simd.md`](docs/simd.md). On `arm64 && !purego`
and `amd64 && !purego`, `AddInto`, `SubtractInto`, `AddScaledInPlace`, and
`AddScalarInto` use small Go assembly kernels. The arm64 kernels are built from
`VFMLA`; the amd64 kernels use baseline SSE2 instructions. Small inputs below
16 elements use the pure Go helper to avoid assembly call overhead. Other
architectures and `purego` builds use the pure Go helpers.

`MultiplyElementsInto`, `MultiplyScalarInto`, and `MultiplyScalarInPlace` remain
pure Go on every architecture. The current Go arm64 assembler has `VFMLA`
support, but direct vector float multiply is still marked TODO in the assembler
testdata. Baseline amd64 has packed multiply, but the standalone multiply
candidates remain pure Go so arm64 and amd64 expose the same integrated
operation set for v2. Using fused multiply-add with a zero accumulator would
not be a direct multiply kernel, so the multiply candidates were rejected for
v2 SIMD integration until direct vector multiply support or an approved
generated kernel is available.

### Commands

```sh
go test ./matrix -run 'Test_Elementwise|Test_Scalar|Test_Divide|Test_NilMatrixValidation'
go test ./matrix -run '^$' -bench='(AddInto|SubtractInto|MultiplyElementsInto|AddScaledInPlace|AddScalarInto|MultiplyScalarInto|MultiplyScalarInPlace)' -benchmem
go test ./matrix
go test ./matrix -run '^$' -bench='ElementwiseCandidates' -benchmem
go test ./matrix -run '^$' -bench='(AddInto|SubtractInto|MultiplyElementsInto|AddScaledInPlace|AddScalarInto|MultiplyScalarInto|MultiplyScalarInPlace)' -benchmem
env GOOS=darwin GOARCH=amd64 go test ./matrix
env GOOS=darwin GOARCH=amd64 go test ./matrix -run '^$' -bench='ElementwiseCandidates/.*/(AddInto|SubtractInto|AddScaledInPlace|AddScalarInto)' -benchmem
go test ./...
go test ./matrix -tags=purego
go fmt ./...
go vet ./...
go test ./... -race
```

### Before Focused Elementwise Benchmark

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_AddInto-8                 	   29370	     36470 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScaledInPlace-8        	   39426	     30043 ns/op	       0 B/op	       0 allocs/op
Benchmark_SubtractInto-8            	   32694	     36648 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyElementsInto-8    	   32392	     36755 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScalarInto-8           	   45784	     26178 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInto-8      	   45817	     26162 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInPlace-8   	   67922	     17724 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	10.679s
```

### Candidate Benchmark Output

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Pure-8         	615545736	         1.824 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Active-8       	757675608	         1.589 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Pure-8    	752978580	         1.594 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Active-8  	752052477	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyElementsInto/Pure-8         	745339683	         1.594 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyElementsInto/Active-8       	749353878	         1.597 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Pure-8             	709598876	         1.633 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Active-8           	724579705	         1.738 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Pure-8                	751727408	         1.595 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Active-8              	751051861	         1.596 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInto/Pure-8           	749190134	         1.599 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInto/Active-8         	750656434	         1.598 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInPlace/Pure-8        	894146409	         1.324 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInPlace/Active-8      	903404422	         1.326 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Pure-8                      	452782275	         2.650 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Active-8                    	463658509	         2.390 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Pure-8                 	500419970	         2.399 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Active-8               	501463558	         2.395 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyElementsInto/Pure-8         	500520420	         2.395 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyElementsInto/Active-8       	500342072	         2.395 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Pure-8             	501331662	         2.392 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Active-8           	501243100	         2.400 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Pure-8                	500894304	         2.500 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Active-8              	478873608	         2.412 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInto/Pure-8           	496693450	         2.402 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInto/Active-8         	499393876	         2.402 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInPlace/Pure-8        	461753426	         2.572 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInPlace/Active-8      	467450815	         2.561 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Pure-8                      	393563271	         3.093 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Active-8                    	446951371	         2.716 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Pure-8                 	406995460	         2.799 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Active-8               	441594861	         2.723 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyElementsInto/Pure-8         	441342784	         2.783 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyElementsInto/Active-8       	437653986	         2.795 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Pure-8             	432388867	         2.795 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Active-8           	436857681	         2.792 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Pure-8                	431382282	         2.903 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Active-8              	426077038	         2.801 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInto/Pure-8           	429000493	         2.908 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInto/Active-8         	423744897	         2.817 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInPlace/Pure-8        	425879388	         2.815 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInPlace/Active-8      	426082838	         2.809 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Pure-8                      	357109201	         3.614 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Active-8                    	405730035	         3.192 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Pure-8                 	381151123	         3.264 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Active-8               	392952559	         3.265 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyElementsInto/Pure-8         	373300801	         3.363 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyElementsInto/Active-8       	369831792	         3.416 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Pure-8             	387603366	         3.210 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Active-8           	365183274	         3.348 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Pure-8                	356842743	         3.539 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Active-8              	373940599	         3.272 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInto/Pure-8           	352139368	         3.524 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInto/Active-8         	398618454	         3.340 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInPlace/Pure-8        	386750517	         3.117 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInPlace/Active-8      	387089125	         3.107 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Pure-8                 	   45559	     26314 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Active-8               	  105346	     10454 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Pure-8            	   45468	     26306 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Active-8          	  104346	     10446 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Pure-8    	   45652	     26327 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Active-8  	   45676	     26270 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Pure-8        	   65707	     18093 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Active-8      	  103530	     11544 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Pure-8           	   68457	     17521 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Active-8         	  191715	      6609 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInto/Pure-8      	   68037	     17896 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInto/Active-8    	   68194	     17513 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Pure-8   	   68451	     17535 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Active-8 	   68590	     17477 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Pure-8                	    2760	    429025 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8              	    5049	    236039 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Pure-8           	    2812	    431848 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Active-8         	    4940	    234843 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Pure-8   	    2802	    432028 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8 	    2775	    432660 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Pure-8       	    3885	    310177 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Active-8     	    6115	    190881 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Pure-8          	    4154	    282672 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Active-8        	    7560	    142029 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Pure-8     	    4203	    284781 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Active-8   	    4246	    283002 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Pure-8  	    4228	    280827 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8         	    4261	    286578 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Pure-8                            	 8831820	       135.4 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Active-8                          	33554466	        35.76 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Pure-8                       	 8865679	       135.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Active-8                     	33563773	        35.84 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyElementsInto/Pure-8               	 8887275	       134.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyElementsInto/Active-8             	 8871367	       134.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Pure-8                   	12132230	        97.47 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Active-8                 	32692796	        37.24 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Pure-8                      	12326745	        98.16 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Active-8                    	38978860	        31.32 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInto/Pure-8                 	12294776	        97.33 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInto/Active-8               	12339891	        97.23 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInPlace/Pure-8              	12608738	        94.46 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInPlace/Active-8            	12580406	        94.48 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Pure-8                          	   45667	     26266 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Active-8                        	  107318	     11802 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Pure-8                     	   45504	     26286 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Active-8                   	  105236	     10426 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyElementsInto/Pure-8             	   45624	     26295 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyElementsInto/Active-8           	   45574	     26272 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Pure-8                 	   65479	     18208 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Active-8               	  103299	     11594 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Pure-8                    	   68505	     17488 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Active-8                  	  150526	      8021 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInto/Pure-8               	   68566	     17716 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInto/Active-8             	   62533	     19241 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInPlace/Pure-8            	   68670	     17467 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInPlace/Active-8          	   68382	     17516 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	159.057s
```

### After Focused Elementwise Benchmark

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_AddInto-8                 	  101328	     12311 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScaledInPlace-8        	  104262	     11583 ns/op	       0 B/op	       0 allocs/op
Benchmark_SubtractInto-8            	  116160	     10230 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyElementsInto-8    	   45560	     26166 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScalarInto-8           	  175576	      9310 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInto-8      	   68827	     17407 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInPlace-8   	   68882	     17353 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	11.859s
```

### AMD64 Focused Candidate Benchmark

```text
goos: darwin
goarch: amd64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: VirtualApple @ 2.50GHz
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Pure-8         	561513948	         2.134 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Active-8       	558361473	         2.138 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Pure-8    	563063944	         2.132 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Active-8  	252130546	         4.756 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Pure-8         	373203712	         3.210 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Active-8       	373074633	         3.224 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Pure-8            	562193974	         2.136 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Active-8          	191898369	         6.242 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Pure-8                  	407978355	         2.936 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Active-8                	407988296	         2.941 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Pure-8             	408856746	         2.933 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Active-8           	218041082	         5.503 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Pure-8         	367857669	         3.257 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Active-8       	345961113	         3.466 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Pure-8            	408663088	         2.936 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Active-8          	168325471	         7.126 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Pure-8                  	364422770	         3.369 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Active-8                	367110470	         3.351 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Pure-8             	357921637	         3.450 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Active-8           	201060384	         5.980 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Pure-8         	338700481	         3.556 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Active-8       	312850123	         3.816 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Pure-8            	352677014	         3.397 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Active-8          	159259566	         7.644 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Pure-8                  	310303164	         3.945 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Active-8                	315202098	         4.213 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Pure-8             	311403328	         3.957 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Active-8           	181326721	         6.677 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Pure-8         	304981845	         3.933 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Active-8       	264542755	         4.553 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Pure-8            	335155000	         3.824 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Active-8          	145533478	         8.353 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Pure-8             	   37687	     31828 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Active-8           	   80376	     14905 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Pure-8        	   37716	     31870 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Active-8      	   80121	     14895 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Pure-8    	   58813	     20344 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Active-8  	   99526	     11982 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Pure-8       	   63614	     19406 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Active-8     	  111032	     11104 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Pure-8            	    2205	    524313 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8          	    4263	    272603 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Pure-8       	    2287	    524728 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Active-8     	    4203	    277864 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Pure-8   	    3666	    329640 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Active-8 	    5956	    203988 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Pure-8      	    3878	    314137 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Active-8    	    6549	    179994 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Pure-8               	 8801661	       135.9 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Active-8             	29873937	        39.30 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Pure-8          	 8858367	       135.5 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Active-8        	28904286	        41.13 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Pure-8      	12100423	        98.99 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Active-8    	24485610	        48.81 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Pure-8         	12120462	        98.95 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Active-8       	35498974	        33.63 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Pure-8             	   37356	     31780 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Active-8           	   79959	     14946 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Pure-8        	   37656	     31808 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Active-8      	   80200	     14869 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Pure-8    	   59012	     20367 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Active-8  	   99830	     12880 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Pure-8       	   63374	     18953 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Active-8     	  112591	     10645 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	94.922s
```

### Test Output

```text
ok  	github.com/itsmontoya/neuralnetwork/activation	0.159s
ok  	github.com/itsmontoya/neuralnetwork/data	0.262s
ok  	github.com/itsmontoya/neuralnetwork/examples/heart	0.329s
ok  	github.com/itsmontoya/neuralnetwork/examples/multiclass	0.422s
ok  	github.com/itsmontoya/neuralnetwork/examples/regression	0.538s
ok  	github.com/itsmontoya/neuralnetwork/examples/toycode	0.659s
ok  	github.com/itsmontoya/neuralnetwork/examples/xor	0.772s
ok  	github.com/itsmontoya/neuralnetwork/internal/testutil	(cached)
ok  	github.com/itsmontoya/neuralnetwork/layer	0.864s
ok  	github.com/itsmontoya/neuralnetwork/loss	0.937s
ok  	github.com/itsmontoya/neuralnetwork/matrix	(cached)
ok  	github.com/itsmontoya/neuralnetwork/metric	1.046s
ok  	github.com/itsmontoya/neuralnetwork/model	0.939s
ok  	github.com/itsmontoya/neuralnetwork/optimizer	0.895s
```

```text
ok  	github.com/itsmontoya/neuralnetwork/matrix	0.473s
```

```text
ok  	github.com/itsmontoya/neuralnetwork/matrix	0.199s
```

### Race Test Output

```text
ok  	github.com/itsmontoya/neuralnetwork/activation	1.206s
ok  	github.com/itsmontoya/neuralnetwork/data	1.330s
ok  	github.com/itsmontoya/neuralnetwork/examples/heart	1.455s
ok  	github.com/itsmontoya/neuralnetwork/examples/multiclass	1.479s
ok  	github.com/itsmontoya/neuralnetwork/examples/regression	1.596s
ok  	github.com/itsmontoya/neuralnetwork/examples/toycode	1.657s
ok  	github.com/itsmontoya/neuralnetwork/examples/xor	1.771s
ok  	github.com/itsmontoya/neuralnetwork/internal/testutil	(cached)
ok  	github.com/itsmontoya/neuralnetwork/layer	1.926s
ok  	github.com/itsmontoya/neuralnetwork/loss	1.166s
ok  	github.com/itsmontoya/neuralnetwork/matrix	1.151s
ok  	github.com/itsmontoya/neuralnetwork/metric	1.237s
ok  	github.com/itsmontoya/neuralnetwork/model	1.156s
ok  	github.com/itsmontoya/neuralnetwork/optimizer	1.231s
```

### Interpretation

The integrated arm64 and amd64 elementwise kernels preserve validation and
ownership because public methods still validate matrix shape before passing
package-owned storage to private helpers. Correctness tests compare the active
helpers with the pure Go references for empty input, sizes below and at the
vector width, multi-vector lengths, scalar tails, uneven lengths, negative
values, zeros, `Inf`, and `NaN`. The `purego` build also passes.

`AddInto`, `SubtractInto`, `AddScaledInPlace`, and `AddScalarInto` showed stable
wins on medium, large, and uneven shapes and are integrated for
`arm64 && !purego` and `amd64 && !purego`. The public arm64 256x256 focused
benchmarks improved from 36.470 us to 12.311 us for `AddInto`, 36.648 us to
10.230 us for `SubtractInto`, 30.043 us to 11.583 us for `AddScaledInPlace`,
and 26.178 us to 9.310 us for `AddScalarInto`, all with zero allocations.

The local darwin/amd64 benchmark ran as an amd64 binary on Apple silicon and
reported `VirtualApple @ 2.50GHz`. It still showed the same integrated set
winning on candidate shapes: 256x256 improved from 31.828 us to 14.905 us for
`AddInto`, 31.870 us to 14.895 us for `SubtractInto`, 20.344 us to 11.982 us
for `AddScaledInPlace`, and 19.406 us to 11.104 us for `AddScalarInto`, all
with zero allocations.

The multiply candidates remain pure Go. `MultiplyElementsInto`,
`MultiplyScalarInto`, and `MultiplyScalarInPlace` still report zero allocations,
and the focused after benchmark shows no allocation regression. Direct SIMD
multiply should wait for direct vector multiply support or an approved generated
kernel so the implementation does not rely on fused multiply-add as a substitute
for plain multiplication.
