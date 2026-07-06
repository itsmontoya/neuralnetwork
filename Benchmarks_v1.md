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
