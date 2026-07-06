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
