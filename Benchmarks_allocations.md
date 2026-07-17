# Allocation Reduction Benchmarks

Captured on July 16, 2026.

## Environment

| Field | Value |
| --- | --- |
| OS | darwin |
| Architecture | arm64 |
| CPU | Apple M3 |
| Go version | go1.26.5 |

## Commands

Each benchmark-bearing package was run separately so package concurrency could
not distort timing comparisons.

```sh
go test ./activation -run '^$' -bench=. -benchmem -count=5
go test ./data -run '^$' -bench=. -benchmem -count=5
go test ./internal/scratch -run '^$' -bench=. -benchmem -count=5
go test ./layer -run '^$' -bench=. -benchmem -count=5
go test ./loss -run '^$' -bench=. -benchmem -count=5
go test ./matrix -run '^$' -bench=. -benchmem -count=5
go test ./metric -run '^$' -bench=. -benchmem -count=5
go test ./model -run '^$' -bench=. -benchmem -count=5
go test ./optimizer -run '^$' -bench=. -benchmem -count=5
```

## Median Summary

This table records the allocation contracts and end-state targets most relevant
to the allocation-reduction work. The complete raw output follows.

| Package | Benchmark | Median ns/op | Median B/op | Median allocs/op |
| --- | --- | ---: | ---: | ---: |
| `internal/scratch` | `MatrixPool_WarmedHit` | 3.184 | 0 | 0 |
| `internal/scratch` | `Float32Pool_WarmedHit` | 3.196 | 0 | 0 |
| `internal/scratch` | `MatrixPool_FourShapeRetention` | 14671 | 373699 | 8 |
| `internal/scratch` | `Float32Pool_FourLengthRetention` | 13232 | 373506 | 4 |
| `matrix` | `MatMul` | 157997 | 16432 | 2 |
| `matrix` | `MatMulInto` | 160301 | 0 | 0 |
| `matrix` | `Add` | 15986 | 262192 | 2 |
| `matrix` | `AddInto` | 5700 | 0 | 0 |
| `matrix` | `SoftmaxRowsInto_MediumBatch` | 63378 | 0 | 0 |
| `matrix` | `SoftmaxRowsBackwardInto_MediumBatch` | 74136 | 0 | 0 |
| `layer` | `DenseForwardBackward_AlternatingShapes` | 1303922 | 0 | 0 |
| `layer` | `ActivationForwardBackward_AlternatingShapes` | 293884 | 0 | 0 |
| `layer` | `ActivationForwardBackward_Softmax_AlternatingShapes` | 387633 | 0 | 0 |
| `layer` | `DropoutForwardBackward_AlternatingShapes` | 233971 | 0 | 0 |
| `layer` | `BatchNormalizationForwardBackward_AlternatingShapes` | 124686 | 0 | 0 |
| `loss` | `MeanSquaredErrorGradient_MediumBatch` | 759.4 | 8240 | 2 |
| `loss` | `MeanSquaredErrorGradientInto_MediumBatch` | 321.8 | 0 | 0 |
| `loss` | `BinaryCrossEntropyGradient_MediumBatch` | 725.6 | 560 | 2 |
| `loss` | `BinaryCrossEntropyGradientInto_MediumBatch` | 666.2 | 0 | 0 |
| `loss` | `CategoricalCrossEntropyGradient_MediumBatch` | 7802 | 8240 | 2 |
| `loss` | `CategoricalCrossEntropyGradientInto_MediumBatch` | 7452 | 0 | 0 |
| `optimizer` | `SGDUpdate_SteadyState` | 2233 | 0 | 0 |
| `optimizer` | `MomentumUpdate_SteadyState` | 2782 | 0 | 0 |
| `optimizer` | `AdamUpdate_SteadyState` | 6761 | 0 | 0 |
| `optimizer` | `RegularizedUpdate_SteadyState/SGD/L1` | 7320 | 0 | 0 |
| `optimizer` | `RegularizedUpdate_SteadyState/SGD/L2` | 5939 | 0 | 0 |
| `optimizer` | `RegularizedUpdate_SteadyState/Adam/L1` | 11880 | 0 | 0 |
| `optimizer` | `RegularizedUpdate_SteadyState/Adam/L2` | 10487 | 0 | 0 |
| `metric` | `MetricValue/MeanSquaredError/Medium` | 2776 | 0 | 0 |
| `metric` | `MetricValue/BinaryF1/Medium` | 604.5 | 0 | 0 |
| `metric` | `MetricValue/CategoricalAccuracy/Medium` | 5280 | 0 | 0 |
| `metric` | `MetricValue/CategoricalMacroF1/Medium` | 5660 | 2048 | 1 |
| `metric` | `ConfusionMatrixConstruction/Binary/Medium` | 631.7 | 80 | 2 |
| `metric` | `ConfusionMatrixConstruction/Categorical/Medium` | 5390 | 2096 | 2 |
| `data` | `DatasetBatches_Unshuffled` | 18538 | 173952 | 82 |
| `data` | `DatasetBatches_Shuffled` | 21691 | 173952 | 82 |
| `data` | `BatchInputs` | 671.7 | 8240 | 2 |
| `data` | `BatchTargets` | 185.5 | 2096 | 2 |
| `model` | `SequentialTrainBatch_XOR` | 1487 | 0 | 0 |
| `model` | `SequentialTrainBatch_SyntheticDense` | 758410 | 0 | 0 |
| `model` | `SequentialTrainBatch_Activations/Softmax` | 28033 | 0 | 0 |
| `model` | `SequentialTrainBatch_Regularized/L1` | 763899 | 0 | 0 |
| `model` | `SequentialTrainBatch_Regularized/L2` | 761642 | 0 | 0 |
| `model` | `SequentialTrainBatch_AlternatingShapes` | 362497 | 0 | 0 |
| `model` | `SequentialTrainFitEpoch_Warmed` | 917 | 0 | 0 |
| `model` | `SequentialFit_SyntheticDense` | 1033894 | 31984 | 10 |
| `model` | `SequentialFit_SyntheticDense_TenEpoch` | 10380257 | 33184 | 14 |
| `model` | `SequentialFit_Scenarios/PartialFinalBatch` | 1053892 | 35920 | 14 |
| `model` | `SequentialFit_Scenarios/Validation` | 1168235 | 46416 | 14 |

## Interpretation

Built-in stable and warmed known-shape training paths meet the zero-allocation
target. Four-shape layer cycles and warmed internal Fit epoch batching also
remain allocation-free. Public allocating matrix methods, public loss
gradients, dataset and batch copies, public batch graphs, confusion matrices,
and confusion counts retain caller-owned allocations by contract.

Categorical macro metrics remain the one scalar-metric exception: their dynamic
class count requires one flat counts allocation per call. Fit retains bounded
per-call workspace allocation plus caller-owned history, while every warmed
epoch after that setup is allocation-free.

The scratch pools retain at most four exact shapes or lengths per logical role.
The retention fixture contains 365056 logical data bytes; allocator size
classes and matrix headers produce the measured byte counts above.

## Raw Output

### activation

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/activation
cpu: Apple M3
Benchmark_ActivationForward/ELU/Small-8   	 9652658	       124.1 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/ELU/Small-8   	 9631134	       124.5 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/ELU/Small-8   	 9580503	       124.7 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/ELU/Small-8   	 9569764	       125.1 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/ELU/Small-8   	 9441693	       125.3 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/ELU/Medium-8  	   36817	     32755 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/ELU/Medium-8  	   36570	     32874 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/ELU/Medium-8  	   36235	     33331 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/ELU/Medium-8  	   36394	     32958 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/ELU/Medium-8  	   33333	     33023 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/GELU/Small-8  	14267646	        84.04 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/GELU/Small-8  	14253001	        83.82 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/GELU/Small-8  	14187219	        84.08 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/GELU/Small-8  	14223121	        84.29 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/GELU/Small-8  	14139667	        84.19 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/GELU/Medium-8 	   41702	     28681 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/GELU/Medium-8 	   41696	     28741 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/GELU/Medium-8 	   41428	     28735 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/GELU/Medium-8 	   41680	     28671 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/GELU/Medium-8 	   41782	     28719 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/LeakyReLU/Small-8         	24275527	        48.91 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/LeakyReLU/Small-8         	24713456	        48.55 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/LeakyReLU/Small-8         	24791488	        48.90 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/LeakyReLU/Small-8         	24284348	        48.70 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/LeakyReLU/Small-8         	24770463	        48.74 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/LeakyReLU/Medium-8        	  130146	      9204 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/LeakyReLU/Medium-8        	  129170	      9220 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/LeakyReLU/Medium-8        	  129308	      9239 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/LeakyReLU/Medium-8        	  129531	      9264 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/LeakyReLU/Medium-8        	  129033	      9289 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Linear/Small-8            	40849093	        29.17 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Linear/Small-8            	41385732	        29.09 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Linear/Small-8            	41457758	        29.30 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Linear/Small-8            	40478716	        29.13 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Linear/Small-8            	40459951	        29.12 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Linear/Medium-8           	  697926	      1874 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Linear/Medium-8           	  647042	      1858 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Linear/Medium-8           	  642543	      1862 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Linear/Medium-8           	  634623	      1861 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Linear/Medium-8           	  626127	      1860 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/ReLU/Small-8              	23851405	        49.89 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/ReLU/Small-8              	24221179	        49.28 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/ReLU/Small-8              	24410481	        49.23 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/ReLU/Small-8              	24328597	        49.22 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/ReLU/Small-8              	24412220	        49.36 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/ReLU/Medium-8             	  127784	      9290 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/ReLU/Medium-8             	  128844	      9290 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/ReLU/Medium-8             	  128138	      9331 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/ReLU/Medium-8             	  129050	      9296 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/ReLU/Medium-8             	  128660	      9295 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Sigmoid/Small-8           	 8772127	       134.3 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Sigmoid/Small-8           	 8969293	       133.6 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Sigmoid/Small-8           	 8947233	       133.9 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Sigmoid/Small-8           	 8967132	       133.4 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Sigmoid/Small-8           	 8977332	       133.5 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Sigmoid/Medium-8          	   23875	     50355 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Sigmoid/Medium-8          	   23734	     50233 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Sigmoid/Medium-8          	   23946	     50255 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Sigmoid/Medium-8          	   23931	     50194 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Sigmoid/Medium-8          	   23917	     50217 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Softmax/Small-8           	 8133976	       147.2 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Softmax/Small-8           	 8087703	       147.3 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Softmax/Small-8           	 8029480	       147.9 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Softmax/Small-8           	 8105642	       147.5 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Softmax/Small-8           	 8115484	       147.4 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Softmax/Medium-8          	   18900	     63255 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Softmax/Medium-8          	   18933	     63205 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Softmax/Medium-8          	   18956	     63069 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Softmax/Medium-8          	   18895	     64546 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Softmax/Medium-8          	   18903	     63564 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Tanh/Small-8              	10430092	       116.3 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Tanh/Small-8              	10498767	       114.1 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Tanh/Small-8              	10376679	       113.2 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Tanh/Small-8              	10601982	       113.7 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Tanh/Small-8              	10547734	       113.5 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationForward/Tanh/Medium-8             	   29803	     40283 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Tanh/Medium-8             	   29828	     40564 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Tanh/Medium-8             	   29761	     40271 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Tanh/Medium-8             	   27770	     40340 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationForward/Tanh/Medium-8             	   29694	     40254 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/ELU/Small-8              	 8133414	       146.8 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/ELU/Small-8              	 8181273	       146.8 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/ELU/Small-8              	 8128513	       148.3 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/ELU/Small-8              	 8137335	       150.7 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/ELU/Small-8              	 8093929	       150.4 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/ELU/Medium-8             	   34321	     34831 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/ELU/Medium-8             	   34437	     34805 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/ELU/Medium-8             	   34339	     34827 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/ELU/Medium-8             	   34113	     34862 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/ELU/Medium-8             	   34407	     34886 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/GELU/Small-8             	 5515946	       216.9 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/GELU/Small-8             	 5500123	       217.0 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/GELU/Small-8             	 4853714	       217.9 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/GELU/Small-8             	 5512041	       217.5 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/GELU/Small-8             	 5521467	       216.8 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/GELU/Medium-8            	   13371	     89500 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/GELU/Medium-8            	   13359	     89509 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/GELU/Medium-8            	   13362	     89473 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/GELU/Medium-8            	   13359	     89524 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/GELU/Medium-8            	   13344	     89653 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/LeakyReLU/Small-8        	18067628	        66.46 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/LeakyReLU/Small-8        	18047713	        66.11 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/LeakyReLU/Small-8        	17642442	        66.31 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/LeakyReLU/Small-8        	18029196	        66.40 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/LeakyReLU/Small-8        	17902876	        66.30 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/LeakyReLU/Medium-8       	  120060	      9939 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/LeakyReLU/Medium-8       	  121044	      9925 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/LeakyReLU/Medium-8       	  120439	      9914 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/LeakyReLU/Medium-8       	  120375	      9920 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/LeakyReLU/Medium-8       	  120369	      9918 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Linear/Small-8           	32702448	        35.98 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Linear/Small-8           	33365926	        36.16 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Linear/Small-8           	33338696	        35.94 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Linear/Small-8           	33457443	        36.06 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Linear/Small-8           	33608697	        36.11 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Linear/Medium-8          	  665424	      1877 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Linear/Medium-8          	  644606	      1886 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Linear/Medium-8          	  635425	      1892 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Linear/Medium-8          	  635112	      1877 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Linear/Medium-8          	  644148	      1881 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/ReLU/Small-8             	17685441	        66.62 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/ReLU/Small-8             	17993072	        66.53 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/ReLU/Small-8             	17927984	        66.22 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/ReLU/Small-8             	17914312	        66.58 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/ReLU/Small-8             	17975709	        66.59 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/ReLU/Medium-8            	  119346	      9916 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/ReLU/Medium-8            	  120876	      9927 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/ReLU/Medium-8            	  121027	      9944 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/ReLU/Medium-8            	  120580	      9920 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/ReLU/Medium-8            	  120210	      9920 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Sigmoid/Small-8          	 7368717	       162.2 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Sigmoid/Small-8          	 7398118	       161.9 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Sigmoid/Small-8          	 7402719	       162.4 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Sigmoid/Small-8          	 7300827	       162.7 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Sigmoid/Small-8          	 7323510	       162.8 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Sigmoid/Medium-8         	   21646	     55634 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Sigmoid/Medium-8         	   21670	     55363 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Sigmoid/Medium-8         	   21564	     55413 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Sigmoid/Medium-8         	   21682	     57312 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Sigmoid/Medium-8         	   21166	     56688 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Softmax/Small-8          	 6628706	       177.9 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Softmax/Small-8          	 6683673	       178.5 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Softmax/Small-8          	 6686313	       177.5 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Softmax/Small-8          	 6734038	       178.0 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Softmax/Small-8          	 6727266	       178.1 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Softmax/Medium-8         	   15937	     75072 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Softmax/Medium-8         	   15957	     75170 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Softmax/Medium-8         	   15423	     74932 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Softmax/Medium-8         	   15990	     76752 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Softmax/Medium-8         	   15918	     75141 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Tanh/Small-8             	 9049304	       131.5 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Tanh/Small-8             	 9023692	       135.2 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Tanh/Small-8             	 9029378	       132.3 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Tanh/Small-8             	 9075937	       131.7 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Tanh/Small-8             	 8975289	       131.7 ns/op	     112 B/op	       2 allocs/op
Benchmark_ActivationBackward/Tanh/Medium-8            	   28099	     42598 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Tanh/Medium-8            	   28152	     42611 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Tanh/Medium-8            	   28164	     43795 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Tanh/Medium-8            	   28015	     42676 ns/op	   32816 B/op	       2 allocs/op
Benchmark_ActivationBackward/Tanh/Medium-8            	   28149	     42657 ns/op	   32816 B/op	       2 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/activation	233.750s
```

### data

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/data
cpu: Apple M3
Benchmark_LoadCSV_ColdPath-8                	    1070	   1131206 ns/op	  863969 B/op	    1088 allocs/op
Benchmark_LoadCSV_ColdPath-8                	    1048	   1132651 ns/op	  863969 B/op	    1088 allocs/op
Benchmark_LoadCSV_ColdPath-8                	    1053	   1136290 ns/op	  863970 B/op	    1088 allocs/op
Benchmark_LoadCSV_ColdPath-8                	    1042	   1136318 ns/op	  863969 B/op	    1088 allocs/op
Benchmark_LoadCSV_ColdPath-8                	    1046	   1146425 ns/op	  863969 B/op	    1088 allocs/op
Benchmark_DatasetSplit_ColdPath-8           	   87540	     13842 ns/op	  175968 B/op	      11 allocs/op
Benchmark_DatasetSplit_ColdPath-8           	   85304	     13961 ns/op	  175968 B/op	      11 allocs/op
Benchmark_DatasetSplit_ColdPath-8           	   85682	     14058 ns/op	  175968 B/op	      11 allocs/op
Benchmark_DatasetSplit_ColdPath-8           	   86330	     13969 ns/op	  175968 B/op	      11 allocs/op
Benchmark_DatasetSplit_ColdPath-8           	   85186	     13952 ns/op	  175968 B/op	      11 allocs/op
Benchmark_NewDataset_ColdPath-8             	  126525	      9160 ns/op	  163952 B/op	       5 allocs/op
Benchmark_NewDataset_ColdPath-8             	  124304	      9146 ns/op	  163952 B/op	       5 allocs/op
Benchmark_NewDataset_ColdPath-8             	  128194	      9459 ns/op	  163952 B/op	       5 allocs/op
Benchmark_NewDataset_ColdPath-8             	  130616	      8973 ns/op	  163952 B/op	       5 allocs/op
Benchmark_NewDataset_ColdPath-8             	  125948	      9446 ns/op	  163952 B/op	       5 allocs/op
Benchmark_DatasetCopyAccessors_ColdPath/Inputs-8         	  166318	      7128 ns/op	  131120 B/op	       2 allocs/op
Benchmark_DatasetCopyAccessors_ColdPath/Inputs-8         	  164024	      7151 ns/op	  131120 B/op	       2 allocs/op
Benchmark_DatasetCopyAccessors_ColdPath/Inputs-8         	  167146	      7227 ns/op	  131120 B/op	       2 allocs/op
Benchmark_DatasetCopyAccessors_ColdPath/Inputs-8         	  165291	      7160 ns/op	  131120 B/op	       2 allocs/op
Benchmark_DatasetCopyAccessors_ColdPath/Inputs-8         	  165745	      7125 ns/op	  131120 B/op	       2 allocs/op
Benchmark_DatasetCopyAccessors_ColdPath/Targets-8        	  594636	      2019 ns/op	   32816 B/op	       2 allocs/op
Benchmark_DatasetCopyAccessors_ColdPath/Targets-8        	  598672	      2010 ns/op	   32816 B/op	       2 allocs/op
Benchmark_DatasetCopyAccessors_ColdPath/Targets-8        	  594854	      1998 ns/op	   32816 B/op	       2 allocs/op
Benchmark_DatasetCopyAccessors_ColdPath/Targets-8        	  587066	      1998 ns/op	   32816 B/op	       2 allocs/op
Benchmark_DatasetCopyAccessors_ColdPath/Targets-8        	  580335	      2000 ns/op	   32816 B/op	       2 allocs/op
Benchmark_DatasetBatches_Unshuffled-8                    	   62174	     19011 ns/op	  173952 B/op	      82 allocs/op
Benchmark_DatasetBatches_Unshuffled-8                    	   63318	     18348 ns/op	  173952 B/op	      82 allocs/op
Benchmark_DatasetBatches_Unshuffled-8                    	   67989	     18282 ns/op	  173952 B/op	      82 allocs/op
Benchmark_DatasetBatches_Unshuffled-8                    	   64567	     18538 ns/op	  173952 B/op	      82 allocs/op
Benchmark_DatasetBatches_Unshuffled-8                    	   64603	     18622 ns/op	  173952 B/op	      82 allocs/op
Benchmark_DatasetBatches_Shuffled-8                      	   53364	     22250 ns/op	  173952 B/op	      82 allocs/op
Benchmark_DatasetBatches_Shuffled-8                      	   54373	     21560 ns/op	  173952 B/op	      82 allocs/op
Benchmark_DatasetBatches_Shuffled-8                      	   49263	     21691 ns/op	  173952 B/op	      82 allocs/op
Benchmark_DatasetBatches_Shuffled-8                      	   55126	     21790 ns/op	  173952 B/op	      82 allocs/op
Benchmark_DatasetBatches_Shuffled-8                      	   55964	     21491 ns/op	  173952 B/op	      82 allocs/op
Benchmark_BatchInputs-8                                  	 1793248	       674.5 ns/op	    8240 B/op	       2 allocs/op
Benchmark_BatchInputs-8                                  	 1795143	       668.3 ns/op	    8240 B/op	       2 allocs/op
Benchmark_BatchInputs-8                                  	 1793360	       673.1 ns/op	    8240 B/op	       2 allocs/op
Benchmark_BatchInputs-8                                  	 1791435	       671.0 ns/op	    8240 B/op	       2 allocs/op
Benchmark_BatchInputs-8                                  	 1786430	       671.7 ns/op	    8240 B/op	       2 allocs/op
Benchmark_BatchTargets-8                                 	 6441721	       185.4 ns/op	    2096 B/op	       2 allocs/op
Benchmark_BatchTargets-8                                 	 6390175	       185.8 ns/op	    2096 B/op	       2 allocs/op
Benchmark_BatchTargets-8                                 	 6470709	       185.7 ns/op	    2096 B/op	       2 allocs/op
Benchmark_BatchTargets-8                                 	 6479730	       184.7 ns/op	    2096 B/op	       2 allocs/op
Benchmark_BatchTargets-8                                 	 6540061	       185.5 ns/op	    2096 B/op	       2 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/data	64.431s
```

### internal/scratch

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/internal/scratch
cpu: Apple M3
Benchmark_MatrixPool_WarmedHit-8              	357794065	         3.175 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatrixPool_WarmedHit-8              	352190096	         3.183 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatrixPool_WarmedHit-8              	376657192	         3.184 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatrixPool_WarmedHit-8              	375585189	         3.191 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatrixPool_WarmedHit-8              	376659310	         3.193 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatrixPool_Miss-8                   	37964071	        31.24 ns/op	      62 B/op	       2 allocs/op
Benchmark_MatrixPool_Miss-8                   	38108595	        31.55 ns/op	      62 B/op	       2 allocs/op
Benchmark_MatrixPool_Miss-8                   	38139481	        31.30 ns/op	      62 B/op	       2 allocs/op
Benchmark_MatrixPool_Miss-8                   	38310812	        31.37 ns/op	      62 B/op	       2 allocs/op
Benchmark_MatrixPool_Miss-8                   	37759944	        31.58 ns/op	      62 B/op	       2 allocs/op
Benchmark_Float32Pool_WarmedHit-8             	375684400	         3.194 ns/op	       0 B/op	       0 allocs/op
Benchmark_Float32Pool_WarmedHit-8             	374610463	         3.196 ns/op	       0 B/op	       0 allocs/op
Benchmark_Float32Pool_WarmedHit-8             	375185588	         3.196 ns/op	       0 B/op	       0 allocs/op
Benchmark_Float32Pool_WarmedHit-8             	375500911	         3.196 ns/op	       0 B/op	       0 allocs/op
Benchmark_Float32Pool_WarmedHit-8             	375739345	         3.197 ns/op	       0 B/op	       0 allocs/op
Benchmark_Float32Pool_Miss-8                  	74035227	        16.64 ns/op	      14 B/op	       1 allocs/op
Benchmark_Float32Pool_Miss-8                  	72892202	        16.50 ns/op	      14 B/op	       1 allocs/op
Benchmark_Float32Pool_Miss-8                  	73280392	        16.52 ns/op	      14 B/op	       1 allocs/op
Benchmark_Float32Pool_Miss-8                  	71528804	        16.61 ns/op	      14 B/op	       1 allocs/op
Benchmark_Float32Pool_Miss-8                  	72043046	        16.63 ns/op	      14 B/op	       1 allocs/op
Benchmark_MatrixPool_FourShapeRetention-8     	   82023	     14506 ns/op	    365056 retained-data-B	  373700 B/op	       8 allocs/op
Benchmark_MatrixPool_FourShapeRetention-8     	   81760	     14891 ns/op	    365056 retained-data-B	  373700 B/op	       8 allocs/op
Benchmark_MatrixPool_FourShapeRetention-8     	   81566	     14671 ns/op	    365056 retained-data-B	  373699 B/op	       8 allocs/op
Benchmark_MatrixPool_FourShapeRetention-8     	   80786	     14817 ns/op	    365056 retained-data-B	  373699 B/op	       8 allocs/op
Benchmark_MatrixPool_FourShapeRetention-8     	   81213	     14659 ns/op	    365056 retained-data-B	  373699 B/op	       8 allocs/op
Benchmark_Float32Pool_FourLengthRetention-8   	   88723	     13276 ns/op	    365056 retained-data-B	  373506 B/op	       4 allocs/op
Benchmark_Float32Pool_FourLengthRetention-8   	   87176	     13232 ns/op	    365056 retained-data-B	  373506 B/op	       4 allocs/op
Benchmark_Float32Pool_FourLengthRetention-8   	   88677	     13181 ns/op	    365056 retained-data-B	  373506 B/op	       4 allocs/op
Benchmark_Float32Pool_FourLengthRetention-8   	   90708	     13117 ns/op	    365056 retained-data-B	  373506 B/op	       4 allocs/op
Benchmark_Float32Pool_FourLengthRetention-8   	   88692	     13287 ns/op	    365056 retained-data-B	  373506 B/op	       4 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/internal/scratch	40.957s
```

### layer

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/layer
cpu: Apple M3
Benchmark_DenseForwardBackward_AlternatingShapes-8                	     852	   1305292 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForwardBackward_AlternatingShapes-8                	     918	   1301438 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForwardBackward_AlternatingShapes-8                	     919	   1303922 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForwardBackward_AlternatingShapes-8                	     916	   1304325 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForwardBackward_AlternatingShapes-8                	     775	   1303442 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForwardBackward_AlternatingShapes-8           	    4089	    294109 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForwardBackward_AlternatingShapes-8           	    4087	    293364 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForwardBackward_AlternatingShapes-8           	    4095	    293884 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForwardBackward_AlternatingShapes-8           	    4088	    293536 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForwardBackward_AlternatingShapes-8           	    4088	    294382 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForwardBackward_Softmax_AlternatingShapes-8   	    3082	    387884 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForwardBackward_Softmax_AlternatingShapes-8   	    3106	    387694 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForwardBackward_Softmax_AlternatingShapes-8   	    3106	    387633 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForwardBackward_Softmax_AlternatingShapes-8   	    3105	    386707 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForwardBackward_Softmax_AlternatingShapes-8   	    3067	    387125 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutForwardBackward_AlternatingShapes-8              	    5190	    230490 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutForwardBackward_AlternatingShapes-8              	    4962	    234395 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutForwardBackward_AlternatingShapes-8              	    5097	    233971 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutForwardBackward_AlternatingShapes-8              	    5133	    233622 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutForwardBackward_AlternatingShapes-8              	    5186	    234229 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationForwardBackward_AlternatingShapes-8   	    9592	    124686 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationForwardBackward_AlternatingShapes-8   	    9568	    124659 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationForwardBackward_AlternatingShapes-8   	    9630	    124575 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationForwardBackward_AlternatingShapes-8   	    9439	    124883 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationForwardBackward_AlternatingShapes-8   	    9638	    127146 ns/op	       0 B/op	       0 allocs/op
Benchmark_NewDense_ColdPath-8                                     	  635672	      1821 ns/op	   25856 B/op	      15 allocs/op
Benchmark_NewDense_ColdPath-8                                     	  647536	      1844 ns/op	   25856 B/op	      15 allocs/op
Benchmark_NewDense_ColdPath-8                                     	  646438	      1852 ns/op	   25856 B/op	      15 allocs/op
Benchmark_NewDense_ColdPath-8                                     	  652722	      1848 ns/op	   25856 B/op	      15 allocs/op
Benchmark_NewDense_ColdPath-8                                     	  656196	      1844 ns/op	   25856 B/op	      15 allocs/op
Benchmark_NewBatchNormalization_ColdPath-8                        	 2419515	       490.1 ns/op	    3616 B/op	      19 allocs/op
Benchmark_NewBatchNormalization_ColdPath-8                        	 2421980	       492.3 ns/op	    3616 B/op	      19 allocs/op
Benchmark_NewBatchNormalization_ColdPath-8                        	 2429847	       489.3 ns/op	    3616 B/op	      19 allocs/op
Benchmark_NewBatchNormalization_ColdPath-8                        	 2403100	       489.5 ns/op	    3616 B/op	      19 allocs/op
Benchmark_NewBatchNormalization_ColdPath-8                        	 2422468	       492.5 ns/op	    3616 B/op	      19 allocs/op
Benchmark_DenseForward_XOR-8                                      	12777324	        93.92 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_XOR-8                                      	12587286	        93.98 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_XOR-8                                      	12758180	        94.10 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_XOR-8                                      	12662468	        93.71 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_XOR-8                                      	12747298	        93.68 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_MediumBatch-8                              	    7741	    155065 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_MediumBatch-8                              	    7756	    155046 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_MediumBatch-8                              	    7695	    154138 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_MediumBatch-8                              	    7773	    158658 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseForward_MediumBatch-8                              	    7747	    155160 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_XOR-8                                     	 8008378	       149.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_XOR-8                                     	 8006293	       149.3 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_XOR-8                                     	 8016116	       148.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_XOR-8                                     	 7849204	       150.1 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_XOR-8                                     	 7960255	       149.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_MediumBatch-8                             	    3814	    314087 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_MediumBatch-8                             	    3826	    313738 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_MediumBatch-8                             	    3829	    313797 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_MediumBatch-8                             	    3826	    313685 ns/op	       0 B/op	       0 allocs/op
Benchmark_DenseBackward_MediumBatch-8                             	    3830	    313607 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForward_MediumBatch-8                         	   23646	     50798 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForward_MediumBatch-8                         	   23692	     50556 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForward_MediumBatch-8                         	   23664	     50589 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForward_MediumBatch-8                         	   23678	     50691 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForward_MediumBatch-8                         	   23715	     50575 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationBackward_MediumBatch-8                        	   22314	     53910 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationBackward_MediumBatch-8                        	   21314	     53826 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationBackward_MediumBatch-8                        	   22263	     53822 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationBackward_MediumBatch-8                        	   22256	     53826 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationBackward_MediumBatch-8                        	   22296	     53822 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForward_Softmax_MediumBatch-8                 	   18742	     63882 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForward_Softmax_MediumBatch-8                 	   18793	     63827 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForward_Softmax_MediumBatch-8                 	   18799	     63760 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForward_Softmax_MediumBatch-8                 	   18765	     63713 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationForward_Softmax_MediumBatch-8                 	   18722	     63752 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationBackward_Softmax_MediumBatch-8                	   16104	     74654 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationBackward_Softmax_MediumBatch-8                	   16172	     76972 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationBackward_Softmax_MediumBatch-8                	   16053	     74644 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationBackward_Softmax_MediumBatch-8                	   15951	     74519 ns/op	       0 B/op	       0 allocs/op
Benchmark_ActivationBackward_Softmax_MediumBatch-8                	   16072	     74333 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutForwardTraining_MediumBatch-8                    	   14684	     81805 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutForwardTraining_MediumBatch-8                    	   14716	     81448 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutForwardTraining_MediumBatch-8                    	   14760	     81822 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutForwardTraining_MediumBatch-8                    	   14666	     83250 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutForwardTraining_MediumBatch-8                    	   14716	     81534 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutBackwardTraining_MediumBatch-8                   	 1929080	       594.4 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutBackwardTraining_MediumBatch-8                   	 2017131	       594.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutBackwardTraining_MediumBatch-8                   	 2024330	       596.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutBackwardTraining_MediumBatch-8                   	 2019187	       595.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_DropoutBackwardTraining_MediumBatch-8                   	 2024073	       589.3 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationForwardTraining_MediumBatch-8         	   54208	     22167 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationForwardTraining_MediumBatch-8         	   54229	     22173 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationForwardTraining_MediumBatch-8         	   54178	     22200 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationForwardTraining_MediumBatch-8         	   53814	     22176 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationForwardTraining_MediumBatch-8         	   54121	     22230 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationBackwardTraining_MediumBatch-8        	   54729	     22357 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationBackwardTraining_MediumBatch-8        	   54700	     21802 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationBackwardTraining_MediumBatch-8        	   54358	     21796 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationBackwardTraining_MediumBatch-8        	   54786	     21778 ns/op	       0 B/op	       0 allocs/op
Benchmark_BatchNormalizationBackwardTraining_MediumBatch-8        	   54756	     21799 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/layer	140.954s
```

### loss

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/loss
cpu: Apple M3
Benchmark_MeanSquaredErrorValue_Small-8                       	61238762	        19.88 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorValue_Small-8                       	52087196	        19.66 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorValue_Small-8                       	61949871	        19.39 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorValue_Small-8                       	60384194	        19.37 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorValue_Small-8                       	62621358	        19.38 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorValue_MediumBatch-8                 	  438788	      2724 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorValue_MediumBatch-8                 	  432150	      2773 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorValue_MediumBatch-8                 	  432012	      2773 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorValue_MediumBatch-8                 	  428130	      2813 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorValue_MediumBatch-8                 	  431162	      2778 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorGradient_Small-8                    	24393279	        47.71 ns/op	      64 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradient_Small-8                    	25050970	        47.76 ns/op	      64 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradient_Small-8                    	25194116	        47.65 ns/op	      64 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradient_Small-8                    	25197158	        48.34 ns/op	      64 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradient_Small-8                    	24838162	        48.02 ns/op	      64 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradient_MediumBatch-8              	 1558668	       763.2 ns/op	    8240 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradient_MediumBatch-8              	 1593945	       749.3 ns/op	    8240 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradient_MediumBatch-8              	 1574258	       764.2 ns/op	    8240 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradient_MediumBatch-8              	 1559858	       758.4 ns/op	    8240 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradient_MediumBatch-8              	 1594935	       759.4 ns/op	    8240 B/op	       2 allocs/op
Benchmark_MeanSquaredErrorGradientInto_Small-8                	48257532	        24.64 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorGradientInto_Small-8                	47629442	        24.59 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorGradientInto_Small-8                	47243785	        24.73 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorGradientInto_Small-8                	47218996	        24.73 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorGradientInto_Small-8                	47087752	        24.98 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorGradientInto_MediumBatch-8          	 3732158	       321.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorGradientInto_MediumBatch-8          	 3728180	       321.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorGradientInto_MediumBatch-8          	 3731181	       322.1 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorGradientInto_MediumBatch-8          	 3731038	       321.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_MeanSquaredErrorGradientInto_MediumBatch-8          	 3732645	       321.9 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyValue_Small-8                     	20113922	        59.95 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyValue_Small-8                     	20405863	        59.34 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyValue_Small-8                     	20423156	        59.04 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyValue_Small-8                     	20058378	        59.04 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyValue_Small-8                     	20465402	        61.04 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyValue_MediumBatch-8               	  894846	      1336 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyValue_MediumBatch-8               	  897109	      1334 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyValue_MediumBatch-8               	  896006	      1336 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyValue_MediumBatch-8               	  884895	      1336 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyValue_MediumBatch-8               	  898304	      1337 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyGradient_Small-8                  	17042054	        68.32 ns/op	      64 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradient_Small-8                  	17580903	        68.40 ns/op	      64 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradient_Small-8                  	17510611	        68.12 ns/op	      64 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradient_Small-8                  	17550583	        68.50 ns/op	      64 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradient_Small-8                  	17602825	        68.44 ns/op	      64 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradient_MediumBatch-8            	 1639629	       732.8 ns/op	     560 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradient_MediumBatch-8            	 1651383	       726.6 ns/op	     560 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradient_MediumBatch-8            	 1650372	       724.1 ns/op	     560 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradient_MediumBatch-8            	 1654075	       725.6 ns/op	     560 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradient_MediumBatch-8            	 1646901	       725.0 ns/op	     560 B/op	       2 allocs/op
Benchmark_BinaryCrossEntropyGradientInto_Small-8              	26896004	        45.28 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyGradientInto_Small-8              	26048307	        44.84 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyGradientInto_Small-8              	25937720	        45.25 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyGradientInto_Small-8              	25514316	        45.21 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyGradientInto_Small-8              	25831495	        45.26 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyGradientInto_MediumBatch-8        	 1791088	       669.1 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyGradientInto_MediumBatch-8        	 1795936	       666.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyGradientInto_MediumBatch-8        	 1800684	       665.1 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyGradientInto_MediumBatch-8        	 1799701	       669.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_BinaryCrossEntropyGradientInto_MediumBatch-8        	 1801558	       666.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyValue_Small-8                	18029104	        66.45 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyValue_Small-8                	17993724	        66.75 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyValue_Small-8                	17982534	        66.71 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyValue_Small-8                	17995129	        66.67 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyValue_Small-8                	17958772	        66.67 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyValue_MediumBatch-8          	  241682	      4972 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyValue_MediumBatch-8          	  241459	      4981 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyValue_MediumBatch-8          	  241476	      4970 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyValue_MediumBatch-8          	  241566	      5020 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyValue_MediumBatch-8          	  240703	      4974 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyGradient_Small-8             	10461991	       110.1 ns/op	      96 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradient_Small-8             	11016555	       116.1 ns/op	      96 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradient_Small-8             	10872950	       109.1 ns/op	      96 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradient_Small-8             	11050321	       109.1 ns/op	      96 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradient_Small-8             	11109130	       107.9 ns/op	      96 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradient_MediumBatch-8       	  150886	      7787 ns/op	    8240 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradient_MediumBatch-8       	  153926	      7837 ns/op	    8240 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradient_MediumBatch-8       	  153529	      7802 ns/op	    8240 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradient_MediumBatch-8       	  153658	      7806 ns/op	    8240 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradient_MediumBatch-8       	  153372	      7794 ns/op	    8240 B/op	       2 allocs/op
Benchmark_CategoricalCrossEntropyGradientInto_Small-8         	13682041	        83.14 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyGradientInto_Small-8         	14389372	        83.56 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyGradientInto_Small-8         	14326989	        83.50 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyGradientInto_Small-8         	14288378	        85.15 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyGradientInto_Small-8         	14212837	        82.38 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyGradientInto_MediumBatch-8   	  160098	      7411 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyGradientInto_MediumBatch-8   	  159600	      7452 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyGradientInto_MediumBatch-8   	  160243	      7452 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyGradientInto_MediumBatch-8   	  161364	      7444 ns/op	       0 B/op	       0 allocs/op
Benchmark_CategoricalCrossEntropyGradientInto_MediumBatch-8   	  158895	      7471 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/loss	124.849s
```

### matrix

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/matrix
cpu: Apple M3
Benchmark_DotProduct/Length1-8                    	330531807	         3.449 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length1-8                    	347879780	         3.447 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length1-8                    	347755107	         3.447 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length1-8                    	347944420	         3.449 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length1-8                    	348158644	         3.450 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length2-8                    	255889500	         4.636 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length2-8                    	256869405	         4.620 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length2-8                    	259207584	         4.625 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length2-8                    	256736571	         4.631 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length2-8                    	260098034	         4.637 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length3-8                    	230574073	         5.006 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length3-8                    	231071994	         4.904 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length3-8                    	245227012	         4.902 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length3-8                    	245656213	         4.901 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length3-8                    	245281398	         4.913 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4-8                    	215472254	         5.570 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4-8                    	215356069	         5.569 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4-8                    	215528256	         5.580 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4-8                    	215555324	         5.568 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4-8                    	215588709	         5.570 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length5-8                    	203828810	         5.894 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length5-8                    	203782917	         5.890 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length5-8                    	203507767	         5.903 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length5-8                    	202789966	         5.891 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length5-8                    	203726929	         5.896 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length31-8                   	100000000	        11.42 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length31-8                   	99420733	        11.37 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length31-8                   	100000000	        11.36 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length31-8                   	100000000	        11.39 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length31-8                   	100000000	        11.43 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length33-8                   	126157141	         9.517 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length33-8                   	126103106	         9.514 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length33-8                   	126220287	         9.516 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length33-8                   	126004336	         9.504 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length33-8                   	126233764	         9.513 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length64-8                   	96342331	        12.89 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length64-8                   	93291442	        12.88 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length64-8                   	96082946	        12.89 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length64-8                   	93550233	        12.87 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length64-8                   	90680102	        12.87 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length257-8                  	33249784	        36.37 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length257-8                  	33136126	        36.34 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length257-8                  	33379732	        36.42 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length257-8                  	33651109	        36.46 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length257-8                  	33251935	        36.35 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4096-8                 	 2383555	       504.1 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4096-8                 	 2381367	       506.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4096-8                 	 2388393	       503.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4096-8                 	 2381166	       503.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4096-8                 	 2381695	       504.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4099-8                 	 2385391	       502.9 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4099-8                 	 2380684	       503.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4099-8                 	 2382609	       503.1 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4099-8                 	 2384133	       503.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length4099-8                 	 2382110	       503.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8                	  151777	      7972 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8                	  150594	      7978 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8                	  150571	      7980 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8                	  150516	      7978 ns/op	       0 B/op	       0 allocs/op
Benchmark_DotProduct/Length65537-8                	  150364	      7973 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small2x2-8         	42432375	        25.28 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small2x2-8         	47553943	        25.46 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small2x2-8         	48027134	        25.13 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small2x2-8         	45519634	        25.05 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small2x2-8         	47946538	        25.03 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small4x4-8         	13453656	        89.28 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small4x4-8         	13461799	        89.23 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small4x4-8         	13440326	        89.16 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small4x4-8         	13458339	        89.28 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Small4x4-8         	13445635	        89.20 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Medium64x64-8      	   22581	     53146 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Medium64x64-8      	   22568	     53135 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Medium64x64-8      	   22585	     53153 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Medium64x64-8      	   22580	     53149 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Medium64x64-8      	   22596	     53205 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Large128x256x128-8 	    2025	    591450 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Large128x256x128-8 	    2040	    591592 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Large128x256x128-8 	    2030	    591892 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Large128x256x128-8 	    2038	    591900 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Large128x256x128-8 	    2031	    591694 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven17x33x19-8   	  385777	      3098 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven17x33x19-8   	  389482	      3094 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven17x33x19-8   	  388100	      3097 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven17x33x19-8   	  388244	      3106 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven17x33x19-8   	  389944	      3100 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven63x65x31-8   	   44198	     27164 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven63x65x31-8   	   43884	     27402 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven63x65x31-8   	   44284	     27396 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven63x65x31-8   	   44295	     27422 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeDotCandidate/Uneven63x65x31-8   	   43770	     27411 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Pure-8       	754282320	         1.593 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Pure-8       	754252294	         1.591 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Pure-8       	754186128	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Pure-8       	753185739	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Pure-8       	754399881	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Active-8     	754270267	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Active-8     	753360694	         1.593 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Active-8     	754461536	         1.591 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Active-8     	754232739	         1.593 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddInto/Active-8     	753842830	         1.593 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Pure-8  	752306877	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Pure-8  	753377839	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Pure-8  	754293380	         1.593 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Pure-8  	752455864	         1.595 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Pure-8  	753377646	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Active-8         	754588249	         1.591 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Active-8         	754324200	         1.596 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Active-8         	753196575	         1.593 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Active-8         	754454226	         1.591 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/SubtractInto/Active-8         	754060928	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyElementsInto/Pure-8   	753931833	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyElementsInto/Pure-8   	754211995	         1.591 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyElementsInto/Pure-8   	754116613	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyElementsInto/Pure-8   	753966372	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyElementsInto/Pure-8   	753443277	         1.593 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyElementsInto/Active-8 	347969812	         3.450 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyElementsInto/Active-8 	348150565	         3.478 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyElementsInto/Active-8 	346947194	         3.459 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyElementsInto/Active-8 	347753511	         3.448 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyElementsInto/Active-8 	348105074	         3.449 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Pure-8       	754264934	         1.591 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Pure-8       	751277367	         1.597 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Pure-8       	753995984	         1.635 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Pure-8       	753952752	         1.599 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Pure-8       	754010193	         1.593 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Active-8     	754000916	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Active-8     	754022630	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Active-8     	754403044	         1.596 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Active-8     	754105552	         1.614 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScaledInPlace/Active-8     	752935468	         1.609 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Pure-8          	754151565	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Pure-8          	754131618	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Pure-8          	734302188	         1.626 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Pure-8          	754115035	         1.593 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Pure-8          	754021246	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Active-8        	752333997	         1.626 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Active-8        	755156301	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Active-8        	754167955	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Active-8        	751956849	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/AddScalarInto/Active-8        	750314780	         1.595 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInto/Pure-8     	754176842	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInto/Pure-8     	753663316	         1.711 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInto/Pure-8     	754765437	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInto/Pure-8     	753650101	         1.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInto/Pure-8     	754336059	         1.591 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInto/Active-8   	453134271	         2.653 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInto/Active-8   	449900671	         2.653 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInto/Active-8   	452707687	         2.654 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInto/Active-8   	450778155	         2.655 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInto/Active-8   	452459613	         2.661 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInPlace/Pure-8  	902658040	         1.326 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInPlace/Pure-8  	904741125	         1.327 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInPlace/Pure-8  	904948936	         1.326 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInPlace/Pure-8  	904673770	         1.327 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInPlace/Pure-8  	904643359	         1.327 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInPlace/Active-8         	502837452	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInPlace/Active-8         	502561052	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInPlace/Active-8         	502850797	         2.389 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInPlace/Active-8         	502801460	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x1/MultiplyScalarInPlace/Active-8         	502070517	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Pure-8                         	501695661	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Pure-8                         	503195817	         2.386 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Pure-8                         	502469427	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Pure-8                         	502923328	         2.390 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Pure-8                         	502748972	         2.392 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Active-8                       	501974169	         2.386 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Active-8                       	502217167	         2.389 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Active-8                       	502338226	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Active-8                       	501960960	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddInto/Active-8                       	502342170	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Pure-8                    	502770914	         2.393 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Pure-8                    	497859291	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Pure-8                    	502737387	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Pure-8                    	502991400	         2.390 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Pure-8                    	502797069	         2.392 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Active-8                  	502777758	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Active-8                  	502319916	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Active-8                  	502761346	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Active-8                  	502469338	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/SubtractInto/Active-8                  	502836837	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyElementsInto/Pure-8            	502793560	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyElementsInto/Pure-8            	502840436	         2.394 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyElementsInto/Pure-8            	502897335	         2.389 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyElementsInto/Pure-8            	502521068	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyElementsInto/Pure-8            	502785219	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyElementsInto/Active-8          	281750464	         4.244 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyElementsInto/Active-8          	282895072	         4.249 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyElementsInto/Active-8          	282810012	         4.245 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyElementsInto/Active-8          	278050518	         4.249 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyElementsInto/Active-8          	282607980	         4.245 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Pure-8                	502536500	         2.394 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Pure-8                	502935361	         2.394 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Pure-8                	502256754	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Pure-8                	502583418	         2.392 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Pure-8                	502396413	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Active-8              	502708340	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Active-8              	501238911	         2.389 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Active-8              	502742301	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Active-8              	502867214	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScaledInPlace/Active-8              	502431996	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Pure-8                   	502393608	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Pure-8                   	502265162	         2.391 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Pure-8                   	503113098	         2.394 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Pure-8                   	502771350	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Pure-8                   	501816733	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Active-8                 	502792945	         2.391 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Active-8                 	502788027	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Active-8                 	501861943	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Active-8                 	502556230	         2.392 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/AddScalarInto/Active-8                 	502857556	         2.390 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInto/Pure-8              	496406395	         2.387 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInto/Pure-8              	452106532	         2.530 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInto/Pure-8              	502482490	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInto/Pure-8              	455449309	         2.420 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInto/Pure-8              	502901989	         2.388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInto/Active-8            	347936518	         3.462 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInto/Active-8            	348084332	         3.448 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInto/Active-8            	347788786	         3.450 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInto/Active-8            	348162475	         3.453 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInto/Active-8            	347948414	         3.450 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInPlace/Pure-8           	461551774	         2.596 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInPlace/Pure-8           	463565148	         2.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInPlace/Pure-8           	464295812	         2.590 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInPlace/Pure-8           	412455526	         2.851 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInPlace/Pure-8           	423097914	         2.588 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInPlace/Active-8         	376947414	         3.184 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInPlace/Active-8         	377107134	         3.182 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInPlace/Active-8         	376360778	         3.183 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInPlace/Active-8         	377143382	         3.192 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x2/MultiplyScalarInPlace/Active-8         	376916088	         3.184 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Pure-8                         	434756617	         2.782 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Pure-8                         	434035430	         2.714 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Pure-8                         	436018653	         2.713 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Pure-8                         	438565902	         2.749 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Pure-8                         	430191945	         2.743 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Active-8                       	441025948	         2.761 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Active-8                       	426325038	         2.804 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Active-8                       	422868368	         2.786 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Active-8                       	427209000	         2.841 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddInto/Active-8                       	429687529	         2.808 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Pure-8                    	419035135	         2.662 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Pure-8                    	445816774	         2.696 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Pure-8                    	435263013	         2.732 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Pure-8                    	421910455	         2.682 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Pure-8                    	409457209	         2.771 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Active-8                  	445990476	         2.696 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Active-8                  	447000278	         2.699 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Active-8                  	450349935	         2.686 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Active-8                  	441191000	         2.713 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/SubtractInto/Active-8                  	428897248	         2.699 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyElementsInto/Pure-8            	428271702	         2.747 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyElementsInto/Pure-8            	433428666	         2.772 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyElementsInto/Pure-8            	449067622	         2.802 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyElementsInto/Pure-8            	424606663	         2.794 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyElementsInto/Pure-8            	433614912	         2.742 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyElementsInto/Active-8          	266206917	         4.537 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyElementsInto/Active-8          	266074650	         4.518 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyElementsInto/Active-8          	265635704	         4.512 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyElementsInto/Active-8          	266147850	         4.509 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyElementsInto/Active-8          	266088835	         4.510 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Pure-8                	445539798	         2.678 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Pure-8                	441942151	         2.716 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Pure-8                	445975972	         2.713 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Pure-8                	440101094	         2.684 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Pure-8                	452479303	         2.710 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Active-8              	420450844	         2.671 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Active-8              	438777243	         2.707 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Active-8              	432427303	         2.667 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Active-8              	446486707	         2.697 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScaledInPlace/Active-8              	437796890	         2.714 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Pure-8                   	438783860	         2.661 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Pure-8                   	449373614	         2.656 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Pure-8                   	445715352	         2.691 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Pure-8                   	448927063	         2.704 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Pure-8                   	442726419	         2.704 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Active-8                 	442385575	         2.737 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Active-8                 	423544921	         2.822 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Active-8                 	438081979	         2.727 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Active-8                 	447428197	         2.791 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/AddScalarInto/Active-8                 	428336925	         2.800 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInto/Pure-8              	430389117	         2.708 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInto/Pure-8              	435789120	         2.702 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInto/Pure-8              	449308204	         2.707 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInto/Pure-8              	441662040	         2.688 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInto/Pure-8              	434702479	         2.712 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInto/Active-8            	323236095	         3.714 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInto/Active-8            	321956277	         3.717 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInto/Active-8            	323058284	         3.714 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInto/Active-8            	322544445	         3.713 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInto/Active-8            	323323804	         3.714 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInPlace/Pure-8           	416782678	         2.818 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInPlace/Pure-8           	426144308	         2.816 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInPlace/Pure-8           	424323145	         2.821 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInPlace/Pure-8           	426309639	         2.816 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInPlace/Pure-8           	426117637	         2.820 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInPlace/Active-8         	347648692	         3.459 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInPlace/Active-8         	347917225	         3.449 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInPlace/Active-8         	348140253	         3.451 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInPlace/Active-8         	348017580	         3.448 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small1x3/MultiplyScalarInPlace/Active-8         	348084627	         3.453 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Pure-8                         	401489860	         3.142 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Pure-8                         	402992047	         3.239 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Pure-8                         	411208456	         3.265 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Pure-8                         	387686692	         3.296 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Pure-8                         	370149974	         3.200 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Active-8                       	337808197	         3.257 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Active-8                       	362548123	         3.402 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Active-8                       	351475017	         3.430 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Active-8                       	332652664	         3.368 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddInto/Active-8                       	365763982	         3.397 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Pure-8                    	397657027	         3.012 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Pure-8                    	392046252	         3.055 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Pure-8                    	403022726	         3.124 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Pure-8                    	325436083	         3.244 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Pure-8                    	397435930	         3.355 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Active-8                  	357016634	         3.042 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Active-8                  	355734652	         3.005 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Active-8                  	384289394	         3.152 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Active-8                  	407545189	         3.237 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/SubtractInto/Active-8                  	411375210	         3.174 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyElementsInto/Pure-8            	402638455	         3.154 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyElementsInto/Pure-8            	325871000	         3.232 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyElementsInto/Pure-8            	373428536	         3.331 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyElementsInto/Pure-8            	383374791	         3.131 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyElementsInto/Pure-8            	357992200	         3.565 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyElementsInto/Active-8          	308287770	         3.875 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyElementsInto/Active-8          	308032063	         3.993 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyElementsInto/Active-8          	307835832	         3.958 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyElementsInto/Active-8          	307814215	         3.867 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyElementsInto/Active-8          	308026792	         3.878 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Pure-8                	336111033	         2.990 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Pure-8                	347757333	         3.625 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Pure-8                	368309947	         3.233 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Pure-8                	401092196	         3.010 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Pure-8                	410705787	         4.587 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Active-8              	380414924	         3.105 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Active-8              	372110714	         3.240 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Active-8              	378951656	         3.411 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Active-8              	367560675	         3.314 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScaledInPlace/Active-8              	382397295	         3.470 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Pure-8                   	357910206	         3.342 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Pure-8                   	358231053	         3.284 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Pure-8                   	380747954	         3.185 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Pure-8                   	319141437	         3.566 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Pure-8                   	352452406	         3.255 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Active-8                 	381796372	         3.101 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Active-8                 	348229452	         3.320 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Active-8                 	373161934	         3.155 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Active-8                 	365645799	         3.178 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/AddScalarInto/Active-8                 	376972774	         3.229 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInto/Pure-8              	371528626	         3.268 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInto/Pure-8              	353791576	         3.209 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInto/Pure-8              	368060901	         3.170 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInto/Pure-8              	377199495	         3.233 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInto/Pure-8              	352374570	         3.217 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInto/Active-8            	409388180	         2.929 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInto/Active-8            	409433518	         2.928 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInto/Active-8            	409067433	         2.918 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInto/Active-8            	413343354	         2.901 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInto/Active-8            	408314647	         2.911 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInPlace/Pure-8           	384777249	         3.091 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInPlace/Pure-8           	387234230	         3.112 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInPlace/Pure-8           	389176514	         3.092 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInPlace/Pure-8           	386995394	         3.084 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInPlace/Pure-8           	388109770	         3.090 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInPlace/Active-8         	348130658	         3.215 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInPlace/Active-8         	373487956	         3.214 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInPlace/Active-8         	373259869	         3.213 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInPlace/Active-8         	373100925	         3.217 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Small2x2/MultiplyScalarInPlace/Active-8         	373352049	         3.222 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Pure-8                    	   45738	     26254 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Pure-8                    	   45488	     26234 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Pure-8                    	   45740	     26276 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Pure-8                    	   45278	     26266 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Pure-8                    	   45733	     26251 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Active-8                  	  168602	      7050 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Active-8                  	  174219	      7059 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Active-8                  	  168547	      7068 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Active-8                  	  170197	      7062 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddInto/Active-8                  	  168594	      7046 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Pure-8               	   45770	     26239 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Pure-8               	   45730	     26266 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Pure-8               	   41668	     26226 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Pure-8               	   45402	     26254 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Pure-8               	   45696	     26253 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Active-8             	  169111	      7061 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Active-8             	  170140	      7056 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Active-8             	  213262	      6025 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Active-8             	  223179	      5631 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/SubtractInto/Active-8             	  169896	      7067 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Pure-8       	   45708	     26303 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Pure-8       	   45726	     26250 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Pure-8       	   45469	     26240 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Pure-8       	   45745	     26239 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Pure-8       	   45686	     26240 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Active-8     	  170176	      7071 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Active-8     	  210576	      7047 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Active-8     	  168399	      6817 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Active-8     	  171640	      7063 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyElementsInto/Active-8     	  206463	      7046 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Pure-8           	   66631	     17971 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Pure-8           	   66619	     17961 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Pure-8           	   66600	     18667 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Pure-8           	   66615	     17947 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Pure-8           	   66405	     17964 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Active-8         	   66486	     17953 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Active-8         	   66346	     18013 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Active-8         	   66488	     18072 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Active-8         	   65694	     18075 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScaledInPlace/Active-8         	   58122	     18040 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Pure-8              	   68565	     17546 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Pure-8              	   68269	     17472 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Pure-8              	   68527	     17523 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Pure-8              	   68071	     17493 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Pure-8              	   67686	     17490 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Active-8            	  246946	      4790 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Active-8            	  252978	      4422 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Active-8            	  248203	      4697 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Active-8            	  250126	      4788 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/AddScalarInto/Active-8            	  246283	      4525 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInto/Pure-8         	   68527	     17486 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInto/Pure-8         	   68480	     17487 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInto/Pure-8         	   68462	     17485 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInto/Pure-8         	   68472	     17492 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInto/Pure-8         	   68383	     17472 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInto/Active-8       	  249729	      4641 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInto/Active-8       	  250146	      4426 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInto/Active-8       	  259371	      4786 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInto/Active-8       	  248856	      4789 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInto/Active-8       	  251865	      4535 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Pure-8      	   68697	     17467 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Pure-8      	   68647	     17459 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Pure-8      	   68091	     17474 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Pure-8      	   68690	     17518 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Pure-8      	   68668	     17465 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Active-8    	  271340	      4413 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Active-8    	  271371	      4413 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Active-8    	  271183	      4467 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Active-8    	  271276	      4436 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Medium256x256/MultiplyScalarInPlace/Active-8    	  272242	      4420 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Pure-8                   	    2833	    423014 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Pure-8                   	    2835	    422852 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Pure-8                   	    2827	    422697 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Pure-8                   	    2847	    422620 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Pure-8                   	    2828	    423456 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8                 	   10000	    112053 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8                 	   10000	    108924 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8                 	   10000	    109735 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8                 	   10000	    112719 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddInto/Active-8                 	   10000	    106763 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Pure-8              	    2830	    423274 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Pure-8              	    2840	    422906 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Pure-8              	    2846	    422894 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Pure-8              	    2842	    423025 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Pure-8              	    2824	    423049 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Active-8            	   10000	    102240 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Active-8            	   10000	    107884 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Active-8            	   12051	    105997 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Active-8            	   10000	    103850 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/SubtractInto/Active-8            	   10000	    112504 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Pure-8      	    2845	    424922 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Pure-8      	    2836	    422641 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Pure-8      	    2842	    423933 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Pure-8      	    2839	    423418 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Pure-8      	    2817	    439544 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8    	   10000	    109892 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8    	   10000	    100574 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8    	   10000	    108437 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8    	   10000	    108395 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyElementsInto/Active-8    	   10000	    112488 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Pure-8          	    4078	    296224 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Pure-8          	    4035	    295874 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Pure-8          	    4056	    301666 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Pure-8          	    4047	    295827 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Pure-8          	    4065	    296214 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Active-8        	    4054	    295550 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Active-8        	    4054	    295862 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Active-8        	    4060	    296323 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Active-8        	    4040	    308831 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScaledInPlace/Active-8        	    4071	    302872 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Pure-8             	    4281	    279867 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Pure-8             	    4294	    280406 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Pure-8             	    4281	    279726 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Pure-8             	    4281	    279980 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Pure-8             	    4282	    280020 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Active-8           	   16215	     74388 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Active-8           	   16202	     74339 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Active-8           	   16266	     74400 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Active-8           	   16096	     71370 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/AddScalarInto/Active-8           	   16070	     74443 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Pure-8        	    4276	    280472 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Pure-8        	    4287	    279602 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Pure-8        	    4288	    279637 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Pure-8        	    4288	    280586 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Pure-8        	    4275	    279737 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Active-8      	   16749	     73572 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Active-8      	   16980	     74225 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Active-8      	   16071	     72908 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Active-8      	   16138	     73766 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInto/Active-8      	   16122	     74380 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Pure-8     	    4276	    280558 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Pure-8     	    4296	    279794 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Pure-8     	    4297	    279618 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Pure-8     	    4299	    280164 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Pure-8     	    4298	    280739 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8   	   16893	     71037 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8   	   16851	     70979 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8   	   16911	     71016 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8   	   16885	     71060 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Large1024x1024/MultiplyScalarInPlace/Active-8   	   16867	     71012 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Pure-8                      	 8901625	       134.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Pure-8                      	 8774703	       134.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Pure-8                      	 8892690	       134.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Pure-8                      	 8891799	       134.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Pure-8                      	 8901278	       134.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Active-8                    	41391976	        29.65 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Active-8                    	41223655	        29.01 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Active-8                    	40560066	        28.99 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Active-8                    	41403878	        28.99 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddInto/Active-8                    	41351682	        28.98 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Pure-8                 	 8882114	       134.9 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Pure-8                 	 8890537	       135.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Pure-8                 	 8789724	       134.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Pure-8                 	 8901190	       134.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Pure-8                 	 8892948	       134.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Active-8               	41219820	        29.01 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Active-8               	41375504	        29.00 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Active-8               	41374316	        28.99 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Active-8               	41046398	        29.05 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/SubtractInto/Active-8               	41363678	        29.00 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyElementsInto/Pure-8         	 8908153	       134.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyElementsInto/Pure-8         	 8903976	       134.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyElementsInto/Pure-8         	 8896059	       134.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyElementsInto/Pure-8         	 8907115	       134.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyElementsInto/Pure-8         	 7663456	       134.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyElementsInto/Active-8       	41381628	        29.06 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyElementsInto/Active-8       	41480686	        28.99 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyElementsInto/Active-8       	41451910	        28.95 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyElementsInto/Active-8       	41400247	        28.97 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyElementsInto/Active-8       	41318994	        30.06 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Pure-8             	12355556	        98.40 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Pure-8             	12157324	       101.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Pure-8             	12138248	        98.43 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Pure-8             	12175195	        97.37 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Pure-8             	12349887	        98.89 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Active-8           	12138039	        97.61 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Active-8           	12310922	        98.50 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Active-8           	12156452	        97.35 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Active-8           	11874794	        97.11 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScaledInPlace/Active-8           	12327210	        98.45 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Pure-8                	12398145	        96.86 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Pure-8                	12349003	       100.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Pure-8                	12396720	        96.88 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Pure-8                	12351847	       100.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Pure-8                	12384092	        96.79 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Active-8              	45570051	        26.40 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Active-8              	45303390	        26.39 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Active-8              	45506472	        26.39 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Active-8              	45234228	        26.40 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/AddScalarInto/Active-8              	45463658	        26.40 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInto/Pure-8           	12410263	        96.79 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInto/Pure-8           	12398342	        96.80 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInto/Pure-8           	12345148	        96.78 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInto/Pure-8           	12383725	        96.84 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInto/Pure-8           	12399618	        96.75 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInto/Active-8         	36944058	        32.40 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInto/Active-8         	36820921	        32.44 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInto/Active-8         	36879672	        32.33 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInto/Active-8         	36558476	        32.34 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInto/Active-8         	37058148	        32.44 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInPlace/Pure-8        	12668032	        94.39 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInPlace/Pure-8        	12699997	        94.47 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInPlace/Pure-8        	12717152	        94.38 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInPlace/Pure-8        	12713929	        94.33 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInPlace/Pure-8        	12707096	        94.49 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInPlace/Active-8      	45987141	        26.03 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInPlace/Active-8      	45606422	        26.06 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInPlace/Active-8      	46110764	        26.05 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInPlace/Active-8      	46078599	        26.03 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven17x19/MultiplyScalarInPlace/Active-8      	46059439	        26.08 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Pure-8                    	   45672	     26247 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Pure-8                    	   45691	     26237 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Pure-8                    	   45705	     26252 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Pure-8                    	   45714	     26255 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Pure-8                    	   45704	     26234 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Active-8                  	  211423	      7087 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Active-8                  	  168595	      6919 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Active-8                  	  169136	      7079 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Active-8                  	  183226	      6791 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddInto/Active-8                  	  168770	      7072 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Pure-8               	   45649	     26311 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Pure-8               	   41997	     26355 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Pure-8               	   45673	     26277 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Pure-8               	   45729	     26254 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Pure-8               	   45606	     27292 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Active-8             	  219231	      5103 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Active-8             	  168688	      7079 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Active-8             	  168856	      7049 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Active-8             	  173725	      7031 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/SubtractInto/Active-8             	  168882	      7079 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyElementsInto/Pure-8       	   45783	     26255 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyElementsInto/Pure-8       	   45735	     26287 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyElementsInto/Pure-8       	   45754	     26240 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyElementsInto/Pure-8       	   45559	     26246 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyElementsInto/Pure-8       	   41884	     26253 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyElementsInto/Active-8     	  168862	      7086 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyElementsInto/Active-8     	  220156	      7082 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyElementsInto/Active-8     	  169159	      7077 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyElementsInto/Active-8     	  168933	      7080 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyElementsInto/Active-8     	  220658	      7064 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Pure-8           	   66679	     17979 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Pure-8           	   66714	     17951 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Pure-8           	   66285	     17950 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Pure-8           	   66638	     17949 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Pure-8           	   66694	     17964 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Active-8         	   66699	     17956 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Active-8         	   66510	     17951 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Active-8         	   65928	     17950 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Active-8         	   66561	     17967 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScaledInPlace/Active-8         	   66642	     17952 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Pure-8              	   68319	     17480 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Pure-8              	   60544	     17533 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Pure-8              	   68569	     17468 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Pure-8              	   68551	     17485 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Pure-8              	   68618	     17492 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Active-8            	  249892	      4804 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Active-8            	  249082	      4562 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Active-8            	  250568	      4814 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Active-8            	  244369	      4566 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/AddScalarInto/Active-8            	  249404	      4561 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInto/Pure-8         	   68493	     17522 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInto/Pure-8         	   68490	     17893 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInto/Pure-8         	   68518	     17476 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInto/Pure-8         	   68426	     17482 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInto/Pure-8         	   67198	     17469 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInto/Active-8       	  245446	      4817 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInto/Active-8       	  248992	      4624 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInto/Active-8       	  247172	      4807 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInto/Active-8       	  241549	      4819 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInto/Active-8       	  248562	      4779 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInPlace/Pure-8      	   61045	     17463 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInPlace/Pure-8      	   68661	     17466 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInPlace/Pure-8      	   68736	     17470 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInPlace/Pure-8      	   68697	     17473 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInPlace/Pure-8      	   68608	     17478 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInPlace/Active-8    	  269764	      4409 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInPlace/Active-8    	  272299	      4419 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInPlace/Active-8    	  271554	      4411 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInPlace/Active-8    	  269470	      4408 ns/op	       0 B/op	       0 allocs/op
Benchmark_ElementwiseCandidates/Uneven255x257/MultiplyScalarInPlace/Active-8    	  269186	      4413 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMul-8                                                              	    7460	    153216 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMul-8                                                              	    7498	    158190 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMul-8                                                              	    7330	    157382 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMul-8                                                              	    7812	    159452 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMul-8                                                              	    7620	    157997 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMulInto-8                                                          	    7352	    160301 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulInto-8                                                          	    7396	    163252 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulInto-8                                                          	    7249	    162914 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulInto-8                                                          	    7314	    155187 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulInto-8                                                          	    8036	    156485 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulLeftTransposeInto-8                                             	    8005	    149768 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulLeftTransposeInto-8                                             	    8043	    149864 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulLeftTransposeInto-8                                             	    8006	    149693 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulLeftTransposeInto-8                                             	    8013	    149745 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulLeftTransposeInto-8                                             	    7984	    149638 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeInto-8                                            	    7447	    161775 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeInto-8                                            	    7425	    161737 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeInto-8                                            	    7447	    161742 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeInto-8                                            	    7423	    161438 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeInto-8                                            	    7425	    161407 ns/op	       0 B/op	       0 allocs/op
Benchmark_Clone-8                                                               	   82557	     15723 ns/op	  262193 B/op	       2 allocs/op
Benchmark_Clone-8                                                               	   78664	     16273 ns/op	  262193 B/op	       2 allocs/op
Benchmark_Clone-8                                                               	   70536	     16399 ns/op	  262193 B/op	       2 allocs/op
Benchmark_Clone-8                                                               	   77390	     17139 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Clone-8                                                               	   70772	     16246 ns/op	  262193 B/op	       2 allocs/op
Benchmark_Values-8                                                              	   70149	     16633 ns/op	  262144 B/op	       1 allocs/op
Benchmark_Values-8                                                              	   69645	     16182 ns/op	  262145 B/op	       1 allocs/op
Benchmark_Values-8                                                              	   70837	     15875 ns/op	  262146 B/op	       1 allocs/op
Benchmark_Values-8                                                              	   68308	     16209 ns/op	  262145 B/op	       1 allocs/op
Benchmark_Values-8                                                              	   70533	     15855 ns/op	  262145 B/op	       1 allocs/op
Benchmark_Add-8                                                                 	   75565	     15924 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Add-8                                                                 	   76022	     15394 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Add-8                                                                 	   71138	     15986 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Add-8                                                                 	   62259	     17866 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Add-8                                                                 	   70414	     17174 ns/op	  262192 B/op	       2 allocs/op
Benchmark_AddInto-8                                                             	  225441	      6279 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddInto-8                                                             	  236378	      5659 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddInto-8                                                             	  232855	      5326 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddInto-8                                                             	  167898	      6724 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddInto-8                                                             	  215070	      5700 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddInPlace-8                                                          	  205675	      5794 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddInPlace-8                                                          	  205867	      5794 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddInPlace-8                                                          	  206359	      5811 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddInPlace-8                                                          	  207106	      5794 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddInPlace-8                                                          	  206841	      6024 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScaledInPlace-8                                                    	   67641	     17595 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScaledInPlace-8                                                    	   68078	     17591 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScaledInPlace-8                                                    	   66591	     17574 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScaledInPlace-8                                                    	   68037	     17593 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScaledInPlace-8                                                    	   68277	     17583 ns/op	       0 B/op	       0 allocs/op
Benchmark_Subtract-8                                                            	   74222	     16620 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Subtract-8                                                            	   74337	     16146 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Subtract-8                                                            	   70921	     16556 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Subtract-8                                                            	   65445	     17646 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Subtract-8                                                            	   74266	     16517 ns/op	  262192 B/op	       2 allocs/op
Benchmark_SubtractInto-8                                                        	  224259	      5570 ns/op	       0 B/op	       0 allocs/op
Benchmark_SubtractInto-8                                                        	  178182	      6254 ns/op	       0 B/op	       0 allocs/op
Benchmark_SubtractInto-8                                                        	  169198	      6161 ns/op	       0 B/op	       0 allocs/op
Benchmark_SubtractInto-8                                                        	  168505	      6237 ns/op	       0 B/op	       0 allocs/op
Benchmark_SubtractInto-8                                                        	  176305	      6172 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyElements-8                                                    	   74829	     16454 ns/op	  262192 B/op	       2 allocs/op
Benchmark_MultiplyElements-8                                                    	   76663	     16341 ns/op	  262192 B/op	       2 allocs/op
Benchmark_MultiplyElements-8                                                    	   63645	     16558 ns/op	  262192 B/op	       2 allocs/op
Benchmark_MultiplyElements-8                                                    	   67592	     16019 ns/op	  262192 B/op	       2 allocs/op
Benchmark_MultiplyElements-8                                                    	   70467	     17712 ns/op	  262192 B/op	       2 allocs/op
Benchmark_MultiplyElementsInto-8                                                	  233076	      6742 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyElementsInto-8                                                	  169572	      6765 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyElementsInto-8                                                	  226311	      5351 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyElementsInto-8                                                	  171154	      6491 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyElementsInto-8                                                	  214077	      4992 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideElements-8                                                      	   21018	     57690 ns/op	  262192 B/op	       2 allocs/op
Benchmark_DivideElements-8                                                      	   21056	     56576 ns/op	  262192 B/op	       2 allocs/op
Benchmark_DivideElements-8                                                      	   21034	     58520 ns/op	  262192 B/op	       2 allocs/op
Benchmark_DivideElements-8                                                      	   19982	     56712 ns/op	  262192 B/op	       2 allocs/op
Benchmark_DivideElements-8                                                      	   20350	     56830 ns/op	  262192 B/op	       2 allocs/op
Benchmark_DivideElementsInto-8                                                  	   27478	     43700 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideElementsInto-8                                                  	   27406	     43717 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideElementsInto-8                                                  	   27415	     43690 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideElementsInto-8                                                  	   27460	     43672 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideElementsInto-8                                                  	   27454	     43718 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScalar-8                                                           	   80330	     16024 ns/op	  262192 B/op	       2 allocs/op
Benchmark_AddScalar-8                                                           	   72228	     16631 ns/op	  262192 B/op	       2 allocs/op
Benchmark_AddScalar-8                                                           	   75775	     15680 ns/op	  262192 B/op	       2 allocs/op
Benchmark_AddScalar-8                                                           	   74826	     16082 ns/op	  262192 B/op	       2 allocs/op
Benchmark_AddScalar-8                                                           	   75638	     16166 ns/op	  262192 B/op	       2 allocs/op
Benchmark_AddScalarInto-8                                                       	  251106	      4416 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScalarInto-8                                                       	  254935	      4408 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScalarInto-8                                                       	  255571	      4437 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScalarInto-8                                                       	  254317	      4430 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddScalarInto-8                                                       	  251838	      4481 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalar-8                                                      	   75284	     16303 ns/op	  262192 B/op	       2 allocs/op
Benchmark_MultiplyScalar-8                                                      	   75266	     15694 ns/op	  262192 B/op	       2 allocs/op
Benchmark_MultiplyScalar-8                                                      	   66490	     15948 ns/op	  262192 B/op	       2 allocs/op
Benchmark_MultiplyScalar-8                                                      	   69776	     16709 ns/op	  262192 B/op	       2 allocs/op
Benchmark_MultiplyScalar-8                                                      	   70048	     16287 ns/op	  262192 B/op	       2 allocs/op
Benchmark_MultiplyScalarInto-8                                                  	  253465	      4488 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInto-8                                                  	  251532	      4711 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInto-8                                                  	  252019	      4410 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInto-8                                                  	  252782	      4532 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInto-8                                                  	  252888	      4415 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInPlace-8                                               	  271659	      4415 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInPlace-8                                               	  271458	      4431 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInPlace-8                                               	  271737	      4499 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInPlace-8                                               	  271317	      4419 ns/op	       0 B/op	       0 allocs/op
Benchmark_MultiplyScalarInPlace-8                                               	  271842	      4422 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideScalar-8                                                        	   29430	     38108 ns/op	  262192 B/op	       2 allocs/op
Benchmark_DivideScalar-8                                                        	   31101	     41715 ns/op	  262192 B/op	       2 allocs/op
Benchmark_DivideScalar-8                                                        	   30168	     39313 ns/op	  262192 B/op	       2 allocs/op
Benchmark_DivideScalar-8                                                        	   28212	     42020 ns/op	  262192 B/op	       2 allocs/op
Benchmark_DivideScalar-8                                                        	   29572	     40041 ns/op	  262192 B/op	       2 allocs/op
Benchmark_DivideScalarInto-8                                                    	   45590	     26203 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideScalarInto-8                                                    	   45738	     26343 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideScalarInto-8                                                    	   45698	     26815 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideScalarInto-8                                                    	   45768	     26233 ns/op	       0 B/op	       0 allocs/op
Benchmark_DivideScalarInto-8                                                    	   45810	     26225 ns/op	       0 B/op	       0 allocs/op
Benchmark_Transpose-8                                                           	   44124	     26989 ns/op	  131120 B/op	       2 allocs/op
Benchmark_Transpose-8                                                           	   44588	     26558 ns/op	  131120 B/op	       2 allocs/op
Benchmark_Transpose-8                                                           	   45792	     26683 ns/op	  131120 B/op	       2 allocs/op
Benchmark_Transpose-8                                                           	   46216	     26435 ns/op	  131120 B/op	       2 allocs/op
Benchmark_Transpose-8                                                           	   44487	     26964 ns/op	  131120 B/op	       2 allocs/op
Benchmark_TransposeInto-8                                                       	   49090	     24912 ns/op	       0 B/op	       0 allocs/op
Benchmark_TransposeInto-8                                                       	   50086	     23373 ns/op	       0 B/op	       0 allocs/op
Benchmark_TransposeInto-8                                                       	   59269	     24090 ns/op	       0 B/op	       0 allocs/op
Benchmark_TransposeInto-8                                                       	   51696	     25371 ns/op	       0 B/op	       0 allocs/op
Benchmark_TransposeInto-8                                                       	   50678	     24513 ns/op	       0 B/op	       0 allocs/op
Benchmark_RowSums-8                                                             	   24178	     46492 ns/op	    1024 B/op	       1 allocs/op
Benchmark_RowSums-8                                                             	   25552	     50318 ns/op	    1024 B/op	       1 allocs/op
Benchmark_RowSums-8                                                             	   24886	     47620 ns/op	    1024 B/op	       1 allocs/op
Benchmark_RowSums-8                                                             	   26364	     47355 ns/op	    1024 B/op	       1 allocs/op
Benchmark_RowSums-8                                                             	   24193	     49865 ns/op	    1024 B/op	       1 allocs/op
Benchmark_RowSumsInto-8                                                         	   27273	     42739 ns/op	       0 B/op	       0 allocs/op
Benchmark_RowSumsInto-8                                                         	   27176	     43253 ns/op	       0 B/op	       0 allocs/op
Benchmark_RowSumsInto-8                                                         	   28040	     42791 ns/op	       0 B/op	       0 allocs/op
Benchmark_RowSumsInto-8                                                         	   28063	     42747 ns/op	       0 B/op	       0 allocs/op
Benchmark_RowSumsInto-8                                                         	   28057	     42766 ns/op	       0 B/op	       0 allocs/op
Benchmark_ColumnSums-8                                                          	   41734	     28670 ns/op	    1024 B/op	       1 allocs/op
Benchmark_ColumnSums-8                                                          	   41755	     28734 ns/op	    1024 B/op	       1 allocs/op
Benchmark_ColumnSums-8                                                          	   38218	     28766 ns/op	    1024 B/op	       1 allocs/op
Benchmark_ColumnSums-8                                                          	   41708	     28741 ns/op	    1024 B/op	       1 allocs/op
Benchmark_ColumnSums-8                                                          	   41764	     28773 ns/op	    1024 B/op	       1 allocs/op
Benchmark_ColumnSumsInto-8                                                      	  188089	      6358 ns/op	       0 B/op	       0 allocs/op
Benchmark_ColumnSumsInto-8                                                      	  189363	      6331 ns/op	       0 B/op	       0 allocs/op
Benchmark_ColumnSumsInto-8                                                      	  186116	      6327 ns/op	       0 B/op	       0 allocs/op
Benchmark_ColumnSumsInto-8                                                      	  188844	      6328 ns/op	       0 B/op	       0 allocs/op
Benchmark_ColumnSumsInto-8                                                      	  189242	      6325 ns/op	       0 B/op	       0 allocs/op
Benchmark_AccumulateColumnSumsInto-8                                            	  188119	      6355 ns/op	       0 B/op	       0 allocs/op
Benchmark_AccumulateColumnSumsInto-8                                            	  188838	      6345 ns/op	       0 B/op	       0 allocs/op
Benchmark_AccumulateColumnSumsInto-8                                            	  188463	      6336 ns/op	       0 B/op	       0 allocs/op
Benchmark_AccumulateColumnSumsInto-8                                            	  183570	      6352 ns/op	       0 B/op	       0 allocs/op
Benchmark_AccumulateColumnSumsInto-8                                            	  189075	      6344 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddRowVectorInPlace-8                                                 	   34507	     34259 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddRowVectorInPlace-8                                                 	   35035	     34322 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddRowVectorInPlace-8                                                 	   35097	     34566 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddRowVectorInPlace-8                                                 	   35128	     34348 ns/op	       0 B/op	       0 allocs/op
Benchmark_AddRowVectorInPlace-8                                                 	   32779	     34340 ns/op	       0 B/op	       0 allocs/op
Benchmark_Apply-8                                                               	   17768	     67261 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Apply-8                                                               	   17174	     69643 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Apply-8                                                               	   17815	     67299 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Apply-8                                                               	   17833	     67356 ns/op	  262192 B/op	       2 allocs/op
Benchmark_Apply-8                                                               	   17733	     67252 ns/op	  262192 B/op	       2 allocs/op
Benchmark_ApplyInto-8                                                           	   17178	     69820 ns/op	       0 B/op	       0 allocs/op
Benchmark_ApplyInto-8                                                           	   17100	     71312 ns/op	       0 B/op	       0 allocs/op
Benchmark_ApplyInto-8                                                           	   17042	     69706 ns/op	       0 B/op	       0 allocs/op
Benchmark_ApplyInto-8                                                           	   17236	     69680 ns/op	       0 B/op	       0 allocs/op
Benchmark_ApplyInto-8                                                           	   17246	     69614 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulShapes/Small2x2-8                                               	28916330	        40.88 ns/op	      64 B/op	       2 allocs/op
Benchmark_MatMulShapes/Small2x2-8                                               	29629477	        40.40 ns/op	      64 B/op	       2 allocs/op
Benchmark_MatMulShapes/Small2x2-8                                               	29580511	        40.77 ns/op	      64 B/op	       2 allocs/op
Benchmark_MatMulShapes/Small2x2-8                                               	29445051	        40.45 ns/op	      64 B/op	       2 allocs/op
Benchmark_MatMulShapes/Small2x2-8                                               	29722098	        40.02 ns/op	      64 B/op	       2 allocs/op
Benchmark_MatMulShapes/Small4x4-8                                               	11743291	       102.2 ns/op	     112 B/op	       2 allocs/op
Benchmark_MatMulShapes/Small4x4-8                                               	11724608	       101.7 ns/op	     112 B/op	       2 allocs/op
Benchmark_MatMulShapes/Small4x4-8                                               	11714902	       102.1 ns/op	     112 B/op	       2 allocs/op
Benchmark_MatMulShapes/Small4x4-8                                               	11806887	       102.0 ns/op	     112 B/op	       2 allocs/op
Benchmark_MatMulShapes/Small4x4-8                                               	11768359	       102.0 ns/op	     112 B/op	       2 allocs/op
Benchmark_MatMulShapes/Medium64x64-8                                            	    7326	    154723 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMulShapes/Medium64x64-8                                            	    8002	    154130 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMulShapes/Medium64x64-8                                            	    7390	    160329 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMulShapes/Medium64x64-8                                            	    7354	    162400 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMulShapes/Medium64x64-8                                            	    7371	    161431 ns/op	   16432 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8                                       	     486	   2469817 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8                                       	     483	   2473730 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8                                       	     486	   2465094 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8                                       	     486	   2466042 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulShapes/Large128x256x128-8                                       	     487	   2467093 ns/op	   65584 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven17x33x19-8                                         	  172414	      6967 ns/op	    1456 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven17x33x19-8                                         	  171628	      6950 ns/op	    1456 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven17x33x19-8                                         	  174166	      6901 ns/op	    1456 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven17x33x19-8                                         	  174472	      6899 ns/op	    1456 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven17x33x19-8                                         	  174481	      6892 ns/op	    1456 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven63x65x31-8                                         	   15621	     77002 ns/op	    8240 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven63x65x31-8                                         	   15610	     76820 ns/op	    8240 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven63x65x31-8                                         	   15585	     77230 ns/op	    8240 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven63x65x31-8                                         	   15610	     76835 ns/op	    8240 B/op	       2 allocs/op
Benchmark_MatMulShapes/Uneven63x65x31-8                                         	   15604	     76791 ns/op	    8240 B/op	       2 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small2x2-8                             	60327022	        20.19 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small2x2-8                             	56080662	        20.19 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small2x2-8                             	59876751	        20.14 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small2x2-8                             	59817057	        20.09 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small2x2-8                             	58570395	        20.14 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small4x4-8                             	15117109	        79.39 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small4x4-8                             	14233467	        80.13 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small4x4-8                             	15089088	        80.30 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small4x4-8                             	15125151	        80.18 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Small4x4-8                             	14892778	        80.21 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Medium64x64-8                          	    7626	    156303 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Medium64x64-8                          	    7671	    156175 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Medium64x64-8                          	    7688	    156268 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Medium64x64-8                          	    7671	    156200 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Medium64x64-8                          	    7688	    156031 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8                     	     400	   2870475 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8                     	     422	   2844474 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8                     	     420	   2849024 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8                     	     420	   2864397 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Large128x256x128-8                     	     414	   2835143 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven17x33x19-8                       	  171651	      6827 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven17x33x19-8                       	  175935	      6835 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven17x33x19-8                       	  176163	      6824 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven17x33x19-8                       	  175994	      6820 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven17x33x19-8                       	  175701	      6823 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven63x65x31-8                       	   15483	     77443 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven63x65x31-8                       	   15482	     77493 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven63x65x31-8                       	   15508	     77555 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven63x65x31-8                       	   15513	     77657 ns/op	       0 B/op	       0 allocs/op
Benchmark_MatMulRightTransposeIntoShapes/Uneven63x65x31-8                       	   15510	     77458 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/RowSumsInto-8                            	174059174	         6.897 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/RowSumsInto-8                            	157285513	         6.897 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/RowSumsInto-8                            	174003554	         6.918 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/RowSumsInto-8                            	173896377	         6.896 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/RowSumsInto-8                            	173954798	         6.916 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/ColumnSumsInto-8                         	156017100	         7.693 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/ColumnSumsInto-8                         	154539144	         7.703 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/ColumnSumsInto-8                         	155987126	         7.727 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/ColumnSumsInto-8                         	155474138	         7.701 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/ColumnSumsInto-8                         	155934594	         7.703 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/AccumulateColumnSumsInto-8               	173983130	         6.899 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/AccumulateColumnSumsInto-8               	173999506	         6.898 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/AccumulateColumnSumsInto-8               	173917075	         6.895 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/AccumulateColumnSumsInto-8               	173169584	         6.901 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x1/AccumulateColumnSumsInto-8               	164776720	         6.898 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/RowSumsInto-8                            	149130088	         8.064 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/RowSumsInto-8                            	149345035	         8.077 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/RowSumsInto-8                            	149107250	         8.085 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/RowSumsInto-8                            	149082181	         8.063 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/RowSumsInto-8                            	149585989	         8.095 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/ColumnSumsInto-8                         	139447566	         8.592 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/ColumnSumsInto-8                         	136240020	         8.500 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/ColumnSumsInto-8                         	141243865	         8.705 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/ColumnSumsInto-8                         	138664768	         8.678 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/ColumnSumsInto-8                         	139096526	         8.581 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/AccumulateColumnSumsInto-8               	146125141	         8.133 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/AccumulateColumnSumsInto-8               	146728938	         8.209 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/AccumulateColumnSumsInto-8               	147943971	         8.142 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/AccumulateColumnSumsInto-8               	147324901	         8.166 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small1x3/AccumulateColumnSumsInto-8               	146356232	         8.155 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/RowSumsInto-8                            	132288985	         9.118 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/RowSumsInto-8                            	132214947	         9.208 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/RowSumsInto-8                            	132419394	         9.097 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/RowSumsInto-8                            	132222055	         9.093 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/RowSumsInto-8                            	131938813	         9.081 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/ColumnSumsInto-8                         	121767588	         9.873 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/ColumnSumsInto-8                         	122137580	         9.818 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/ColumnSumsInto-8                         	122294452	         9.836 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/ColumnSumsInto-8                         	122200530	         9.837 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/ColumnSumsInto-8                         	122279368	         9.830 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/AccumulateColumnSumsInto-8               	133027849	         9.033 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/AccumulateColumnSumsInto-8               	133012802	         9.086 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/AccumulateColumnSumsInto-8               	132598588	         9.148 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/AccumulateColumnSumsInto-8               	131923939	         9.033 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Small3x1/AccumulateColumnSumsInto-8               	133098002	         9.026 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/RowSumsInto-8                         	  522775	      2286 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/RowSumsInto-8                         	  524478	      2287 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/RowSumsInto-8                         	  524356	      2836 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/RowSumsInto-8                         	  524294	      2286 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/RowSumsInto-8                         	  523788	      2289 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/ColumnSumsInto-8                      	 1921384	       622.1 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/ColumnSumsInto-8                      	 1917536	       625.1 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/ColumnSumsInto-8                      	 1929870	       625.4 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/ColumnSumsInto-8                      	 1915774	       621.5 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/ColumnSumsInto-8                      	 1930380	       625.3 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/AccumulateColumnSumsInto-8            	 1944463	       618.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/AccumulateColumnSumsInto-8            	 1934205	       618.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/AccumulateColumnSumsInto-8            	 1941697	       622.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/AccumulateColumnSumsInto-8            	 1932224	       617.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium64x64/AccumulateColumnSumsInto-8            	 1939322	       621.4 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/RowSumsInto-8                       	   55966	     21437 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/RowSumsInto-8                       	   56218	     21511 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/RowSumsInto-8                       	   56065	     22509 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/RowSumsInto-8                       	   56148	     21357 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/RowSumsInto-8                       	   56150	     21346 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/ColumnSumsInto-8                    	  380607	      3139 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/ColumnSumsInto-8                    	  381883	      3148 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/ColumnSumsInto-8                    	  382417	      3140 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/ColumnSumsInto-8                    	  382651	      3152 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/ColumnSumsInto-8                    	  381495	      3142 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/AccumulateColumnSumsInto-8          	  382194	      3138 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/AccumulateColumnSumsInto-8          	  381745	      3146 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/AccumulateColumnSumsInto-8          	  341731	      3208 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/AccumulateColumnSumsInto-8          	  382166	      3154 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Medium128x256/AccumulateColumnSumsInto-8          	  382935	      3134 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/RowSumsInto-8                     	  262279	      4559 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/RowSumsInto-8                     	  263122	      4565 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/RowSumsInto-8                     	  263323	      4566 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/RowSumsInto-8                     	  262852	      4563 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/RowSumsInto-8                     	  262788	      5894 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/ColumnSumsInto-8                  	  976204	      1242 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/ColumnSumsInto-8                  	  976700	      1238 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/ColumnSumsInto-8                  	  974124	      1240 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/ColumnSumsInto-8                  	  973593	      1239 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/ColumnSumsInto-8                  	  972742	      1240 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/AccumulateColumnSumsInto-8        	  980478	      1224 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/AccumulateColumnSumsInto-8        	  978818	      1233 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/AccumulateColumnSumsInto-8        	  977539	      1233 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/AccumulateColumnSumsInto-8        	  980629	      1233 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/DenseBias128x64/AccumulateColumnSumsInto-8        	  978264	      1232 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/RowSumsInto-8                        	    6945	    172426 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/RowSumsInto-8                        	    6962	    173616 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/RowSumsInto-8                        	    6962	    172199 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/RowSumsInto-8                        	    6963	    172841 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/RowSumsInto-8                        	    6946	    172281 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/ColumnSumsInto-8                     	   51853	     23087 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/ColumnSumsInto-8                     	   51873	     23134 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/ColumnSumsInto-8                     	   51824	     23105 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/ColumnSumsInto-8                     	   51925	     23133 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/ColumnSumsInto-8                     	   51932	     23131 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/AccumulateColumnSumsInto-8           	   52011	     23061 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/AccumulateColumnSumsInto-8           	   51724	     23191 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/AccumulateColumnSumsInto-8           	   51818	     23164 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/AccumulateColumnSumsInto-8           	   51744	     23150 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Large512x512/AccumulateColumnSumsInto-8           	   51868	     23155 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/RowSumsInto-8                        	  421296	      2847 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/RowSumsInto-8                        	  420999	      2847 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/RowSumsInto-8                        	  420495	      2859 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/RowSumsInto-8                        	  420848	      2849 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/RowSumsInto-8                        	  420188	      2849 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/ColumnSumsInto-8                     	 2796808	       429.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/ColumnSumsInto-8                     	 2781444	       431.9 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/ColumnSumsInto-8                     	 2795290	       425.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/ColumnSumsInto-8                     	 2783434	       429.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/ColumnSumsInto-8                     	 2794627	       424.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/AccumulateColumnSumsInto-8           	 2873811	       424.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/AccumulateColumnSumsInto-8           	 2823126	       422.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/AccumulateColumnSumsInto-8           	 2867497	       423.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/AccumulateColumnSumsInto-8           	 2836196	       418.1 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven17x257/AccumulateColumnSumsInto-8           	 2872459	       423.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/RowSumsInto-8                        	  542619	      2209 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/RowSumsInto-8                        	  543163	      2209 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/RowSumsInto-8                        	  543560	      2209 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/RowSumsInto-8                        	  535422	      2208 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/RowSumsInto-8                        	  544442	      2208 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/ColumnSumsInto-8                     	  832083	      1439 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/ColumnSumsInto-8                     	  845757	      1449 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/ColumnSumsInto-8                     	  832795	      1448 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/ColumnSumsInto-8                     	  837807	      1455 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/ColumnSumsInto-8                     	  841305	      1451 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/AccumulateColumnSumsInto-8           	  835353	      1454 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/AccumulateColumnSumsInto-8           	  833848	      1460 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/AccumulateColumnSumsInto-8           	  837265	      1460 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/AccumulateColumnSumsInto-8           	  839941	      1469 ns/op	       0 B/op	       0 allocs/op
Benchmark_ReductionCandidates/Uneven257x17/AccumulateColumnSumsInto-8           	  827210	      1462 ns/op	       0 B/op	       0 allocs/op
Benchmark_SoftmaxRowsInto_MediumBatch-8                                         	   18964	     63266 ns/op	       0 B/op	       0 allocs/op
Benchmark_SoftmaxRowsInto_MediumBatch-8                                         	   18919	     63723 ns/op	       0 B/op	       0 allocs/op
Benchmark_SoftmaxRowsInto_MediumBatch-8                                         	   18961	     63449 ns/op	       0 B/op	       0 allocs/op
Benchmark_SoftmaxRowsInto_MediumBatch-8                                         	   18950	     63378 ns/op	       0 B/op	       0 allocs/op
Benchmark_SoftmaxRowsInto_MediumBatch-8                                         	   18951	     63257 ns/op	       0 B/op	       0 allocs/op
Benchmark_SoftmaxRowsBackwardInto_MediumBatch-8                                 	   16196	     74136 ns/op	       0 B/op	       0 allocs/op
Benchmark_SoftmaxRowsBackwardInto_MediumBatch-8                                 	   16192	     74150 ns/op	       0 B/op	       0 allocs/op
Benchmark_SoftmaxRowsBackwardInto_MediumBatch-8                                 	   16192	     74128 ns/op	       0 B/op	       0 allocs/op
Benchmark_SoftmaxRowsBackwardInto_MediumBatch-8                                 	   16189	     74101 ns/op	       0 B/op	       0 allocs/op
Benchmark_SoftmaxRowsBackwardInto_MediumBatch-8                                 	   16134	     74191 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/matrix	1497.053s
```

### metric

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/metric
cpu: Apple M3
Benchmark_MetricValue/MeanSquaredError/Small-8         	11615925	       103.4 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/MeanSquaredError/Small-8         	11618721	       103.3 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/MeanSquaredError/Small-8         	11598409	       104.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/MeanSquaredError/Small-8         	11605396	       103.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/MeanSquaredError/Small-8         	11652921	       103.7 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/MeanSquaredError/Medium-8        	  431761	      2771 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/MeanSquaredError/Medium-8        	  431971	      2778 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/MeanSquaredError/Medium-8        	  432092	      2775 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/MeanSquaredError/Medium-8        	  431581	      2776 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/MeanSquaredError/Medium-8        	  432015	      2778 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryAccuracy/Small-8           	35683486	        32.78 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryAccuracy/Small-8           	35961925	        32.70 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryAccuracy/Small-8           	36132243	        32.73 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryAccuracy/Small-8           	36705711	        32.75 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryAccuracy/Small-8           	36587317	        32.70 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryAccuracy/Medium-8          	 1982538	       607.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryAccuracy/Medium-8          	 1984798	       609.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryAccuracy/Medium-8          	 1987173	       603.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryAccuracy/Medium-8          	 1984321	       603.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryAccuracy/Medium-8          	 1991236	       602.8 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryPrecision/Small-8          	36371994	        32.85 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryPrecision/Small-8          	36302628	        33.08 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryPrecision/Small-8          	36231381	        32.77 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryPrecision/Small-8          	36444208	        33.42 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryPrecision/Small-8          	35825664	        32.77 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryPrecision/Medium-8         	 1983973	       609.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryPrecision/Medium-8         	 1990938	       603.5 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryPrecision/Medium-8         	 1988472	       611.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryPrecision/Medium-8         	 1985851	       603.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryPrecision/Medium-8         	 1984972	       602.5 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryRecall/Small-8             	36028282	        32.67 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryRecall/Small-8             	36298372	        32.67 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryRecall/Small-8             	35348308	        32.67 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryRecall/Small-8             	36641173	        32.67 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryRecall/Small-8             	36613225	        32.71 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryRecall/Medium-8            	 1989195	       606.5 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryRecall/Medium-8            	 1959181	       608.9 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryRecall/Medium-8            	 1989771	       603.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryRecall/Medium-8            	 1990431	       602.3 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryRecall/Medium-8            	 1991028	       603.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryF1/Small-8                 	34799546	        33.48 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryF1/Small-8                 	36018729	        33.33 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryF1/Small-8                 	35983627	        33.30 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryF1/Small-8                 	34636407	        33.29 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryF1/Small-8                 	35674293	        33.36 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryF1/Medium-8                	 1966824	       614.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryF1/Medium-8                	 1943448	       604.5 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryF1/Medium-8                	 1989100	       603.4 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryF1/Medium-8                	 1986878	       605.4 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/BinaryF1/Medium-8                	 1985908	       604.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/CategoricalAccuracy/Small-8      	 6805778	       177.6 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/CategoricalAccuracy/Small-8      	 6739600	       177.4 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/CategoricalAccuracy/Small-8      	 6741433	       177.3 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/CategoricalAccuracy/Small-8      	 6735536	       180.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/CategoricalAccuracy/Small-8      	 6650604	       177.9 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/CategoricalAccuracy/Medium-8     	  214891	      5348 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/CategoricalAccuracy/Medium-8     	  225349	      5280 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/CategoricalAccuracy/Medium-8     	  229459	      5306 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/CategoricalAccuracy/Medium-8     	  224689	      5241 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/CategoricalAccuracy/Medium-8     	  227234	      5245 ns/op	       0 B/op	       0 allocs/op
Benchmark_MetricValue/CategoricalMacroPrecision/Small-8         	 2927066	       406.6 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroPrecision/Small-8         	 2969948	       406.3 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroPrecision/Small-8         	 2957756	       405.3 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroPrecision/Small-8         	 2957262	       405.2 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroPrecision/Small-8         	 2918716	       407.1 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroPrecision/Medium-8        	  217108	      5495 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroPrecision/Medium-8        	  219189	      5470 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroPrecision/Medium-8        	  219621	      5479 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroPrecision/Medium-8        	  219265	      5502 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroPrecision/Medium-8        	  216255	      5496 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroRecall/Small-8            	 3179354	       378.2 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroRecall/Small-8            	 3180889	       378.1 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroRecall/Small-8            	 3176440	       376.8 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroRecall/Small-8            	 3151615	       377.2 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroRecall/Small-8            	 3190131	       378.1 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroRecall/Medium-8           	  215154	      5486 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroRecall/Medium-8           	  217495	      5483 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroRecall/Medium-8           	  219392	      5489 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroRecall/Medium-8           	  219381	      5485 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroRecall/Medium-8           	  217356	      5485 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroF1/Small-8                	 2316014	       514.9 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroF1/Small-8                	 2288776	       512.7 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroF1/Small-8                	 2325924	       512.9 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroF1/Small-8                	 2337738	       513.4 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroF1/Small-8                	 2337350	       517.1 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroF1/Medium-8               	  210216	      5660 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroF1/Medium-8               	  209324	      5645 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroF1/Medium-8               	  212422	      5703 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroF1/Medium-8               	  211852	      5654 ns/op	    2048 B/op	       1 allocs/op
Benchmark_MetricValue/CategoricalMacroF1/Medium-8               	  212889	      5665 ns/op	    2048 B/op	       1 allocs/op
Benchmark_ConfusionMatrixConstruction/Binary/Small-8            	20506963	        57.75 ns/op	      80 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Binary/Small-8            	20953759	        57.75 ns/op	      80 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Binary/Small-8            	20932027	        57.72 ns/op	      80 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Binary/Small-8            	20856205	        57.35 ns/op	      80 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Binary/Small-8            	20909398	        57.33 ns/op	      80 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Binary/Medium-8           	 1909638	       630.3 ns/op	      80 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Binary/Medium-8           	 1906357	       635.1 ns/op	      80 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Binary/Medium-8           	 1899106	       631.7 ns/op	      80 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Binary/Medium-8           	 1895194	       631.0 ns/op	      80 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Binary/Medium-8           	 1898222	       633.5 ns/op	      80 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Categorical/Small-8       	 4074043	       298.6 ns/op	    2096 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Categorical/Small-8       	 4035468	       298.5 ns/op	    2096 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Categorical/Small-8       	 4025299	       299.2 ns/op	    2096 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Categorical/Small-8       	 4027069	       299.6 ns/op	    2096 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Categorical/Small-8       	 4052433	       303.9 ns/op	    2096 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Categorical/Medium-8      	  221281	      5425 ns/op	    2096 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Categorical/Medium-8      	  222144	      5405 ns/op	    2096 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Categorical/Medium-8      	  223101	      5390 ns/op	    2096 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Categorical/Medium-8      	  222782	      5383 ns/op	    2096 B/op	       2 allocs/op
Benchmark_ConfusionMatrixConstruction/Categorical/Medium-8      	  222667	      5389 ns/op	    2096 B/op	       2 allocs/op
Benchmark_ConfusionMatrixCounts_ColdPath-8                      	 3825466	       305.6 ns/op	    2432 B/op	      17 allocs/op
Benchmark_ConfusionMatrixCounts_ColdPath-8                      	 3925800	       305.5 ns/op	    2432 B/op	      17 allocs/op
Benchmark_ConfusionMatrixCounts_ColdPath-8                      	 3909234	       306.4 ns/op	    2432 B/op	      17 allocs/op
Benchmark_ConfusionMatrixCounts_ColdPath-8                      	 3916344	       305.9 ns/op	    2432 B/op	      17 allocs/op
Benchmark_ConfusionMatrixCounts_ColdPath-8                      	 3929842	       305.7 ns/op	    2432 B/op	      17 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/metric	167.979s
```

### model

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/model
cpu: Apple M3
Benchmark_SequentialTrainFitEpoch_Warmed-8              	 1290558	       919.2 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainFitEpoch_Warmed-8              	 1313197	       917.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainFitEpoch_Warmed-8              	 1319217	       914.9 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainFitEpoch_Warmed-8              	 1304542	       912.4 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainFitEpoch_Warmed-8              	 1296004	       926.0 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialParameters-8                        	41740159	        28.34 ns/op	      32 B/op	       1 allocs/op
Benchmark_SequentialParameters-8                        	43040962	        27.69 ns/op	      32 B/op	       1 allocs/op
Benchmark_SequentialParameters-8                        	43190324	        27.70 ns/op	      32 B/op	       1 allocs/op
Benchmark_SequentialParameters-8                        	42902407	        29.23 ns/op	      32 B/op	       1 allocs/op
Benchmark_SequentialParameters-8                        	42759532	        28.40 ns/op	      32 B/op	       1 allocs/op
Benchmark_SequentialTrainBatch_XOR-8                    	  838060	      1490 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_XOR-8                    	  812067	      1492 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_XOR-8                    	  820844	      1487 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_XOR-8                    	  829090	      1484 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_XOR-8                    	  819405	      1485 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialFit_XOR-8                           	  536836	      2234 ns/op	     368 B/op	      10 allocs/op
Benchmark_SequentialFit_XOR-8                           	  538854	      2233 ns/op	     368 B/op	      10 allocs/op
Benchmark_SequentialFit_XOR-8                           	  543051	      2232 ns/op	     368 B/op	      10 allocs/op
Benchmark_SequentialFit_XOR-8                           	  544881	      2230 ns/op	     368 B/op	      10 allocs/op
Benchmark_SequentialFit_XOR-8                           	  534164	      2228 ns/op	     368 B/op	      10 allocs/op
Benchmark_SequentialFit_XOR_Accuracy-8                  	  533005	      2275 ns/op	     368 B/op	      10 allocs/op
Benchmark_SequentialFit_XOR_Accuracy-8                  	  536077	      2270 ns/op	     368 B/op	      10 allocs/op
Benchmark_SequentialFit_XOR_Accuracy-8                  	  533307	      2270 ns/op	     368 B/op	      10 allocs/op
Benchmark_SequentialFit_XOR_Accuracy-8                  	  532749	      2270 ns/op	     368 B/op	      10 allocs/op
Benchmark_SequentialFit_XOR_Accuracy-8                  	  525346	      2269 ns/op	     368 B/op	      10 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8         	    1582	    758580 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8         	    1586	    758410 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8         	    1585	    758656 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8         	    1582	    757938 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_SyntheticDense-8         	    1585	    757844 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialFit_SyntheticDense-8                	    1129	   1035839 ns/op	   31984 B/op	      10 allocs/op
Benchmark_SequentialFit_SyntheticDense-8                	    1134	   1033894 ns/op	   31984 B/op	      10 allocs/op
Benchmark_SequentialFit_SyntheticDense-8                	    1162	   1033124 ns/op	   31984 B/op	      10 allocs/op
Benchmark_SequentialFit_SyntheticDense-8                	    1158	   1034361 ns/op	   31984 B/op	      10 allocs/op
Benchmark_SequentialFit_SyntheticDense-8                	    1161	   1033662 ns/op	   31984 B/op	      10 allocs/op
Benchmark_SequentialTrainBatch_Activations/ELU-8        	   53972	     22047 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/ELU-8        	   54211	     21978 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/ELU-8        	   54118	     21950 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/ELU-8        	   54060	     21947 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/ELU-8        	   54186	     22007 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/GELU-8       	   43110	     27762 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/GELU-8       	   43110	     27930 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/GELU-8       	   43159	     27773 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/GELU-8       	   43192	     27786 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/GELU-8       	   43096	     27774 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/LeakyReLU-8  	   54840	     21796 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/LeakyReLU-8  	   54690	     21763 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/LeakyReLU-8  	   54488	     21962 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/LeakyReLU-8  	   54768	     21985 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/LeakyReLU-8  	   54523	     21764 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Linear-8     	   58419	     20535 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Linear-8     	   58424	     20624 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Linear-8     	   58226	     20715 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Linear-8     	   57476	     20609 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Linear-8     	   57838	     20650 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/ReLU-8       	   54213	     22125 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/ReLU-8       	   54331	     22049 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/ReLU-8       	   54309	     22121 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/ReLU-8       	   54094	     22056 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/ReLU-8       	   54259	     22066 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Sigmoid-8    	   44371	     27092 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Sigmoid-8    	   44391	     27104 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Sigmoid-8    	   44328	     27377 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Sigmoid-8    	   44263	     27079 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Sigmoid-8    	   44289	     27173 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Tanh-8       	   47835	     24733 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Tanh-8       	   47628	     24700 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Tanh-8       	   47884	     24702 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Tanh-8       	   47955	     24679 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Tanh-8       	   47709	     24737 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Softmax-8    	   42703	     28033 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Softmax-8    	   42486	     27974 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Softmax-8    	   42747	     28059 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Softmax-8    	   42450	     28048 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Activations/Softmax-8    	   42558	     28026 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Regularized/L1-8         	    1572	    763605 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Regularized/L1-8         	    1572	    764659 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Regularized/L1-8         	    1572	    763899 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Regularized/L1-8         	    1563	    763867 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Regularized/L1-8         	    1573	    765281 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Regularized/L2-8         	    1579	    761642 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Regularized/L2-8         	    1579	    760662 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Regularized/L2-8         	    1578	    761680 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Regularized/L2-8         	    1575	    760787 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_Regularized/L2-8         	    1576	    765176 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_AlternatingShapes-8      	    3314	    362508 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_AlternatingShapes-8      	    3314	    364805 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_AlternatingShapes-8      	    3319	    362497 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_AlternatingShapes-8      	    3288	    361522 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialTrainBatch_AlternatingShapes-8      	    3316	    362376 ns/op	       0 B/op	       0 allocs/op
Benchmark_SequentialFit_SyntheticDense_ColdOneEpoch-8   	    1132	   1055721 ns/op	  262272 B/op	      48 allocs/op
Benchmark_SequentialFit_SyntheticDense_ColdOneEpoch-8   	    1136	   1054764 ns/op	  262272 B/op	      48 allocs/op
Benchmark_SequentialFit_SyntheticDense_ColdOneEpoch-8   	    1138	   1053798 ns/op	  262272 B/op	      48 allocs/op
Benchmark_SequentialFit_SyntheticDense_ColdOneEpoch-8   	    1140	   1053880 ns/op	  262272 B/op	      48 allocs/op
Benchmark_SequentialFit_SyntheticDense_ColdOneEpoch-8   	    1137	   1057365 ns/op	  262272 B/op	      48 allocs/op
Benchmark_SequentialFit_SyntheticDense_TenEpoch-8       	     100	  10356350 ns/op	   33184 B/op	      14 allocs/op
Benchmark_SequentialFit_SyntheticDense_TenEpoch-8       	     100	  10332930 ns/op	   33184 B/op	      14 allocs/op
Benchmark_SequentialFit_SyntheticDense_TenEpoch-8       	     100	  10390116 ns/op	   33184 B/op	      14 allocs/op
Benchmark_SequentialFit_SyntheticDense_TenEpoch-8       	     100	  10380257 ns/op	   33184 B/op	      14 allocs/op
Benchmark_SequentialFit_SyntheticDense_TenEpoch-8       	     100	  10461265 ns/op	   33184 B/op	      14 allocs/op
Benchmark_SequentialFit_Scenarios/PartialFinalBatch-8   	    1129	   1057101 ns/op	   35920 B/op	      14 allocs/op
Benchmark_SequentialFit_Scenarios/PartialFinalBatch-8   	    1138	   1054044 ns/op	   35920 B/op	      14 allocs/op
Benchmark_SequentialFit_Scenarios/PartialFinalBatch-8   	    1138	   1053892 ns/op	   35920 B/op	      14 allocs/op
Benchmark_SequentialFit_Scenarios/PartialFinalBatch-8   	    1138	   1053577 ns/op	   35920 B/op	      14 allocs/op
Benchmark_SequentialFit_Scenarios/PartialFinalBatch-8   	    1135	   1053779 ns/op	   35920 B/op	      14 allocs/op
Benchmark_SequentialFit_Scenarios/Shuffle-8             	    1150	   1037400 ns/op	   31984 B/op	      10 allocs/op
Benchmark_SequentialFit_Scenarios/Shuffle-8             	    1158	   1037504 ns/op	   31984 B/op	      10 allocs/op
Benchmark_SequentialFit_Scenarios/Shuffle-8             	    1156	   1036079 ns/op	   31984 B/op	      10 allocs/op
Benchmark_SequentialFit_Scenarios/Shuffle-8             	    1146	   1039327 ns/op	   31984 B/op	      10 allocs/op
Benchmark_SequentialFit_Scenarios/Shuffle-8             	    1153	   1037004 ns/op	   31984 B/op	      10 allocs/op
Benchmark_SequentialFit_Scenarios/Validation-8          	    1027	   1168235 ns/op	   46416 B/op	      14 allocs/op
Benchmark_SequentialFit_Scenarios/Validation-8          	    1028	   1168377 ns/op	   46416 B/op	      14 allocs/op
Benchmark_SequentialFit_Scenarios/Validation-8          	    1026	   1167148 ns/op	   46416 B/op	      14 allocs/op
Benchmark_SequentialFit_Scenarios/Validation-8          	    1026	   1168629 ns/op	   46416 B/op	      14 allocs/op
Benchmark_SequentialFit_Scenarios/Validation-8          	    1026	   1168178 ns/op	   46416 B/op	      14 allocs/op
Benchmark_SequentialSave_ColdPath-8                     	   10000	    115335 ns/op	  166966 B/op	      38 allocs/op
Benchmark_SequentialSave_ColdPath-8                     	   10000	    115577 ns/op	  166953 B/op	      38 allocs/op
Benchmark_SequentialSave_ColdPath-8                     	   10000	    115470 ns/op	  166970 B/op	      38 allocs/op
Benchmark_SequentialSave_ColdPath-8                     	   10000	    115810 ns/op	  166966 B/op	      38 allocs/op
Benchmark_SequentialSave_ColdPath-8                     	   10000	    116006 ns/op	  166973 B/op	      38 allocs/op
Benchmark_LoadSequential_ColdPath-8                     	    4129	    285494 ns/op	  195488 B/op	     147 allocs/op
Benchmark_LoadSequential_ColdPath-8                     	    4231	    284144 ns/op	  195488 B/op	     147 allocs/op
Benchmark_LoadSequential_ColdPath-8                     	    4218	    283884 ns/op	  195488 B/op	     147 allocs/op
Benchmark_LoadSequential_ColdPath-8                     	    4254	    290039 ns/op	  195488 B/op	     147 allocs/op
Benchmark_LoadSequential_ColdPath-8                     	    4057	    285379 ns/op	  195488 B/op	     147 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/model	168.628s
```

### optimizer

```text
goos: darwin
goarch: arm64
pkg: github.com/itsmontoya/neuralnetwork/optimizer
cpu: Apple M3
Benchmark_SGDUpdate_SteadyState-8           	  535738	      2228 ns/op	       0 B/op	       0 allocs/op
Benchmark_SGDUpdate_SteadyState-8           	  535405	      2236 ns/op	       0 B/op	       0 allocs/op
Benchmark_SGDUpdate_SteadyState-8           	  533132	      2233 ns/op	       0 B/op	       0 allocs/op
Benchmark_SGDUpdate_SteadyState-8           	  541345	      2242 ns/op	       0 B/op	       0 allocs/op
Benchmark_SGDUpdate_SteadyState-8           	  540873	      2230 ns/op	       0 B/op	       0 allocs/op
Benchmark_MomentumUpdate_SteadyState-8      	  429996	      2787 ns/op	       0 B/op	       0 allocs/op
Benchmark_MomentumUpdate_SteadyState-8      	  429243	      2779 ns/op	       0 B/op	       0 allocs/op
Benchmark_MomentumUpdate_SteadyState-8      	  430814	      2778 ns/op	       0 B/op	       0 allocs/op
Benchmark_MomentumUpdate_SteadyState-8      	  428791	      2790 ns/op	       0 B/op	       0 allocs/op
Benchmark_MomentumUpdate_SteadyState-8      	  429285	      2782 ns/op	       0 B/op	       0 allocs/op
Benchmark_AdamUpdate_SteadyState-8          	  178426	      6772 ns/op	       0 B/op	       0 allocs/op
Benchmark_AdamUpdate_SteadyState-8          	  178617	      6761 ns/op	       0 B/op	       0 allocs/op
Benchmark_AdamUpdate_SteadyState-8          	  178358	      6756 ns/op	       0 B/op	       0 allocs/op
Benchmark_AdamUpdate_SteadyState-8          	  178048	      6791 ns/op	       0 B/op	       0 allocs/op
Benchmark_AdamUpdate_SteadyState-8          	  178190	      6757 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/SGD/L1-8         	  162561	      7341 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/SGD/L1-8         	  163506	      7320 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/SGD/L1-8         	  163179	      7312 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/SGD/L1-8         	  163542	      7320 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/SGD/L1-8         	  163519	      7316 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/SGD/L2-8         	  199701	      5926 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/SGD/L2-8         	  201027	      5943 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/SGD/L2-8         	  200745	      5928 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/SGD/L2-8         	  199916	      5951 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/SGD/L2-8         	  200043	      5939 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/Adam/L1-8        	   96531	     11901 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/Adam/L1-8        	  101397	     11876 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/Adam/L1-8        	  101314	     11907 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/Adam/L1-8        	  101299	     11880 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/Adam/L1-8        	  101334	     11861 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/Adam/L2-8        	  114168	     10470 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/Adam/L2-8        	  114440	     10498 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/Adam/L2-8        	  114421	     10481 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/Adam/L2-8        	  114386	     10493 ns/op	       0 B/op	       0 allocs/op
Benchmark_RegularizedUpdate_SteadyState/Adam/L2-8        	  114609	     10487 ns/op	       0 B/op	       0 allocs/op
PASS
ok  	github.com/itsmontoya/neuralnetwork/optimizer	45.274s
```

