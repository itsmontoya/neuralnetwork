package data_test

import (
	"math/rand"
	"strconv"
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

const (
	benchmarkDataViewsOrdinarySamples        = 4096
	benchmarkDataViewsOrdinaryPartialSamples = 4100
	benchmarkDataViewsOrdinaryInputs         = 256
	benchmarkDataViewsOrdinaryTargets        = 16
	benchmarkDataViewsOrdinaryBatchSize      = 256
	benchmarkDataViewsSequenceSamples        = 512
	benchmarkDataViewsSequenceSteps          = 128
	benchmarkDataViewsSequenceFeatures       = 32
	benchmarkDataViewsSequenceInputs         = 4096
	benchmarkDataViewsSequenceTargets        = 8
	benchmarkDataViewsSequenceBatchSize      = 256
	benchmarkDataViewsSequencePartialBatch   = 192
)

var (
	benchmarkDataViewsDataset         *data.Dataset
	benchmarkDataViewsDatasetOther    *data.Dataset
	benchmarkDataViewsSequenceDataset *data.SequenceDataset
	benchmarkDataViewsSequenceOther   *data.SequenceDataset
	benchmarkDataViewsBatches         []*data.Batch
	benchmarkDataViewsSequenceBatches []*data.SequenceBatch
	benchmarkDataViewsMatrix          *matrix.Matrix
	benchmarkDataViewsMatrixOther     *matrix.Matrix
	benchmarkDataViewsLengths         *data.SequenceLengths
	benchmarkDataViewsLengthValues    []int
	benchmarkDataViewsChecksum        float32
	benchmarkDataViewsCopied          bool
)

func Benchmark_DataViewsCopyBaseline(b *testing.B) {
	b.Run("Constructor/Ordinary4096x256_4096x16", benchmarkDataViewsOrdinaryConstructor)
	b.Run("WholeDatasetCopiedAccess/Ordinary4096x256_4096x16", benchmarkDataViewsWholeDatasetCopiedAccess)
	b.Run("WholeBatchCopiedAccess/Ordinary256x256_256x16", benchmarkDataViewsWholeBatchCopiedAccess)
	b.Run("OrderedBatches/Ordinary4096_Batch256", benchmarkDataViewsOrderedBatches)
	b.Run("OrderedPartialBatches/Ordinary4100_Batch256", benchmarkDataViewsOrderedPartialBatches)
	b.Run("ShuffledBatches/Ordinary4096_Batch256", benchmarkDataViewsShuffledBatches)
	b.Run("OrderedSplit/Ordinary4096_75_25", benchmarkDataViewsOrderedSplit)
	b.Run("ShuffledSplit/Ordinary4096_75_25", benchmarkDataViewsShuffledSplit)
	b.Run("SelectedRows/Contiguous256", benchmarkDataViewsContiguousSelectedRows)
	b.Run("SelectedRows/ArbitraryRepeated256", benchmarkDataViewsArbitrarySelectedRows)
}

func Benchmark_DataViewsView(b *testing.B) {
	b.Run("WholeDatasetView/Ordinary4096x256_4096x16", benchmarkDataViewsWholeDatasetView)
	b.Run("WholeBatchView/Ordinary256x256_256x16", benchmarkDataViewsWholeBatchView)
	b.Run("RowView/Contiguous256", benchmarkDataViewsContiguousRowView)
	b.Run("RowView/PartialFinal4", benchmarkDataViewsPartialFinalRowView)
}

func Benchmark_SequenceDataViewsCopyBaseline(b *testing.B) {
	b.Run("Constructor/Sequence512x4096_512x8_Lengths512", benchmarkSequenceDataViewsConstructor)
	b.Run("WholeDatasetCopiedAccess/Sequence512x4096_512x8_Lengths512", benchmarkSequenceDataViewsWholeDatasetCopiedAccess)
	b.Run("WholeBatchCopiedAccess/Sequence256x4096_256x8_Lengths256", benchmarkSequenceDataViewsWholeBatchCopiedAccess)
	b.Run("BatchLengthCopy/Lengths256", benchmarkSequenceDataViewsBatchLengthCopy)
	b.Run("OrderedBatches/Sequence512_Batch256", benchmarkSequenceDataViewsOrderedBatches)
	b.Run("OrderedPartialBatches/Sequence512_Batch192", benchmarkSequenceDataViewsOrderedPartialBatches)
	b.Run("ShuffledBatches/Sequence512_Batch256", benchmarkSequenceDataViewsShuffledBatches)
	b.Run("OrderedSplit/Sequence512_75_25", benchmarkSequenceDataViewsOrderedSplit)
	b.Run("ShuffledSplit/Sequence512_75_25", benchmarkSequenceDataViewsShuffledSplit)
	b.Run("SelectedRows/Contiguous256", benchmarkSequenceDataViewsContiguousSelectedRows)
	b.Run("SelectedRows/ArbitraryRepeated256", benchmarkSequenceDataViewsArbitrarySelectedRows)
}

func Benchmark_SequenceDataViewsView(b *testing.B) {
	b.Run("WholeDatasetView/Sequence512x4096_512x8_Lengths512", benchmarkSequenceDataViewsWholeDatasetView)
	b.Run("WholeBatchView/Sequence256x4096_256x8_Lengths256", benchmarkSequenceDataViewsWholeBatchView)
	b.Run("RowView/Contiguous256", benchmarkSequenceDataViewsContiguousRowView)
	b.Run("RowView/PartialFinal128", benchmarkSequenceDataViewsPartialFinalRowView)
}

func benchmarkDataViewsOrdinaryConstructor(b *testing.B) {
	var (
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		dataset      *data.Dataset
		logicalBytes int64
		err          error
		index        int
	)

	inputs, targets = benchmarkDataViewsOrdinaryMatrices(
		b,
		benchmarkDataViewsOrdinarySamples,
	)
	logicalBytes = benchmarkDataViewsOrdinaryBytes(
		benchmarkDataViewsOrdinarySamples,
	)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		dataset, err = data.NewDataset(inputs, targets)
		if err != nil {
			b.Fatalf("NewDataset returned error: %v", err)
		}
	}

	b.StopTimer()
	verifyDataViewsOrdinaryDataset(
		b,
		dataset,
		benchmarkDataViewsOrdinarySamples,
	)
	benchmarkDataViewsDataset = dataset
}

func benchmarkDataViewsWholeDatasetCopiedAccess(b *testing.B) {
	var (
		dataset      *data.Dataset
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		logicalBytes int64
		err          error
		index        int
	)

	dataset = benchmarkDataViewsOrdinaryDataset(
		b,
		benchmarkDataViewsOrdinarySamples,
	)
	logicalBytes = benchmarkDataViewsOrdinaryBytes(
		benchmarkDataViewsOrdinarySamples,
	)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		if inputs, err = dataset.Inputs(); err != nil {
			b.Fatalf("Inputs returned error: %v", err)
		}
		if targets, err = dataset.Targets(); err != nil {
			b.Fatalf("Targets returned error: %v", err)
		}
	}

	b.StopTimer()
	verifyDataViewsOrdinaryMatrices(
		b,
		inputs,
		targets,
		0,
		benchmarkDataViewsOrdinarySamples,
	)
	benchmarkDataViewsMatrix = inputs
	benchmarkDataViewsMatrixOther = targets
}

func benchmarkDataViewsWholeBatchCopiedAccess(b *testing.B) {
	var (
		dataset      *data.Dataset
		batches      []*data.Batch
		batch        *data.Batch
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		logicalBytes int64
		err          error
		index        int
	)

	dataset = benchmarkDataViewsOrdinaryDataset(
		b,
		benchmarkDataViewsOrdinarySamples,
	)
	if batches, err = dataset.Batches(benchmarkDataViewsOrdinaryBatchSize, nil); err != nil {
		b.Fatalf("Batches returned error: %v", err)
	}
	batch = batches[0]
	verifyDataViewsOrdinaryBatch(b, batch, 0, benchmarkDataViewsOrdinaryBatchSize)
	logicalBytes = benchmarkDataViewsOrdinaryBytes(
		benchmarkDataViewsOrdinaryBatchSize,
	)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		if inputs, err = batch.Inputs(); err != nil {
			b.Fatalf("batch Inputs returned error: %v", err)
		}
		if targets, err = batch.Targets(); err != nil {
			b.Fatalf("batch Targets returned error: %v", err)
		}
	}

	b.StopTimer()
	verifyDataViewsOrdinaryMatrices(
		b,
		inputs,
		targets,
		0,
		benchmarkDataViewsOrdinaryBatchSize,
	)
	benchmarkDataViewsMatrix = inputs
	benchmarkDataViewsMatrixOther = targets
}

func benchmarkDataViewsOrderedBatches(b *testing.B) {
	benchmarkDataViewsBatchesOperation(
		b,
		benchmarkDataViewsOrdinarySamples,
		benchmarkDataViewsOrdinaryBatchSize,
		nil,
	)
}

func benchmarkDataViewsOrderedPartialBatches(b *testing.B) {
	benchmarkDataViewsBatchesOperation(
		b,
		benchmarkDataViewsOrdinaryPartialSamples,
		benchmarkDataViewsOrdinaryBatchSize,
		nil,
	)
}

func benchmarkDataViewsShuffledBatches(b *testing.B) {
	benchmarkDataViewsBatchesOperation(
		b,
		benchmarkDataViewsOrdinarySamples,
		benchmarkDataViewsOrdinaryBatchSize,
		rand.New(rand.NewSource(7)),
	)
}

func benchmarkDataViewsBatchesOperation(
	b *testing.B,
	samples,
	batchSize int,
	random *rand.Rand,
) {
	var (
		dataset      *data.Dataset
		batches      []*data.Batch
		logicalBytes int64
		err          error
		index        int
	)

	dataset = benchmarkDataViewsOrdinaryDataset(b, samples)
	logicalBytes = benchmarkDataViewsOrdinaryBytes(samples)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		batches, err = dataset.Batches(batchSize, random)
		if err != nil {
			b.Fatalf("Batches returned error: %v", err)
		}
	}

	b.StopTimer()
	verifyDataViewsOrdinaryBatches(b, batches)
	benchmarkDataViewsBatches = batches
}

func benchmarkDataViewsOrderedSplit(b *testing.B) {
	benchmarkDataViewsSplitOperation(b, nil)
}

func benchmarkDataViewsShuffledSplit(b *testing.B) {
	benchmarkDataViewsSplitOperation(
		b,
		rand.New(rand.NewSource(11)),
	)
}

func benchmarkDataViewsSplitOperation(b *testing.B, random *rand.Rand) {
	var (
		dataset      *data.Dataset
		train        *data.Dataset
		test         *data.Dataset
		logicalBytes int64
		err          error
		index        int
	)

	dataset = benchmarkDataViewsOrdinaryDataset(
		b,
		benchmarkDataViewsOrdinarySamples,
	)
	logicalBytes = benchmarkDataViewsOrdinaryBytes(
		benchmarkDataViewsOrdinarySamples,
	)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		train, test, err = dataset.Split(0.25, random)
		if err != nil {
			b.Fatalf("Split returned error: %v", err)
		}
	}

	b.StopTimer()
	verifyDataViewsOrdinaryAlignedDataset(b, train)
	verifyDataViewsOrdinaryAlignedDataset(b, test)
	benchmarkDataViewsDataset = train
	benchmarkDataViewsDatasetOther = test
}

func benchmarkDataViewsContiguousSelectedRows(b *testing.B) {
	var (
		indexes []int
		index   int
	)

	indexes = make([]int, benchmarkDataViewsOrdinaryBatchSize)
	for index = range indexes {
		indexes[index] = index + benchmarkDataViewsOrdinaryBatchSize
	}

	benchmarkDataViewsSelectedRows(b, indexes)
}

func benchmarkDataViewsArbitrarySelectedRows(b *testing.B) {
	var (
		indexes []int
		index   int
	)

	indexes = make([]int, benchmarkDataViewsOrdinaryBatchSize)
	for index = range indexes {
		indexes[index] = (index * 17) % 127
	}

	benchmarkDataViewsSelectedRows(b, indexes)
}

func benchmarkDataViewsWholeDatasetView(b *testing.B) {
	benchmarkDataViewsViewOperation(
		b,
		benchmarkDataViewsOrdinarySamples,
		0,
		benchmarkDataViewsOrdinarySamples,
		true,
	)
}

func benchmarkDataViewsWholeBatchView(b *testing.B) {
	var (
		dataset      *data.Dataset
		batches      []*data.Batch
		batch        *data.Batch
		checksum     float32
		logicalBytes int64
		copied       bool
		err          error
		index        int
	)

	dataset = benchmarkDataViewsOrdinaryDataset(b, benchmarkDataViewsOrdinarySamples)
	if batches, err = dataset.Batches(benchmarkDataViewsOrdinaryBatchSize, nil); err != nil {
		b.Fatalf("Batches returned error: %v", err)
	}
	batch = batches[0]
	logicalBytes = benchmarkDataViewsOrdinaryBytes(benchmarkDataViewsOrdinaryBatchSize)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, 0)

	for index = 0; index < b.N; index++ {
		err = batch.WithView(func(view *data.DatasetView) (viewErr error) {
			checksum, copied, viewErr = observeDataViewsView(view)
			return viewErr
		})
		if err != nil {
			b.Fatalf("batch WithView returned error: %v", err)
		}
	}

	benchmarkDataViewsChecksum = checksum
	benchmarkDataViewsCopied = copied
}

func benchmarkDataViewsContiguousRowView(b *testing.B) {
	benchmarkDataViewsViewOperation(
		b,
		benchmarkDataViewsOrdinarySamples,
		benchmarkDataViewsOrdinaryBatchSize,
		2*benchmarkDataViewsOrdinaryBatchSize,
		false,
	)
}

func benchmarkDataViewsPartialFinalRowView(b *testing.B) {
	benchmarkDataViewsViewOperation(
		b,
		benchmarkDataViewsOrdinaryPartialSamples,
		benchmarkDataViewsOrdinarySamples,
		benchmarkDataViewsOrdinaryPartialSamples,
		false,
	)
}

func benchmarkDataViewsViewOperation(b *testing.B, samples, start, end int, whole bool) {
	var (
		dataset      *data.Dataset
		checksum     float32
		logicalBytes int64
		copied       bool
		err          error
		index        int
	)

	dataset = benchmarkDataViewsOrdinaryDataset(b, samples)
	logicalBytes = benchmarkDataViewsOrdinaryBytes(end - start)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, 0)

	for index = 0; index < b.N; index++ {
		if whole {
			err = dataset.WithView(func(view *data.DatasetView) (viewErr error) {
				checksum, copied, viewErr = observeDataViewsView(view)
				return viewErr
			})
		} else {
			err = dataset.WithRowView(start, end, func(view *data.DatasetView) (viewErr error) {
				checksum, copied, viewErr = observeDataViewsView(view)
				return viewErr
			})
		}
		if err != nil {
			b.Fatalf("view operation returned error: %v", err)
		}
	}

	benchmarkDataViewsChecksum = checksum
	benchmarkDataViewsCopied = copied
}

func observeDataViewsView(view *data.DatasetView) (checksum float32, copied bool, err error) {
	var (
		inputs     *matrix.Matrix
		targets    *matrix.Matrix
		firstInput float32
		lastInput  float32
		lastTarget float32
	)

	if inputs, err = view.Inputs(); err != nil {
		return 0, false, err
	}
	if targets, err = view.Targets(); err != nil {
		return 0, false, err
	}
	if firstInput, err = inputs.At(0, 0); err != nil {
		return 0, false, err
	}
	if lastInput, err = inputs.At(inputs.Rows()-1, inputs.Cols()-1); err != nil {
		return 0, false, err
	}
	if lastTarget, err = targets.At(targets.Rows()-1, targets.Cols()-1); err != nil {
		return 0, false, err
	}

	checksum = firstInput + lastInput + lastTarget
	copied = view.Copied()
	return checksum, copied, nil
}

func benchmarkDataViewsSelectedRows(b *testing.B, indexes []int) {
	var (
		dataset      *data.Dataset
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		logicalBytes int64
		err          error
		index        int
	)

	dataset = benchmarkDataViewsOrdinaryDataset(
		b,
		benchmarkDataViewsOrdinarySamples,
	)
	if inputs, err = matrix.New(len(indexes), benchmarkDataViewsOrdinaryInputs); err != nil {
		b.Fatalf("matrix.New inputs returned error: %v", err)
	}
	if targets, err = matrix.New(len(indexes), benchmarkDataViewsOrdinaryTargets); err != nil {
		b.Fatalf("matrix.New targets returned error: %v", err)
	}
	logicalBytes = benchmarkDataViewsOrdinaryBytes(len(indexes))
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		if err = dataset.SelectRowsInto(indexes, inputs, targets); err != nil {
			b.Fatalf("SelectRowsInto returned error: %v", err)
		}
	}

	b.StopTimer()
	verifyDataViewsOrdinarySelection(b, indexes, inputs, targets)
	benchmarkDataViewsMatrix = inputs
	benchmarkDataViewsMatrixOther = targets
	benchmarkDataViewsChecksum = benchmarkDataViewsMatrixChecksum(b, inputs, targets)
}

func benchmarkSequenceDataViewsConstructor(b *testing.B) {
	var (
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		lengths      *data.SequenceLengths
		dataset      *data.SequenceDataset
		logicalBytes int64
		err          error
		index        int
	)

	inputs, targets, lengths = benchmarkDataViewsSequenceValues(
		b,
		benchmarkDataViewsSequenceSamples,
	)
	logicalBytes = benchmarkDataViewsSequenceBytes(
		benchmarkDataViewsSequenceSamples,
	)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		dataset, err = data.NewSequenceDataset(inputs, targets, lengths)
		if err != nil {
			b.Fatalf("NewSequenceDataset returned error: %v", err)
		}
	}

	b.StopTimer()
	verifyDataViewsSequenceDataset(
		b,
		dataset,
		benchmarkDataViewsSequenceSamples,
	)
	benchmarkDataViewsSequenceDataset = dataset
}

func benchmarkSequenceDataViewsWholeDatasetCopiedAccess(b *testing.B) {
	var (
		dataset      *data.SequenceDataset
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		lengths      *data.SequenceLengths
		logicalBytes int64
		err          error
		index        int
	)

	dataset = benchmarkDataViewsSequenceDatasetFixture(
		b,
		benchmarkDataViewsSequenceSamples,
	)
	logicalBytes = benchmarkDataViewsSequenceBytes(
		benchmarkDataViewsSequenceSamples,
	)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		if inputs, err = dataset.Inputs(); err != nil {
			b.Fatalf("Inputs returned error: %v", err)
		}
		if targets, err = dataset.Targets(); err != nil {
			b.Fatalf("Targets returned error: %v", err)
		}
		if lengths, err = dataset.Lengths(); err != nil {
			b.Fatalf("Lengths returned error: %v", err)
		}
	}

	b.StopTimer()
	verifyDataViewsSequenceValues(
		b,
		inputs,
		targets,
		lengths,
		0,
		benchmarkDataViewsSequenceSamples,
	)
	benchmarkDataViewsMatrix = inputs
	benchmarkDataViewsMatrixOther = targets
	benchmarkDataViewsLengths = lengths
}

func benchmarkSequenceDataViewsWholeBatchCopiedAccess(b *testing.B) {
	var (
		batch        *data.SequenceBatch
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		lengths      *data.SequenceLengths
		logicalBytes int64
		err          error
		index        int
	)

	batch = benchmarkDataViewsSequenceBatch(b)
	logicalBytes = benchmarkDataViewsSequenceBytes(
		benchmarkDataViewsSequenceBatchSize,
	)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		if inputs, err = batch.Inputs(); err != nil {
			b.Fatalf("batch Inputs returned error: %v", err)
		}
		if targets, err = batch.Targets(); err != nil {
			b.Fatalf("batch Targets returned error: %v", err)
		}
		if lengths, err = batch.Lengths(); err != nil {
			b.Fatalf("batch Lengths returned error: %v", err)
		}
	}

	b.StopTimer()
	verifyDataViewsSequenceValues(
		b,
		inputs,
		targets,
		lengths,
		0,
		benchmarkDataViewsSequenceBatchSize,
	)
	benchmarkDataViewsMatrix = inputs
	benchmarkDataViewsMatrixOther = targets
	benchmarkDataViewsLengths = lengths
}

func benchmarkSequenceDataViewsBatchLengthCopy(b *testing.B) {
	var (
		batch        *data.SequenceBatch
		lengths      *data.SequenceLengths
		logicalBytes int64
		err          error
		index        int
	)

	batch = benchmarkDataViewsSequenceBatch(b)
	logicalBytes = benchmarkDataViewsLengthBytes(
		benchmarkDataViewsSequenceBatchSize,
	)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		lengths, err = batch.Lengths()
		if err != nil {
			b.Fatalf("batch Lengths returned error: %v", err)
		}
	}

	b.StopTimer()
	if err = lengths.Validate(); err != nil {
		b.Fatalf("copied batch lengths are invalid: %v", err)
	}
	benchmarkDataViewsLengths = lengths
}

func benchmarkSequenceDataViewsOrderedBatches(b *testing.B) {
	benchmarkSequenceDataViewsBatchesOperation(
		b,
		benchmarkDataViewsSequenceBatchSize,
		nil,
	)
}

func benchmarkSequenceDataViewsOrderedPartialBatches(b *testing.B) {
	benchmarkSequenceDataViewsBatchesOperation(
		b,
		benchmarkDataViewsSequencePartialBatch,
		nil,
	)
}

func benchmarkSequenceDataViewsShuffledBatches(b *testing.B) {
	benchmarkSequenceDataViewsBatchesOperation(
		b,
		benchmarkDataViewsSequenceBatchSize,
		rand.New(rand.NewSource(13)),
	)
}

func benchmarkSequenceDataViewsBatchesOperation(
	b *testing.B,
	batchSize int,
	random *rand.Rand,
) {
	var (
		dataset      *data.SequenceDataset
		batches      []*data.SequenceBatch
		logicalBytes int64
		err          error
		index        int
	)

	dataset = benchmarkDataViewsSequenceDatasetFixture(
		b,
		benchmarkDataViewsSequenceSamples,
	)
	logicalBytes = benchmarkDataViewsSequenceBytes(
		benchmarkDataViewsSequenceSamples,
	)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		batches, err = dataset.Batches(batchSize, random)
		if err != nil {
			b.Fatalf("Batches returned error: %v", err)
		}
	}

	b.StopTimer()
	verifyDataViewsSequenceBatches(b, batches)
	benchmarkDataViewsSequenceBatches = batches
}

func benchmarkSequenceDataViewsOrderedSplit(b *testing.B) {
	benchmarkSequenceDataViewsSplitOperation(b, nil)
}

func benchmarkSequenceDataViewsShuffledSplit(b *testing.B) {
	benchmarkSequenceDataViewsSplitOperation(
		b,
		rand.New(rand.NewSource(17)),
	)
}

func benchmarkSequenceDataViewsSplitOperation(
	b *testing.B,
	random *rand.Rand,
) {
	var (
		dataset      *data.SequenceDataset
		train        *data.SequenceDataset
		test         *data.SequenceDataset
		logicalBytes int64
		err          error
		index        int
	)

	dataset = benchmarkDataViewsSequenceDatasetFixture(
		b,
		benchmarkDataViewsSequenceSamples,
	)
	logicalBytes = benchmarkDataViewsSequenceBytes(
		benchmarkDataViewsSequenceSamples,
	)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		train, test, err = dataset.Split(0.25, random)
		if err != nil {
			b.Fatalf("Split returned error: %v", err)
		}
	}

	b.StopTimer()
	verifyDataViewsSequenceAlignedDataset(b, train)
	verifyDataViewsSequenceAlignedDataset(b, test)
	benchmarkDataViewsSequenceDataset = train
	benchmarkDataViewsSequenceOther = test
}

func benchmarkSequenceDataViewsContiguousSelectedRows(b *testing.B) {
	var (
		indexes []int
		index   int
	)

	indexes = make([]int, benchmarkDataViewsSequenceBatchSize)
	for index = range indexes {
		indexes[index] = index + benchmarkDataViewsSequenceBatchSize
	}

	benchmarkSequenceDataViewsSelectedRows(b, indexes)
}

func benchmarkSequenceDataViewsArbitrarySelectedRows(b *testing.B) {
	var (
		indexes []int
		index   int
	)

	indexes = make([]int, benchmarkDataViewsSequenceBatchSize)
	for index = range indexes {
		indexes[index] = (index * 19) % 127
	}

	benchmarkSequenceDataViewsSelectedRows(b, indexes)
}

func benchmarkSequenceDataViewsWholeDatasetView(b *testing.B) {
	benchmarkSequenceDataViewsViewOperation(
		b,
		0,
		benchmarkDataViewsSequenceSamples,
		true,
	)
}

func benchmarkSequenceDataViewsWholeBatchView(b *testing.B) {
	var (
		batch        *data.SequenceBatch
		checksum     float32
		logicalBytes int64
		copied       bool
		err          error
		index        int
	)

	batch = benchmarkDataViewsSequenceBatch(b)
	logicalBytes = benchmarkDataViewsSequenceBytes(
		benchmarkDataViewsSequenceBatchSize,
	)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, 0)

	for index = 0; index < b.N; index++ {
		err = batch.WithView(func(view *data.SequenceDatasetView) (viewErr error) {
			checksum, copied, viewErr = observeSequenceDataViewsView(view)
			return viewErr
		})
		if err != nil {
			b.Fatalf("batch WithView returned error: %v", err)
		}
	}

	benchmarkDataViewsChecksum = checksum
	benchmarkDataViewsCopied = copied
}

func benchmarkSequenceDataViewsContiguousRowView(b *testing.B) {
	benchmarkSequenceDataViewsViewOperation(
		b,
		benchmarkDataViewsSequenceBatchSize,
		2*benchmarkDataViewsSequenceBatchSize,
		false,
	)
}

func benchmarkSequenceDataViewsPartialFinalRowView(b *testing.B) {
	benchmarkSequenceDataViewsViewOperation(
		b,
		2*benchmarkDataViewsSequencePartialBatch,
		benchmarkDataViewsSequenceSamples,
		false,
	)
}

func benchmarkSequenceDataViewsViewOperation(
	b *testing.B,
	start,
	end int,
	whole bool,
) {
	var (
		dataset      *data.SequenceDataset
		checksum     float32
		logicalBytes int64
		copied       bool
		err          error
		index        int
	)

	dataset = benchmarkDataViewsSequenceDatasetFixture(
		b,
		benchmarkDataViewsSequenceSamples,
	)
	logicalBytes = benchmarkDataViewsSequenceBytes(end - start)
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, 0)

	for index = 0; index < b.N; index++ {
		if whole {
			err = dataset.WithView(func(view *data.SequenceDatasetView) (viewErr error) {
				checksum, copied, viewErr = observeSequenceDataViewsView(view)
				return viewErr
			})
		} else {
			err = dataset.WithRowView(
				start,
				end,
				func(view *data.SequenceDatasetView) (viewErr error) {
					checksum, copied, viewErr = observeSequenceDataViewsView(view)
					return viewErr
				},
			)
		}
		if err != nil {
			b.Fatalf("sequence view operation returned error: %v", err)
		}
	}

	benchmarkDataViewsChecksum = checksum
	benchmarkDataViewsCopied = copied
}

func observeSequenceDataViewsView(
	view *data.SequenceDatasetView,
) (checksum float32, copied bool, err error) {
	var (
		inputs     *matrix.Matrix
		targets    *matrix.Matrix
		lengths    []int
		firstInput float32
		lastInput  float32
		lastTarget float32
	)

	if inputs, err = view.Inputs(); err != nil {
		return 0, false, err
	}
	if targets, err = view.Targets(); err != nil {
		return 0, false, err
	}
	if lengths, err = view.Lengths(); err != nil {
		return 0, false, err
	}
	if firstInput, err = inputs.At(0, 0); err != nil {
		return 0, false, err
	}
	if lastInput, err = inputs.At(inputs.Rows()-1, inputs.Cols()-1); err != nil {
		return 0, false, err
	}
	if lastTarget, err = targets.At(targets.Rows()-1, targets.Cols()-1); err != nil {
		return 0, false, err
	}

	checksum = firstInput + lastInput + lastTarget +
		float32(lengths[0]+lengths[len(lengths)-1])
	copied = view.Copied()
	return checksum, copied, nil
}

func benchmarkSequenceDataViewsSelectedRows(
	b *testing.B,
	indexes []int,
) {
	var (
		dataset      *data.SequenceDataset
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		lengths      []int
		logicalBytes int64
		err          error
		index        int
	)

	dataset = benchmarkDataViewsSequenceDatasetFixture(
		b,
		benchmarkDataViewsSequenceSamples,
	)
	if inputs, err = matrix.New(len(indexes), benchmarkDataViewsSequenceInputs); err != nil {
		b.Fatalf("matrix.New inputs returned error: %v", err)
	}
	if targets, err = matrix.New(len(indexes), benchmarkDataViewsSequenceTargets); err != nil {
		b.Fatalf("matrix.New targets returned error: %v", err)
	}
	lengths = make([]int, len(indexes))
	logicalBytes = benchmarkDataViewsSequenceBytes(len(indexes))
	b.ResetTimer()
	reportDataViewsMetrics(b, logicalBytes, logicalBytes)

	for index = 0; index < b.N; index++ {
		if err = dataset.SelectRowsInto(
			indexes,
			inputs,
			targets,
			lengths,
		); err != nil {
			b.Fatalf("SelectRowsInto returned error: %v", err)
		}
	}

	b.StopTimer()
	verifyDataViewsSequenceSelection(
		b,
		indexes,
		inputs,
		targets,
		lengths,
	)
	benchmarkDataViewsMatrix = inputs
	benchmarkDataViewsMatrixOther = targets
	benchmarkDataViewsLengthValues = lengths
	benchmarkDataViewsChecksum = benchmarkSequenceDataViewsSelectionChecksum(
		b,
		inputs,
		targets,
		lengths,
	)
}

func benchmarkDataViewsOrdinaryDataset(
	tb testing.TB,
	samples int,
) (dataset *data.Dataset) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	tb.Helper()

	inputs, targets = benchmarkDataViewsOrdinaryMatrices(tb, samples)
	if dataset, err = data.NewDataset(inputs, targets); err != nil {
		tb.Fatalf("NewDataset returned error: %v", err)
	}
	verifyDataViewsOrdinaryDataset(tb, dataset, samples)
	return dataset
}

func benchmarkDataViewsOrdinaryMatrices(
	tb testing.TB,
	samples int,
) (inputs, targets *matrix.Matrix) {
	var (
		inputValues  []float32
		targetValues []float32
		row          int
		column       int
		err          error
	)

	tb.Helper()

	inputValues = make([]float32, samples*benchmarkDataViewsOrdinaryInputs)
	targetValues = make([]float32, samples*benchmarkDataViewsOrdinaryTargets)
	for row = 0; row < samples; row++ {
		for column = 0; column < benchmarkDataViewsOrdinaryInputs; column++ {
			inputValues[row*benchmarkDataViewsOrdinaryInputs+column] =
				benchmarkDataViewsOrdinaryInputValue(row, column)
		}
		for column = 0; column < benchmarkDataViewsOrdinaryTargets; column++ {
			targetValues[row*benchmarkDataViewsOrdinaryTargets+column] =
				benchmarkDataViewsOrdinaryTargetValue(row, column)
		}
	}

	if inputs, err = matrix.FromSlice(
		samples,
		benchmarkDataViewsOrdinaryInputs,
		inputValues,
	); err != nil {
		tb.Fatalf("FromSlice inputs returned error: %v", err)
	}
	if targets, err = matrix.FromSlice(
		samples,
		benchmarkDataViewsOrdinaryTargets,
		targetValues,
	); err != nil {
		tb.Fatalf("FromSlice targets returned error: %v", err)
	}
	return inputs, targets
}

func benchmarkDataViewsSequenceDatasetFixture(
	tb testing.TB,
	samples int,
) (dataset *data.SequenceDataset) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		lengths *data.SequenceLengths
		err     error
	)

	tb.Helper()

	inputs, targets, lengths = benchmarkDataViewsSequenceValues(tb, samples)
	if dataset, err = data.NewSequenceDataset(inputs, targets, lengths); err != nil {
		tb.Fatalf("NewSequenceDataset returned error: %v", err)
	}
	verifyDataViewsSequenceDataset(tb, dataset, samples)
	return dataset
}

func benchmarkDataViewsSequenceValues(
	tb testing.TB,
	samples int,
) (
	inputs,
	targets *matrix.Matrix,
	lengths *data.SequenceLengths,
) {
	var (
		inputValues  []float32
		targetValues []float32
		lengthValues []int
		row          int
		column       int
		err          error
	)

	tb.Helper()

	inputValues = make([]float32, samples*benchmarkDataViewsSequenceInputs)
	targetValues = make([]float32, samples*benchmarkDataViewsSequenceTargets)
	lengthValues = make([]int, samples)
	for row = 0; row < samples; row++ {
		for column = 0; column < benchmarkDataViewsSequenceInputs; column++ {
			inputValues[row*benchmarkDataViewsSequenceInputs+column] =
				benchmarkDataViewsSequenceInputValue(row, column)
		}
		for column = 0; column < benchmarkDataViewsSequenceTargets; column++ {
			targetValues[row*benchmarkDataViewsSequenceTargets+column] =
				benchmarkDataViewsSequenceTargetValue(row, column)
		}
		lengthValues[row] = row%benchmarkDataViewsSequenceSteps + 1
	}

	if inputs, err = matrix.FromSlice(
		samples,
		benchmarkDataViewsSequenceInputs,
		inputValues,
	); err != nil {
		tb.Fatalf("FromSlice sequence inputs returned error: %v", err)
	}
	if targets, err = matrix.FromSlice(
		samples,
		benchmarkDataViewsSequenceTargets,
		targetValues,
	); err != nil {
		tb.Fatalf("FromSlice sequence targets returned error: %v", err)
	}
	if lengths, err = data.NewSequenceLengths(
		benchmarkDataViewsSequenceSteps,
		lengthValues,
	); err != nil {
		tb.Fatalf("NewSequenceLengths returned error: %v", err)
	}
	return inputs, targets, lengths
}

func benchmarkDataViewsSequenceBatch(
	tb testing.TB,
) (batch *data.SequenceBatch) {
	var (
		dataset *data.SequenceDataset
		batches []*data.SequenceBatch
		err     error
	)

	tb.Helper()

	dataset = benchmarkDataViewsSequenceDatasetFixture(
		tb,
		benchmarkDataViewsSequenceSamples,
	)
	if batches, err = dataset.Batches(benchmarkDataViewsSequenceBatchSize, nil); err != nil {
		tb.Fatalf("Batches returned error: %v", err)
	}
	batch = batches[0]
	verifyDataViewsSequenceBatch(
		tb,
		batch,
		0,
		benchmarkDataViewsSequenceBatchSize,
	)
	return batch
}

func verifyDataViewsOrdinaryDataset(
	tb testing.TB,
	dataset *data.Dataset,
	samples int,
) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	tb.Helper()

	if inputs, err = dataset.Inputs(); err != nil {
		tb.Fatalf("verify Inputs returned error: %v", err)
	}
	if targets, err = dataset.Targets(); err != nil {
		tb.Fatalf("verify Targets returned error: %v", err)
	}
	verifyDataViewsOrdinaryMatrices(tb, inputs, targets, 0, samples)
}

func verifyDataViewsOrdinaryBatch(
	tb testing.TB,
	batch *data.Batch,
	firstRow,
	samples int,
) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	tb.Helper()

	if inputs, err = batch.Inputs(); err != nil {
		tb.Fatalf("verify batch Inputs returned error: %v", err)
	}
	if targets, err = batch.Targets(); err != nil {
		tb.Fatalf("verify batch Targets returned error: %v", err)
	}
	verifyDataViewsOrdinaryMatrices(tb, inputs, targets, firstRow, samples)
}

func verifyDataViewsOrdinaryBatches(
	tb testing.TB,
	batches []*data.Batch,
) {
	var (
		batch   *data.Batch
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	tb.Helper()

	if len(batches) == 0 {
		tb.Fatal("batch result is empty")
	}
	for _, batch = range batches {
		if inputs, err = batch.Inputs(); err != nil {
			tb.Fatalf("verify batch Inputs returned error: %v", err)
		}
		if targets, err = batch.Targets(); err != nil {
			tb.Fatalf("verify batch Targets returned error: %v", err)
		}
		verifyDataViewsOrdinaryAlignedMatrices(tb, inputs, targets)
	}
}

func verifyDataViewsOrdinaryAlignedDataset(
	tb testing.TB,
	dataset *data.Dataset,
) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		err     error
	)

	tb.Helper()

	if inputs, err = dataset.Inputs(); err != nil {
		tb.Fatalf("verify aligned dataset Inputs returned error: %v", err)
	}
	if targets, err = dataset.Targets(); err != nil {
		tb.Fatalf("verify aligned dataset Targets returned error: %v", err)
	}
	verifyDataViewsOrdinaryAlignedMatrices(tb, inputs, targets)
}

func verifyDataViewsOrdinaryAlignedMatrices(
	tb testing.TB,
	inputs,
	targets *matrix.Matrix,
) {
	var (
		row         int
		sourceRow   int
		inputValue  float32
		targetValue float32
		err         error
	)

	tb.Helper()

	if inputs.Rows() != targets.Rows() {
		tb.Fatalf(
			"aligned matrix rows differ: inputs=%d targets=%d",
			inputs.Rows(),
			targets.Rows(),
		)
	}
	for row = 0; row < inputs.Rows(); row++ {
		if inputValue, err = inputs.At(row, 0); err != nil {
			tb.Fatalf("aligned input At returned error: %v", err)
		}
		sourceRow = int(inputValue) / 512
		if inputValue != benchmarkDataViewsOrdinaryInputValue(sourceRow, 0) {
			tb.Fatalf("input row %d does not encode a source row: value=%g", row, inputValue)
		}
		if targetValue, err = targets.At(row, 0); err != nil {
			tb.Fatalf("aligned target At returned error: %v", err)
		}
		if targetValue != benchmarkDataViewsOrdinaryTargetValue(sourceRow, 0) {
			tb.Fatalf(
				"target row %d is misaligned: got=%g sourceRow=%d",
				row,
				targetValue,
				sourceRow,
			)
		}
	}
}

func verifyDataViewsOrdinarySelection(
	tb testing.TB,
	indexes []int,
	inputs,
	targets *matrix.Matrix,
) {
	var (
		position    int
		sourceRow   int
		inputValue  float32
		targetValue float32
		err         error
	)

	tb.Helper()

	for position, sourceRow = range indexes {
		if inputValue, err = inputs.At(position, 0); err != nil {
			tb.Fatalf("selected input At returned error: %v", err)
		}
		if inputValue != benchmarkDataViewsOrdinaryInputValue(sourceRow, 0) {
			tb.Fatalf(
				"selected input row %d = %g, want source row %d",
				position,
				inputValue,
				sourceRow,
			)
		}
		if targetValue, err = targets.At(position, 0); err != nil {
			tb.Fatalf("selected target At returned error: %v", err)
		}
		if targetValue != benchmarkDataViewsOrdinaryTargetValue(sourceRow, 0) {
			tb.Fatalf(
				"selected target row %d = %g, want source row %d",
				position,
				targetValue,
				sourceRow,
			)
		}
	}
}

func verifyDataViewsOrdinaryMatrices(
	tb testing.TB,
	inputs,
	targets *matrix.Matrix,
	firstRow,
	samples int,
) {
	var (
		lastRow int
		got     float32
		err     error
	)

	tb.Helper()

	lastRow = samples - 1
	if got, err = inputs.At(0, 0); err != nil {
		tb.Fatalf("verify input first value returned error: %v", err)
	} else if got != benchmarkDataViewsOrdinaryInputValue(firstRow, 0) {
		tb.Fatalf("input first value = %g, want aligned row %d", got, firstRow)
	}
	if got, err = inputs.At(lastRow, benchmarkDataViewsOrdinaryInputs-1); err != nil {
		tb.Fatalf("verify input last value returned error: %v", err)
	} else if got != benchmarkDataViewsOrdinaryInputValue(
		firstRow+lastRow,
		benchmarkDataViewsOrdinaryInputs-1,
	) {
		tb.Fatalf("input last value = %g, want aligned row %d", got, firstRow+lastRow)
	}
	if got, err = targets.At(lastRow, benchmarkDataViewsOrdinaryTargets-1); err != nil {
		tb.Fatalf("verify target last value returned error: %v", err)
	} else if got != benchmarkDataViewsOrdinaryTargetValue(
		firstRow+lastRow,
		benchmarkDataViewsOrdinaryTargets-1,
	) {
		tb.Fatalf("target last value = %g, want aligned row %d", got, firstRow+lastRow)
	}
}

func verifyDataViewsSequenceDataset(
	tb testing.TB,
	dataset *data.SequenceDataset,
	samples int,
) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		lengths *data.SequenceLengths
		err     error
	)

	tb.Helper()

	if inputs, err = dataset.Inputs(); err != nil {
		tb.Fatalf("verify sequence Inputs returned error: %v", err)
	}
	if targets, err = dataset.Targets(); err != nil {
		tb.Fatalf("verify sequence Targets returned error: %v", err)
	}
	if lengths, err = dataset.Lengths(); err != nil {
		tb.Fatalf("verify sequence Lengths returned error: %v", err)
	}
	verifyDataViewsSequenceValues(tb, inputs, targets, lengths, 0, samples)
}

func verifyDataViewsSequenceBatch(
	tb testing.TB,
	batch *data.SequenceBatch,
	firstRow,
	samples int,
) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		lengths *data.SequenceLengths
		err     error
	)

	tb.Helper()

	if inputs, err = batch.Inputs(); err != nil {
		tb.Fatalf("verify sequence batch Inputs returned error: %v", err)
	}
	if targets, err = batch.Targets(); err != nil {
		tb.Fatalf("verify sequence batch Targets returned error: %v", err)
	}
	if lengths, err = batch.Lengths(); err != nil {
		tb.Fatalf("verify sequence batch Lengths returned error: %v", err)
	}
	verifyDataViewsSequenceValues(
		tb,
		inputs,
		targets,
		lengths,
		firstRow,
		samples,
	)
}

func verifyDataViewsSequenceValues(
	tb testing.TB,
	inputs,
	targets *matrix.Matrix,
	lengths *data.SequenceLengths,
	firstRow,
	samples int,
) {
	var (
		lengthValues []int
		lastRow      int
		got          float32
		err          error
	)

	tb.Helper()

	lastRow = samples - 1
	if got, err = inputs.At(0, 0); err != nil {
		tb.Fatalf("verify sequence input first value returned error: %v", err)
	} else if got != benchmarkDataViewsSequenceInputValue(firstRow, 0) {
		tb.Fatalf("sequence input first value = %g, want aligned row %d", got, firstRow)
	}
	if got, err = inputs.At(lastRow, benchmarkDataViewsSequenceInputs-1); err != nil {
		tb.Fatalf("verify sequence input last value returned error: %v", err)
	} else if got != benchmarkDataViewsSequenceInputValue(
		firstRow+lastRow,
		benchmarkDataViewsSequenceInputs-1,
	) {
		tb.Fatalf("sequence input last value = %g, want aligned row %d", got, firstRow+lastRow)
	}
	if got, err = targets.At(lastRow, benchmarkDataViewsSequenceTargets-1); err != nil {
		tb.Fatalf("verify sequence target last value returned error: %v", err)
	} else if got != benchmarkDataViewsSequenceTargetValue(
		firstRow+lastRow,
		benchmarkDataViewsSequenceTargets-1,
	) {
		tb.Fatalf("sequence target last value = %g, want aligned row %d", got, firstRow+lastRow)
	}
	if lengthValues, err = lengths.Values(); err != nil {
		tb.Fatalf("verify sequence length values returned error: %v", err)
	}
	if lengthValues[0] != firstRow%benchmarkDataViewsSequenceSteps+1 {
		tb.Fatalf("sequence first length = %d, want aligned row %d", lengthValues[0], firstRow)
	}
	if lengthValues[lastRow] != (firstRow+lastRow)%benchmarkDataViewsSequenceSteps+1 {
		tb.Fatalf(
			"sequence last length = %d, want aligned row %d",
			lengthValues[lastRow],
			firstRow+lastRow,
		)
	}
}

func verifyDataViewsSequenceBatches(
	tb testing.TB,
	batches []*data.SequenceBatch,
) {
	var (
		batch   *data.SequenceBatch
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		lengths *data.SequenceLengths
		err     error
	)

	tb.Helper()

	if len(batches) == 0 {
		tb.Fatal("sequence batch result is empty")
	}
	for _, batch = range batches {
		if inputs, err = batch.Inputs(); err != nil {
			tb.Fatalf("verify sequence batch Inputs returned error: %v", err)
		}
		if targets, err = batch.Targets(); err != nil {
			tb.Fatalf("verify sequence batch Targets returned error: %v", err)
		}
		if lengths, err = batch.Lengths(); err != nil {
			tb.Fatalf("verify sequence batch Lengths returned error: %v", err)
		}
		verifyDataViewsSequenceAlignedValues(tb, inputs, targets, lengths)
	}
}

func verifyDataViewsSequenceAlignedDataset(
	tb testing.TB,
	dataset *data.SequenceDataset,
) {
	var (
		inputs  *matrix.Matrix
		targets *matrix.Matrix
		lengths *data.SequenceLengths
		err     error
	)

	tb.Helper()

	if inputs, err = dataset.Inputs(); err != nil {
		tb.Fatalf("verify aligned sequence Inputs returned error: %v", err)
	}
	if targets, err = dataset.Targets(); err != nil {
		tb.Fatalf("verify aligned sequence Targets returned error: %v", err)
	}
	if lengths, err = dataset.Lengths(); err != nil {
		tb.Fatalf("verify aligned sequence Lengths returned error: %v", err)
	}
	verifyDataViewsSequenceAlignedValues(tb, inputs, targets, lengths)
}

func verifyDataViewsSequenceAlignedValues(
	tb testing.TB,
	inputs,
	targets *matrix.Matrix,
	lengths *data.SequenceLengths,
) {
	var (
		lengthValues []int
		row          int
		sourceRow    int
		inputValue   float32
		targetValue  float32
		err          error
	)

	tb.Helper()

	if inputs.Rows() != targets.Rows() || inputs.Rows() != lengths.SampleCount() {
		tb.Fatalf(
			"aligned sequence rows differ: inputs=%d targets=%d lengths=%d",
			inputs.Rows(),
			targets.Rows(),
			lengths.SampleCount(),
		)
	}
	if lengthValues, err = lengths.Values(); err != nil {
		tb.Fatalf("aligned sequence length Values returned error: %v", err)
	}
	for row = 0; row < inputs.Rows(); row++ {
		if inputValue, err = inputs.At(row, 0); err != nil {
			tb.Fatalf("aligned sequence input At returned error: %v", err)
		}
		sourceRow = int(inputValue) / 8192
		if inputValue != benchmarkDataViewsSequenceInputValue(sourceRow, 0) {
			tb.Fatalf(
				"sequence input row %d does not encode a source row: value=%g",
				row,
				inputValue,
			)
		}
		if targetValue, err = targets.At(row, 0); err != nil {
			tb.Fatalf("aligned sequence target At returned error: %v", err)
		}
		if targetValue != benchmarkDataViewsSequenceTargetValue(sourceRow, 0) {
			tb.Fatalf(
				"sequence target row %d is misaligned: got=%g sourceRow=%d",
				row,
				targetValue,
				sourceRow,
			)
		}
		if lengthValues[row] != sourceRow%benchmarkDataViewsSequenceSteps+1 {
			tb.Fatalf(
				"sequence length row %d is misaligned: got=%d sourceRow=%d",
				row,
				lengthValues[row],
				sourceRow,
			)
		}
	}
}

func verifyDataViewsSequenceSelection(
	tb testing.TB,
	indexes []int,
	inputs,
	targets *matrix.Matrix,
	lengths []int,
) {
	var (
		position    int
		sourceRow   int
		inputValue  float32
		targetValue float32
		err         error
	)

	tb.Helper()

	for position, sourceRow = range indexes {
		if inputValue, err = inputs.At(position, 0); err != nil {
			tb.Fatalf("selected sequence input At returned error: %v", err)
		}
		if inputValue != benchmarkDataViewsSequenceInputValue(sourceRow, 0) {
			tb.Fatalf(
				"selected sequence input row %d = %g, want source row %d",
				position,
				inputValue,
				sourceRow,
			)
		}
		if targetValue, err = targets.At(position, 0); err != nil {
			tb.Fatalf("selected sequence target At returned error: %v", err)
		}
		if targetValue != benchmarkDataViewsSequenceTargetValue(sourceRow, 0) {
			tb.Fatalf(
				"selected sequence target row %d = %g, want source row %d",
				position,
				targetValue,
				sourceRow,
			)
		}
		if lengths[position] != sourceRow%benchmarkDataViewsSequenceSteps+1 {
			tb.Fatalf(
				"selected sequence length row %d = %d, want source row %d",
				position,
				lengths[position],
				sourceRow,
			)
		}
	}
}

func benchmarkDataViewsMatrixChecksum(
	tb testing.TB,
	inputs,
	targets *matrix.Matrix,
) (checksum float32) {
	var (
		inputValue  float32
		targetValue float32
		err         error
	)

	tb.Helper()

	if inputValue, err = inputs.At(inputs.Rows()-1, inputs.Cols()-1); err != nil {
		tb.Fatalf("checksum input At returned error: %v", err)
	}
	if targetValue, err = targets.At(targets.Rows()-1, targets.Cols()-1); err != nil {
		tb.Fatalf("checksum target At returned error: %v", err)
	}
	checksum = inputValue + targetValue
	return checksum
}

func benchmarkSequenceDataViewsSelectionChecksum(
	tb testing.TB,
	inputs,
	targets *matrix.Matrix,
	lengths []int,
) (checksum float32) {
	tb.Helper()

	checksum = benchmarkDataViewsMatrixChecksum(tb, inputs, targets)
	checksum += float32(lengths[len(lengths)-1])
	return checksum
}

func benchmarkDataViewsOrdinaryInputValue(row, column int) (value float32) {
	value = float32(row*512 + column)
	return value
}

func benchmarkDataViewsOrdinaryTargetValue(row, column int) (value float32) {
	value = float32(row*32 + column)
	return value
}

func benchmarkDataViewsSequenceInputValue(row, column int) (value float32) {
	value = float32(row*8192 + column)
	return value
}

func benchmarkDataViewsSequenceTargetValue(row, column int) (value float32) {
	value = float32(row*16 + column)
	return value
}

func benchmarkDataViewsOrdinaryBytes(samples int) (bytes int64) {
	bytes = int64(samples) *
		int64(benchmarkDataViewsOrdinaryInputs+benchmarkDataViewsOrdinaryTargets) *
		4
	return bytes
}

func benchmarkDataViewsSequenceBytes(samples int) (bytes int64) {
	var matrixBytes int64

	matrixBytes = int64(samples) *
		int64(benchmarkDataViewsSequenceInputs+benchmarkDataViewsSequenceTargets) *
		4
	bytes = matrixBytes + benchmarkDataViewsLengthBytes(samples)
	return bytes
}

func benchmarkDataViewsLengthBytes(samples int) (bytes int64) {
	bytes = int64(samples) * int64(strconv.IntSize/8)
	return bytes
}

func reportDataViewsMetrics(
	b *testing.B,
	logicalBytesRead,
	logicalBytesCopied int64,
) {
	b.Helper()

	b.ReportAllocs()
	b.SetBytes(logicalBytesRead)
	b.ReportMetric(float64(logicalBytesRead), "logical-B-read/op")
	b.ReportMetric(float64(logicalBytesCopied), "logical-B-copied/op")
}
