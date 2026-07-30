package model

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/data"
)

var allocationViewFitMetrics EpochMetrics

func Test_Sequential_OrderedViewFitEpochUsesOnlyBoundedMetadataAllocations(
	t *testing.T,
) {
	var (
		network     *Sequential
		dataset     *data.Dataset
		config      ViewFitConfig
		scratch     fitScratch
		allocations float64
		err         error
	)

	network = viewFitOrdinaryModel(t)
	dataset = viewFitOrdinaryDataset(t)
	config = viewFitValidConfig(t)
	config.FitConfig.ValidationData = dataset
	if allocationViewFitMetrics, err = runAllocationViewFitEpoch(
		network,
		dataset,
		config,
		1,
		&scratch,
	); err != nil {
		t.Fatalf("warm-up view fit epoch returned error: %v", err)
	}

	allocations = testing.AllocsPerRun(100, func() {
		if allocationViewFitMetrics, err = runAllocationViewFitEpoch(
			network,
			dataset,
			config,
			2,
			&scratch,
		); err != nil {
			panic(err)
		}
	})
	if allocations > 24 {
		t.Fatalf(
			"warmed ordered view fit epoch allocations = %g, want <= 24 metadata allocations",
			allocations,
		)
	}
}

func Test_Sequential_OrderedLengthViewFitEpochUsesOnlyBoundedMetadataAllocations(
	t *testing.T,
) {
	var (
		network       *Sequential
		dataset       *data.SequenceDataset
		config        SequenceViewFitConfig
		selector      sequenceLengthLayer
		selectorIndex int
		scratch       fitScratch
		allocations   float64
		err           error
	)

	network = viewFitSequenceModel(t)
	dataset = viewFitSequenceDataset(t)
	config = viewFitValidSequenceConfig(t)
	config.SequenceFitConfig.ValidationData = dataset
	if selector, selectorIndex, err = network.lengthAwareGraph(
		"allocation test",
	); err != nil {
		t.Fatalf("lengthAwareGraph returned error: %v", err)
	}
	if allocationViewFitMetrics, err = runAllocationLengthViewFitEpoch(
		network,
		dataset,
		config,
		selector,
		selectorIndex,
		1,
		&scratch,
	); err != nil {
		t.Fatalf("warm-up length-view fit epoch returned error: %v", err)
	}

	allocations = testing.AllocsPerRun(100, func() {
		if allocationViewFitMetrics, err = runAllocationLengthViewFitEpoch(
			network,
			dataset,
			config,
			selector,
			selectorIndex,
			2,
			&scratch,
		); err != nil {
			panic(err)
		}
	})
	if allocations > 24 {
		t.Fatalf(
			"warmed ordered length-view fit epoch allocations = %g, want <= 24 metadata allocations",
			allocations,
		)
	}
}

func runAllocationViewFitEpoch(
	network *Sequential,
	dataset *data.Dataset,
	config ViewFitConfig,
	epoch int,
	scratch *fitScratch,
) (metrics EpochMetrics, err error) {
	if err = network.trainViewFitEpoch(
		dataset,
		config,
		epoch,
		scratch,
	); err != nil {
		return metrics, err
	}
	metrics, err = network.viewFitEpochMetrics(
		epoch,
		dataset,
		config.FitConfig,
	)
	return metrics, err
}

func runAllocationLengthViewFitEpoch(
	network *Sequential,
	dataset *data.SequenceDataset,
	config SequenceViewFitConfig,
	selector sequenceLengthLayer,
	selectorIndex,
	epoch int,
	scratch *fitScratch,
) (metrics EpochMetrics, err error) {
	if err = network.trainLengthViewFitEpoch(
		dataset,
		config,
		epoch,
		selector,
		selectorIndex,
		scratch,
	); err != nil {
		return metrics, err
	}
	metrics, err = network.lengthViewFitEpochMetrics(
		epoch,
		dataset,
		config.SequenceFitConfig,
		selector,
		selectorIndex,
	)
	return metrics, err
}
