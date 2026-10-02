package model_test

import (
	"bytes"
	"testing"

	"github.com/itsmontoya/neuralnetwork/model"
)

func Benchmark_SequentialArtifactLoad(b *testing.B) {
	var (
		network        *model.Sequential
		jsonDocument   bytes.Buffer
		binaryDocument bytes.Buffer
		err            error
	)

	network = benchmarkOCRCandidateModel(b)
	if err = network.Save(&jsonDocument); err != nil {
		b.Fatalf("Save returned error: %v", err)
	}
	if _, err = network.SaveBinary(&binaryDocument); err != nil {
		b.Fatalf("SaveBinary returned error: %v", err)
	}

	b.Run("JSON", func(b *testing.B) {
		benchmarkSequentialJSONLoad(b, jsonDocument.Bytes())
	})
	b.Run("Binary", func(b *testing.B) {
		benchmarkSequentialBinaryLoad(b, binaryDocument.Bytes())
	})
}

func benchmarkSequentialJSONLoad(b *testing.B, document []byte) {
	var (
		loaded *model.Sequential
		err    error
		index  int
	)

	b.ReportAllocs()
	b.SetBytes(int64(len(document)))
	b.ResetTimer()
	b.ReportMetric(float64(len(document)), "bytes/artifact")
	for index = 0; index < b.N; index++ {
		if loaded, err = model.LoadSequential(bytes.NewReader(document)); err != nil {
			b.Fatalf("LoadSequential returned error: %v", err)
		}
	}
	benchmarkArtifactModel = loaded
}

func benchmarkSequentialBinaryLoad(b *testing.B, document []byte) {
	var (
		loaded *model.Sequential
		err    error
		index  int
	)

	b.ReportAllocs()
	b.SetBytes(int64(len(document)))
	b.ResetTimer()
	b.ReportMetric(float64(len(document)), "bytes/artifact")
	for index = 0; index < b.N; index++ {
		if loaded, _, err = model.LoadSequentialBinary(bytes.NewReader(document)); err != nil {
			b.Fatalf("LoadSequentialBinary returned error: %v", err)
		}
	}
	benchmarkArtifactModel = loaded
}

var benchmarkArtifactModel *model.Sequential
