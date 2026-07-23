//go:build darwin && cgo && metal && !purego

package model_test

import (
	"testing"

	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
)

func beginResidentPredictMetrics() {
	metaltest.Enable()
}

func endResidentPredictMetrics(b *testing.B) {
	var (
		counters   metaltest.Counters
		iterations float64
	)

	counters = metaltest.Snapshot()
	metaltest.Disable()
	iterations = float64(b.N)
	b.ReportMetric(float64(counters.BufferCreations)/iterations, "buffers/op")
	b.ReportMetric(float64(counters.InputUploads)/iterations, "uploads/op")
	b.ReportMetric(float64(counters.InputUploadBytes)/iterations, "upload-bytes/op")
	b.ReportMetric(float64(counters.ResultDownloads)/iterations, "downloads/op")
	b.ReportMetric(float64(counters.ResultDownloadBytes)/iterations, "download-bytes/op")
	b.ReportMetric(float64(counters.CommandSubmissions)/iterations, "commands/op")
	b.ReportMetric(float64(counters.Waits)/iterations, "waits/op")
}

func beginResidentBackwardMetrics() {
	beginResidentPredictMetrics()
}

func endResidentBackwardMetrics(b *testing.B) {
	endResidentPredictMetrics(b)
}

func beginResidentTrainingMetrics() {
	beginResidentPredictMetrics()
}

func endResidentTrainingMetrics(b *testing.B) {
	endResidentPredictMetrics(b)
}
