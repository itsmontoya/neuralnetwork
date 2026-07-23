//go:build darwin && cgo && metal && !purego

package matrix

import (
	"github.com/itsmontoya/neuralnetwork/internal/device"
	"github.com/itsmontoya/neuralnetwork/internal/metaltest"
)

func recordExecutionActivity(snapshot device.ExecutionSnapshot) {
	metaltest.RecordBridgeActivity(
		snapshot.BufferCreations,
		snapshot.InputUploads,
		snapshot.InputUploadBytes,
		snapshot.ResultDownloads,
		snapshot.ResultDownloadBytes,
		snapshot.KernelEncodes,
		snapshot.CommandSubmissions,
		snapshot.Waits,
		snapshot.Barriers,
		snapshot.FallbackBarriers,
	)
}
