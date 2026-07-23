//go:build darwin && cgo && metal && !purego

package matrix

import "github.com/itsmontoya/neuralnetwork/internal/metaltest"

func recordResidencyDownload(bytes uint64) {
	metaltest.RecordBridgeActivity(0, 0, 0, 1, bytes, 0, 0)
}
