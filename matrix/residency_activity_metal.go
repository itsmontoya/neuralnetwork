//go:build darwin && cgo && metal && !purego

package matrix

import "github.com/itsmontoya/neuralnetwork/internal/metaltest"

func recordResidencyDownload() {
	metaltest.RecordBridgeActivity(0, 0, 1, 0, 0)
}
