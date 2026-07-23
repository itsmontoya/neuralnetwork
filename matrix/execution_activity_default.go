//go:build !metal || purego || !darwin || !cgo

package matrix

import "github.com/itsmontoya/neuralnetwork/internal/device"

func recordExecutionActivity(device.ExecutionSnapshot) {}
