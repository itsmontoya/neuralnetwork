//go:build !darwin || !cgo || !metal || purego

package model_test

import "testing"

func beginResidentPredictMetrics() {}

func endResidentPredictMetrics(*testing.B) {}

func beginResidentBackwardMetrics() {}

func endResidentBackwardMetrics(*testing.B) {}
