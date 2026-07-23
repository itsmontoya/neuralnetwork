//go:build !metal || purego || !darwin || !cgo

package matrix

func recordResidencyDownload(uint64) {}
