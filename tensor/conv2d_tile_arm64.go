//go:build arm64 && !purego

package tensor

const conv2DHasInteriorTile = true

// conv2DInteriorTile uses SIMD lanes for adjacent spatial outputs, one vector
// per filter. The caller clips padding and bounds complete input/weight spans.
// ARM64's scalar Go path also uses fused float32 multiply-adds.
//
//go:noescape
func conv2DInteriorTile(input, weights []float32, sums *[16]float32, channels, rows, kernelW, inputPlane, inputW, kernelPlane int)
