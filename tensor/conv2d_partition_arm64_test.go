//go:build arm64 && !purego

package tensor

import (
	"math"
	"testing"
)

// BenchmarkConv2DPartitions partitions the original four-column traversal. It includes per-call geometry and destination stores.
// Synthetic finite payloads make this a diagnostic, not the ORT comparison.
func BenchmarkConv2DPartitions(b *testing.B) {
	var tests = []struct {
		name                     string
		channels, width, filters int64
	}{
		{name: "First", channels: 1, width: 28, filters: 8},
		{name: "Second", channels: 8, width: 14, filters: 16},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(convolution *testing.B) {
			var (
				input, weights, output, reference                 *Tensor
				shape                                             []int64
				scalarColumns, paddingColumns, tailColumns, tiles []int
				config                                            = Conv2DConfig{Strides: [2]int64{1, 1}, Dilations: [2]int64{1, 1}, Pads: [4]int64{2, 2, 2, 2}, Groups: 1}
				err                                               error
			)

			if input, err = New(Float32, []int64{1, tt.channels, tt.width, tt.width}); err != nil {
				convolution.Fatal(err)
			}

			if weights, err = New(Float32, []int64{tt.filters, tt.channels, 5, 5}); err != nil {
				convolution.Fatal(err)
			}

			for index := range input.floats {
				input.floats[index] = float32(index%17-8) * 0.125
			}
			for index := range weights.floats {
				weights.floats[index] = float32(index%13-6) * 0.125
			}
			if shape, err = Conv2DShape(input.shape, weights.shape, config); err != nil {
				convolution.Fatal(err)
			}

			if output, err = New(Float32, shape); err != nil {
				convolution.Fatal(err)
			}

			if reference, err = New(Float32, shape); err != nil {
				convolution.Fatal(err)
			}

			for column := 0; column < int(tt.width); column++ {
				start := column - 2
				if column+3 < int(tt.width) && start >= 0 && start+7 < int(tt.width) {
					tiles = append(tiles, column)
					column += 3
					continue
				}

				scalarColumns = append(scalarColumns, column)
				if start < 0 || start+4 >= int(tt.width) {
					paddingColumns = append(paddingColumns, column)
					continue
				}

				tailColumns = append(tailColumns, column)
			}
			if err = Conv2DInto(input, weights, nil, reference, config); err != nil {
				convolution.Fatal(err)
			}

			benchmarkConv2DPartition(input, weights, output, config, scalarColumns, false, false)
			benchmarkConv2DPartition(input, weights, output, config, tiles, true, false)
			for index, expected := range reference.floats {
				if math.Float32bits(output.floats[index]) != math.Float32bits(expected) {
					convolution.Fatalf("partition differs at output %d", index)
				}
			}
			benchmarkConv2DPartition(input, weights, output, config, scalarColumns, false, true)
			for index, expected := range reference.floats {
				if math.Float32bits(output.floats[index]) != math.Float32bits(expected) {
					convolution.Fatalf("residual tile partition differs at output %d", index)
				}
			}
			convolution.Logf("scalar=%v padding=%v interior tail=%v tiles=%v", scalarColumns, paddingColumns, tailColumns, tiles)
			convolution.Run("Complete", func(complete *testing.B) {
				complete.ReportAllocs()
				for complete.Loop() {
					if err = Conv2DInto(input, weights, nil, output, config); err != nil {
						complete.Fatal(err)
					}
				}
			})
			for _, path := range []struct {
				name          string
				columns       []int
				interior      bool
				residualTiles bool
			}{
				{name: "ResidualColumns", columns: scalarColumns},
				{name: "PaddingColumns", columns: paddingColumns},
				{name: "InteriorTailColumns", columns: tailColumns},
				{name: "InteriorTiles", columns: tiles, interior: true},
				{name: "ResidualTiles", columns: scalarColumns, residualTiles: true},
				{name: "PaddingTiles", columns: paddingColumns, residualTiles: true},
				{name: "InteriorTailTiles", columns: tailColumns, residualTiles: true},
			} {
				if len(path.columns) == 0 {
					continue
				}

				convolution.Run(path.name, func(partition *testing.B) {
					partition.ReportAllocs()
					for partition.Loop() {
						benchmarkConv2DPartition(input, weights, output, config, path.columns, path.interior, path.residualTiles)
					}
				})
			}
		})
	}
}

func benchmarkConv2DPartition(input, weights, output *Tensor, config Conv2DConfig, columns []int, interior bool, residualTiles bool) {
	var (
		width  = int(output.shape[3])
		height = int(output.shape[2])
		plane  = height * width
	)

	for channel := 0; channel < int(output.shape[1]); channel += 4 {
		for row := 0; row < height; row++ {
			for position := 0; position < len(columns); position++ {
				var column = columns[position]
				offset := channel*plane + row*width + column
				if interior {
					if !conv2DFourByFour(input, weights, nil, output, 0, channel, row, column-2, offset, config) {
						panic("ARM64 interior tile unavailable")
					}

					continue
				}

				if residualTiles {
					var count = 1

					for count < 4 && position+count < len(columns) && columns[position+count] == column+count {
						count++
					}
					if count > 1 {
						conv2DResidualTile(input, weights, nil, output, 0, channel, row, column-2, offset, count, config)
						position += count - 1
						continue
					}
				}

				if !conv2DResidualChannels(input, weights, nil, output, 0, channel, row, column-2, offset, config) {
					panic("ARM64 residual assembly unavailable")
				}
			}
		}
	}
}
