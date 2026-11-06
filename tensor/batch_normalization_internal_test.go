package tensor

import (
	"fmt"
	"slices"
	"testing"
)

func Test_BatchNormalization_ParameterStorageAliases(t *testing.T) {
	for _, channels := range []int64{0, 2} {
		for parameterIndex := 0; parameterIndex < 4; parameterIndex++ {
			t.Run(fmt.Sprint(channels, "/", parameterIndex), func(t *testing.T) {
				var (
					input      *Tensor
					parameters = make([]*Tensor, 4)
					output     Tensor
					before     []float32
					err        error
				)
				if input, err = New(Float32, []int64{1, channels}); err != nil {
					t.Fatal(err)
				}
				for index := range parameters {
					if parameters[index], err = New(Float32, []int64{channels}); err != nil {
						t.Fatal(err)
					}
					for element := range parameters[index].floats {
						parameters[index].floats[element] = float32(19 + element)
					}
				}
				// No public view API creates these distinct shapes over one buffer;
				// exercise the storage-identity guard independently of shape errors.
				output = *parameters[parameterIndex]
				output.shape = []int64{1, channels}
				before = slices.Clone(output.floats)
				if err = BatchNormalizationInto(input, parameters[0], parameters[1], parameters[2], parameters[3], &output, 0.125); err == nil {
					t.Fatal("destination sharing parameter storage succeeded")
				}
				if !slices.Equal(output.floats, before) {
					t.Fatal("parameter storage changed on error")
				}
			})
		}
	}
}
