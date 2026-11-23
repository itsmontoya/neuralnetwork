package gpu_test

import (
	"context"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/gpu"
)

func Example() {
	var ctx = context.Background()
	var d, err = gpu.OpenDefault()
	if err != nil {
		panic(err)
	}
	defer d.Close()

	var output *gpu.Tensor
	if output, err = d.NewFloat32([]int64{2, 3}); err != nil {
		panic(err)
	}
	defer output.Close()

	var scope *gpu.Scope
	if scope, err = d.NewScope(); err != nil {
		panic(err)
	}
	defer scope.Close()
	if err = scope.Fill(output, 2); err != nil {
		panic(err)
	}
	if err = scope.Submit(ctx); err != nil {
		panic(err)
	}
	if err = scope.Wait(ctx); err != nil {
		panic(err)
	}

	var values [6]float32
	if err = output.DownloadInto(values[:]); err != nil {
		panic(err)
	}
	fmt.Println(values) // [2 2 2 2 2 2]
}
