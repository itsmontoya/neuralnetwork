package main

import (
	"fmt"
	"log"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	var (
		segmented     *activation.SegmentedSoftmax
		logits        *matrix.Matrix
		probabilities *matrix.Matrix
		animalIndex   int
		colorIndex    int
	)

	if segmented, err = activation.NewSegmentedSoftmax(2, 3); err != nil {
		return err
	}
	if logits, err = matrix.FromSlice(1, 5, []float32{0, 2, 3, 1, -1}); err != nil {
		return err
	}
	if probabilities, err = segmented.Forward(logits); err != nil {
		return err
	}
	if animalIndex, err = segmentArgmax(probabilities, 0, 2); err != nil {
		return err
	}
	if colorIndex, err = segmentArgmax(probabilities, 2, 3); err != nil {
		return err
	}

	fmt.Printf("animal=%s color=%s\n", []string{"cat", "dog"}[animalIndex], []string{"red", "green", "blue"}[colorIndex])
	return nil
}

func segmentArgmax(values *matrix.Matrix, start, width int) (index int, err error) {
	var (
		column  int
		value   float32
		maximum float32
	)

	if maximum, err = values.At(0, start); err != nil {
		return 0, err
	}
	for column = 1; column < width; column++ {
		if value, err = values.At(0, start+column); err != nil {
			return 0, err
		}
		if value > maximum {
			maximum = value
			index = column
		}
	}

	return index, nil
}
