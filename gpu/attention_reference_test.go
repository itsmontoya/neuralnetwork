//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"math"
	"math/rand/v2"
)

const attentionEpsilon float32 = 1e-5

func attentionValues(count int64, seed uint64, scale float32) (values []float32) {
	var random = rand.New(rand.NewPCG(seed, 23))
	values = make([]float32, count)
	for index := range values {
		values[index] = (2*random.Float32() - 1) * scale
	}
	return values
}

func attentionCoefficients(positions, width int64) (cosine, sine []float32) {
	cosine = make([]float32, positions*width/2)
	sine = make([]float32, len(cosine))
	for row := range positions {
		for pair := range width / 2 {
			var (
				angle = float64(row) / math.Pow(10_000, float64(2*pair)/float64(width))
				s, c  = math.Sincos(angle)
				index = row*(width/2) + pair
			)
			cosine[index], sine[index] = float32(c), float32(s)
		}
	}
	return cosine, sine
}

func referenceNorm(input, weight []float32) (output []float64) {
	output = make([]float64, len(input))
	var width = len(weight)
	for start := 0; start < len(input); start += width {
		var square float64
		for _, value := range input[start : start+width] {
			square += float64(value) * float64(value)
		}
		var rms = math.Sqrt(square/float64(width) + float64(attentionEpsilon))
		for index, value := range input[start : start+width] {
			output[start+index] = float64(value) / rms * float64(weight[index])
		}
	}
	return output
}

func referenceProjection(input []float64, weight []float32, width int64) (output []float64) {
	var (
		rows = int64(len(input)) / width
		cols = int64(len(weight)) / width
	)
	output = make([]float64, rows*cols)
	for row := range rows {
		var left = input[row*width : (row+1)*width]
		for col := range cols {
			var (
				right = weight[col*width : (col+1)*width]
				dot   float64
			)
			for index, value := range left {
				dot += value * float64(right[index])
			}
			output[row*cols+col] = dot
		}
	}
	return output
}

func referenceRotate(values []float64, cosine, sine []float32, position, heads, width int64) {
	for row := range int64(len(values)) / (heads * width) {
		for head := range heads {
			for pair := range width / 2 {
				var (
					start = (row*heads+head)*width + 2*pair
					coef  = (position+row)*(width/2) + pair
					c     = float64(cosine[coef])
					s     = float64(sine[coef])
					x, y  = values[start], values[start+1]
				)
				values[start] = x*c - y*s
				values[start+1] = y*c + x*s
			}
		}
	}
}

func referenceAttention(queries, keys, values []float64, queryHeads, kvHeads, width, start int64) (output []float64) {
	var (
		history = int64(len(keys)) / (kvHeads * width)
		rows    = int64(len(queries)) / (queryHeads * width)
		scores  = make([]float64, history)
		scale   = 1 / math.Sqrt(float64(width))
	)
	output = make([]float64, len(queries))
	for row := range rows {
		for head := range queryHeads {
			var (
				queryStart = (row*queryHeads + head) * width
				kvHead     = head / (queryHeads / kvHeads)
				visible    = min(start+row+1, history)
				maximum    = math.Inf(-1)
				sum        float64
			)
			for keyRow := range visible {
				var (
					keyStart = (keyRow*kvHeads + kvHead) * width
					dot      float64
				)
				for feature := range width {
					dot += queries[queryStart+feature] * keys[keyStart+feature]
				}
				scores[keyRow] = dot * scale
				maximum = max(maximum, scores[keyRow])
			}
			for keyRow := range visible {
				scores[keyRow] = math.Exp(scores[keyRow] - maximum)
				sum += scores[keyRow]
			}
			for keyRow := range visible {
				var (
					probability = scores[keyRow] / sum
					valueStart  = (keyRow*kvHeads + kvHead) * width
				)
				for feature := range width {
					output[queryStart+feature] += probability * values[valueStart+feature]
				}
			}
		}
	}
	return output
}
