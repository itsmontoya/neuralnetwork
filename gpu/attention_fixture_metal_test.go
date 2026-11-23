//go:build darwin && arm64 && cgo && metal && !purego

package gpu

import (
	"math"
	"slices"
	"testing"
)

func newAttentionFixture(t testing.TB, d *Device, heads, kvHeads, width, positions int64) (fixture *attentionFixture) {
	t.Helper()
	var f attentionFixture
	f.heads, f.kvHeads, f.width = heads, kvHeads, width
	f.hidden = heads * width
	f.kvWidth = kvHeads * width
	f.positions = positions
	f.cosine = requireTensor(t, d, positions, width/2)
	f.sine = requireTensor(t, d, positions, width/2)
	f.norm = requireTensor(t, d, f.hidden)
	f.q = requireTensor(t, d, f.hidden, f.hidden)
	f.k = requireTensor(t, d, f.kvWidth, f.hidden)
	f.v = requireTensor(t, d, f.kvWidth, f.hidden)
	f.o = requireTensor(t, d, f.hidden, f.hidden)
	return &f
}

type attentionFixture struct {
	heads, kvHeads, width, hidden, kvWidth, positions int64
	cosine, sine, norm, q, k, v, o                    *Tensor
	cosValues, sinValues, normValues                  []float32
	qValues, kValues, vValues, oValues                []float32
}

func (f *attentionFixture) prepare() {
	var scale = float32(1 / math.Sqrt(float64(f.hidden)))
	f.cosValues, f.sinValues = attentionCoefficients(f.positions, f.width)
	f.normValues = attentionValues(f.hidden, 19, 0.25)
	for index := range f.normValues {
		f.normValues[index] += 1
	}
	f.qValues = attentionValues(f.hidden*f.hidden, 31, scale)
	f.kValues = attentionValues(f.kvWidth*f.hidden, 37, scale)
	f.vValues = attentionValues(f.kvWidth*f.hidden, 41, scale)
	f.oValues = attentionValues(f.hidden*f.hidden, 43, scale)
}

func (f *attentionFixture) upload(t testing.TB) {
	t.Helper()
	for _, source := range []struct {
		tensor *Tensor
		values []float32
	}{
		{f.cosine, f.cosValues}, {f.sine, f.sinValues}, {f.norm, f.normValues},
		{f.q, f.qValues}, {f.k, f.kValues}, {f.v, f.vValues}, {f.o, f.oValues},
	} {
		requireError(t, source.tensor.Upload(source.values), nil)
	}
}

func (f *attentionFixture) reference(input []float32, position int64, oldKeys, oldValues []float64) (output, keys, values []float64) {
	var (
		normalized = referenceNorm(input, f.normValues)
		q          = referenceProjection(normalized, f.qValues, f.hidden)
		k          = referenceProjection(normalized, f.kValues, f.hidden)
		v          = referenceProjection(normalized, f.vValues, f.hidden)
	)
	referenceRotate(q, f.cosValues, f.sinValues, position, f.heads, f.width)
	referenceRotate(k, f.cosValues, f.sinValues, position, f.kvHeads, f.width)
	keys = append(slices.Clone(oldKeys), k...)
	values = append(slices.Clone(oldValues), v...)
	var attended = referenceAttention(q, keys, values, f.heads, f.kvHeads, f.width, position)
	output = referenceProjection(attended, f.oValues, f.hidden)
	for index, residual := range input {
		output[index] += float64(residual)
	}
	return output, keys, values
}
