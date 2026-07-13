package model

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"

	activationpkg "github.com/itsmontoya/neuralnetwork/activation"
	layerpkg "github.com/itsmontoya/neuralnetwork/layer"
	matrixpkg "github.com/itsmontoya/neuralnetwork/matrix"
)

const (
	serializationFormatSequential        = "neuralnetwork.sequential"
	serializationLayerActivation         = "activation"
	serializationLayerBatchNormalization = "batch_normalization"
	serializationLayerDense              = "dense"
	serializationLayerDropout            = "dropout"
	serializationDropoutSeed             = 1
	serializationVersion                 = 1
)

type sequentialDocument struct {
	Format  string            `json:"format"`
	Version int               `json:"version"`
	Layers  []serializedLayer `json:"layers"`
}

func sequentialDocumentFromModel(s *Sequential) (document sequentialDocument, err error) {
	var (
		index        int
		currentLayer layerpkg.Layer
		serialized   serializedLayer
	)

	if s == nil {
		err = errors.New("model: sequential model is nil")
		return document, err
	}

	document = sequentialDocument{
		Format:  serializationFormatSequential,
		Version: serializationVersion,
		Layers:  make([]serializedLayer, 0, len(s.layers)),
	}

	for index, currentLayer = range s.layers {
		if serialized, err = serializedLayerFromLayer(index, currentLayer); err != nil {
			return document, err
		}

		document.Layers = append(document.Layers, serialized)
	}

	return document, nil
}

func (d sequentialDocument) model() (s *Sequential, err error) {
	var (
		index        int
		serialized   serializedLayer
		currentLayer layerpkg.Layer
	)

	if d.Format != serializationFormatSequential {
		err = fmt.Errorf("model: unsupported serialization format %q", d.Format)
		return nil, err
	}

	if d.Version != serializationVersion {
		err = fmt.Errorf("model: unsupported serialization version %d", d.Version)
		return nil, err
	}

	if s, err = NewSequential(); err != nil {
		return nil, err
	}

	for index, serialized = range d.Layers {
		if currentLayer, err = serialized.layer(index); err != nil {
			return nil, err
		}

		if err = s.Add(currentLayer); err != nil {
			err = fmt.Errorf("model: layer %d add failed: %w", index, err)
			return nil, err
		}
	}

	return s, nil
}

type serializedLayer struct {
	Type            string            `json:"type"`
	InputSize       int               `json:"input_size,omitempty"`
	OutputSize      int               `json:"output_size,omitempty"`
	FeatureSize     int               `json:"feature_size,omitempty"`
	Activation      string            `json:"activation,omitempty"`
	Rate            float32           `json:"rate,omitempty"`
	Momentum        float32           `json:"momentum,omitempty"`
	Epsilon         float32           `json:"epsilon,omitempty"`
	Weights         *serializedMatrix `json:"weights,omitempty"`
	Biases          *serializedMatrix `json:"biases,omitempty"`
	Gamma           *serializedMatrix `json:"gamma,omitempty"`
	Beta            *serializedMatrix `json:"beta,omitempty"`
	RunningMean     *serializedMatrix `json:"running_mean,omitempty"`
	RunningVariance *serializedMatrix `json:"running_variance,omitempty"`
}

func serializedLayerFromLayer(index int, currentLayer layerpkg.Layer) (serialized serializedLayer, err error) {
	if currentLayer == nil {
		err = fmt.Errorf("model: layer %d is nil", index)
		return serialized, err
	}

	switch current := currentLayer.(type) {
	case *layerpkg.Activation:
		serialized, err = serializedActivationLayer(index, current)
	case *layerpkg.BatchNormalization:
		serialized, err = serializedBatchNormalizationLayer(index, current)
	case *layerpkg.Dense:
		serialized, err = serializedDenseLayer(index, current)
	case *layerpkg.Dropout:
		serialized, err = serializedDropoutLayer(index, current)
	default:
		err = fmt.Errorf("model: layer %d unsupported layer type %T", index, currentLayer)
		return serialized, err
	}

	return serialized, err
}

func (s serializedLayer) layer(index int) (currentLayer layerpkg.Layer, err error) {
	switch s.Type {
	case serializationLayerActivation:
		currentLayer, err = s.activationLayer(index)
	case serializationLayerBatchNormalization:
		currentLayer, err = s.batchNormalizationLayer(index)
	case serializationLayerDense:
		currentLayer, err = s.denseLayer(index)
	case serializationLayerDropout:
		currentLayer, err = s.dropoutLayer(index)
	default:
		err = fmt.Errorf("model: layer %d unknown layer type %q", index, s.Type)
		return nil, err
	}

	return currentLayer, err
}

func serializedActivationLayer(index int, activationLayer *layerpkg.Activation) (serialized serializedLayer, err error) {
	var name string

	if activationLayer == nil {
		err = fmt.Errorf("model: layer %d activation layer is nil", index)
		return serialized, err
	}

	if name, err = activationpkg.Name(activationLayer.Function()); err != nil {
		err = fmt.Errorf("model: layer %d activation name failed: %w", index, err)
		return serialized, err
	}

	serialized = serializedLayer{
		Type:       serializationLayerActivation,
		Activation: name,
	}
	return serialized, nil
}

func (s serializedLayer) activationLayer(index int) (activationLayer *layerpkg.Activation, err error) {
	var function activationpkg.Activation

	if function, err = activationpkg.FromName(s.Activation); err != nil {
		err = fmt.Errorf("model: layer %d activation load failed: %w", index, err)
		return nil, err
	}

	if activationLayer, err = layerpkg.NewActivation(function); err != nil {
		err = fmt.Errorf("model: layer %d activation construct failed: %w", index, err)
		return nil, err
	}

	return activationLayer, nil
}

func serializedBatchNormalizationLayer(index int, batchNormLayer *layerpkg.BatchNormalization) (serialized serializedLayer, err error) {
	var (
		gamma           serializedMatrix
		beta            serializedMatrix
		runningMean     serializedMatrix
		runningVariance serializedMatrix
	)

	if batchNormLayer == nil {
		err = fmt.Errorf("model: layer %d batch normalization layer is nil", index)
		return serialized, err
	}

	if batchNormLayer.Gamma() == nil {
		err = fmt.Errorf("model: layer %d batch normalization gamma parameter is nil", index)
		return serialized, err
	}

	if batchNormLayer.Beta() == nil {
		err = fmt.Errorf("model: layer %d batch normalization beta parameter is nil", index)
		return serialized, err
	}

	if gamma, err = serializedMatrixFromMatrix(batchNormLayer.Gamma().Values()); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization gamma serialize failed: %w", index, err)
		return serialized, err
	}

	if beta, err = serializedMatrixFromMatrix(batchNormLayer.Beta().Values()); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization beta serialize failed: %w", index, err)
		return serialized, err
	}

	if runningMean, err = serializedMatrixFromMatrix(batchNormLayer.RunningMean()); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization running mean serialize failed: %w", index, err)
		return serialized, err
	}

	if runningVariance, err = serializedMatrixFromMatrix(batchNormLayer.RunningVariance()); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization running variance serialize failed: %w", index, err)
		return serialized, err
	}

	serialized = serializedLayer{
		Type:            serializationLayerBatchNormalization,
		FeatureSize:     batchNormLayer.FeatureSize(),
		Momentum:        batchNormLayer.Momentum(),
		Epsilon:         batchNormLayer.Epsilon(),
		Gamma:           &gamma,
		Beta:            &beta,
		RunningMean:     &runningMean,
		RunningVariance: &runningVariance,
	}
	return serialized, nil
}

func (s serializedLayer) batchNormalizationLayer(index int) (batchNormLayer *layerpkg.BatchNormalization, err error) {
	var (
		gamma           *matrixpkg.Matrix
		beta            *matrixpkg.Matrix
		runningMean     *matrixpkg.Matrix
		runningVariance *matrixpkg.Matrix
	)

	if s.Gamma == nil {
		err = fmt.Errorf("model: layer %d batch normalization gamma is missing", index)
		return nil, err
	}

	if s.Beta == nil {
		err = fmt.Errorf("model: layer %d batch normalization beta is missing", index)
		return nil, err
	}

	if s.RunningMean == nil {
		err = fmt.Errorf("model: layer %d batch normalization running mean is missing", index)
		return nil, err
	}

	if s.RunningVariance == nil {
		err = fmt.Errorf("model: layer %d batch normalization running variance is missing", index)
		return nil, err
	}

	if gamma, err = s.Gamma.matrix(); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization gamma load failed: %w", index, err)
		return nil, err
	}

	if beta, err = s.Beta.matrix(); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization beta load failed: %w", index, err)
		return nil, err
	}

	if runningMean, err = s.RunningMean.matrix(); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization running mean load failed: %w", index, err)
		return nil, err
	}

	if runningVariance, err = s.RunningVariance.matrix(); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization running variance load failed: %w", index, err)
		return nil, err
	}

	if batchNormLayer, err = layerpkg.NewBatchNormalizationWithConfig(s.FeatureSize, s.Momentum, s.Epsilon); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization construct failed: %w", index, err)
		return nil, err
	}

	if err = batchNormLayer.Gamma().Values().CopyFrom(gamma); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization gamma copy failed: %w", index, err)
		return nil, err
	}

	if err = batchNormLayer.Beta().Values().CopyFrom(beta); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization beta copy failed: %w", index, err)
		return nil, err
	}

	if err = batchNormLayer.RunningMean().CopyFrom(runningMean); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization running mean copy failed: %w", index, err)
		return nil, err
	}

	if err = batchNormLayer.RunningVariance().CopyFrom(runningVariance); err != nil {
		err = fmt.Errorf("model: layer %d batch normalization running variance copy failed: %w", index, err)
		return nil, err
	}

	return batchNormLayer, nil
}

func serializedDenseLayer(index int, denseLayer *layerpkg.Dense) (serialized serializedLayer, err error) {
	var (
		weights serializedMatrix
		biases  serializedMatrix
	)

	if denseLayer == nil {
		err = fmt.Errorf("model: layer %d dense layer is nil", index)
		return serialized, err
	}

	if denseLayer.Weights() == nil {
		err = fmt.Errorf("model: layer %d dense weights parameter is nil", index)
		return serialized, err
	}

	if denseLayer.Biases() == nil {
		err = fmt.Errorf("model: layer %d dense biases parameter is nil", index)
		return serialized, err
	}

	if weights, err = serializedMatrixFromMatrix(denseLayer.Weights().Values()); err != nil {
		err = fmt.Errorf("model: layer %d dense weights serialize failed: %w", index, err)
		return serialized, err
	}

	if biases, err = serializedMatrixFromMatrix(denseLayer.Biases().Values()); err != nil {
		err = fmt.Errorf("model: layer %d dense biases serialize failed: %w", index, err)
		return serialized, err
	}

	serialized = serializedLayer{
		Type:       serializationLayerDense,
		InputSize:  denseLayer.InputSize(),
		OutputSize: denseLayer.OutputSize(),
		Weights:    &weights,
		Biases:     &biases,
	}
	return serialized, nil
}

func (s serializedLayer) denseLayer(index int) (denseLayer *layerpkg.Dense, err error) {
	var (
		weights *matrixpkg.Matrix
		biases  *matrixpkg.Matrix
	)

	if s.Weights == nil {
		err = fmt.Errorf("model: layer %d dense weights are missing", index)
		return nil, err
	}

	if s.Biases == nil {
		err = fmt.Errorf("model: layer %d dense biases are missing", index)
		return nil, err
	}

	if weights, err = s.Weights.matrix(); err != nil {
		err = fmt.Errorf("model: layer %d dense weights load failed: %w", index, err)
		return nil, err
	}

	if biases, err = s.Biases.matrix(); err != nil {
		err = fmt.Errorf("model: layer %d dense biases load failed: %w", index, err)
		return nil, err
	}

	if denseLayer, err = layerpkg.NewDense(s.InputSize, s.OutputSize, func(inputSize, outputSize int) (initialized *matrixpkg.Matrix, err error) {
		initialized = weights
		return initialized, nil
	}); err != nil {
		err = fmt.Errorf("model: layer %d dense construct failed: %w", index, err)
		return nil, err
	}

	if err = denseLayer.Biases().Values().CopyFrom(biases); err != nil {
		err = fmt.Errorf("model: layer %d dense biases copy failed: %w", index, err)
		return nil, err
	}

	return denseLayer, nil
}

func serializedDropoutLayer(index int, dropoutLayer *layerpkg.Dropout) (serialized serializedLayer, err error) {
	if dropoutLayer == nil {
		err = fmt.Errorf("model: layer %d dropout layer is nil", index)
		return serialized, err
	}

	serialized = serializedLayer{
		Type: serializationLayerDropout,
		Rate: dropoutLayer.Rate(),
	}
	return serialized, nil
}

func (s serializedLayer) dropoutLayer(index int) (dropoutLayer *layerpkg.Dropout, err error) {
	var random *rand.Rand

	random = rand.New(rand.NewSource(serializationDropoutSeed + int64(index)))
	if dropoutLayer, err = layerpkg.NewDropout(s.Rate, random); err != nil {
		err = fmt.Errorf("model: layer %d dropout construct failed: %w", index, err)
		return nil, err
	}

	return dropoutLayer, nil
}

type serializedMatrix struct {
	Rows   int       `json:"rows"`
	Cols   int       `json:"cols"`
	Values []float32 `json:"values"`
}

func serializedMatrixFromMatrix(source *matrixpkg.Matrix) (serialized serializedMatrix, err error) {
	var values []float32

	if values, err = source.Values(); err != nil {
		return serialized, err
	}

	serialized = serializedMatrix{
		Rows:   source.Rows(),
		Cols:   source.Cols(),
		Values: values,
	}
	return serialized, nil
}

func (s serializedMatrix) matrix() (m *matrixpkg.Matrix, err error) {
	m, err = matrixpkg.FromSlice(s.Rows, s.Cols, s.Values)
	return m, err
}

func encodeSequential(writer io.Writer, s *Sequential) (err error) {
	var (
		document sequentialDocument
		encoder  *json.Encoder
	)

	if document, err = sequentialDocumentFromModel(s); err != nil {
		return err
	}

	encoder = json.NewEncoder(writer)
	encoder.SetIndent("", "  ")
	if err = encoder.Encode(document); err != nil {
		err = fmt.Errorf("encode JSON: %w", err)
		return err
	}

	return nil
}

func decodeSequential(reader io.Reader) (s *Sequential, err error) {
	var (
		document sequentialDocument
		decoder  *json.Decoder
	)

	decoder = json.NewDecoder(reader)
	if err = decoder.Decode(&document); err != nil {
		err = fmt.Errorf("decode JSON: %w", err)
		return nil, err
	}

	if err = ensureNoTrailingJSON(decoder); err != nil {
		return nil, err
	}

	s, err = document.model()
	return s, err
}

func ensureNoTrailingJSON(decoder *json.Decoder) (err error) {
	var extra interface{}

	if err = decoder.Decode(&extra); err == nil {
		err = errors.New("model: JSON contains multiple values")
		return err
	}

	if errors.Is(err, io.EOF) {
		return nil
	}

	err = fmt.Errorf("decode trailing JSON: %w", err)
	return err
}
