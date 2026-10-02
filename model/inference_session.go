package model

import (
	"errors"
	"fmt"
	"sync"

	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/matrix"
)

var (
	// ErrInferenceSessionsActive reports an operation that is prohibited while
	// one or more sessions share model parameters.
	ErrInferenceSessionsActive = errors.New("model: inference sessions are active")
	// ErrSequentialInUse reports a concurrent operation on a Sequential model.
	ErrSequentialInUse = errors.New("model: sequential model is in use")
	// ErrInferenceSessionClosed reports use after session closure.
	ErrInferenceSessionClosed = errors.New("model: inference session is closed")
	// ErrInferenceSessionInUse reports concurrent use or closure of one session.
	ErrInferenceSessionInUse = errors.New("model: inference session is in use")
)

func newInferenceSession(owner *Sequential) (out *InferenceSession, err error) {
	var (
		index        int
		current      layer.Layer
		cloned       layer.Layer
		clonedLayers []layer.Layer
		sessionModel *Sequential
		s            InferenceSession
	)

	if err = owner.beginInferenceSessionCreation(); err != nil {
		return nil, err
	}
	defer owner.endOwnerUse()
	if err = owner.validateReady(); err != nil {
		return nil, err
	}

	clonedLayers = make([]layer.Layer, 0, len(owner.layers))
	for index, current = range owner.layers {
		if cloned, err = layer.CloneForInference(current); err != nil {
			err = fmt.Errorf("model: inference session layer %d clone failed: %w", index, err)
			return nil, err
		}
		clonedLayers = append(clonedLayers, cloned)
	}
	if sessionModel, err = NewSequential(clonedLayers...); err != nil {
		err = fmt.Errorf("model: inference session model construct failed: %w", err)
		return nil, err
	}
	if err = sessionModel.SetTraining(false); err != nil {
		err = fmt.Errorf("model: inference session evaluation mode failed: %w", err)
		return nil, err
	}

	s.owner = owner
	s.model = sessionModel
	owner.registerInferenceSession()
	return &s, nil
}

// InferenceSession owns mutable inference state while sharing immutable model
// parameters with its parent Sequential model.
//
// A session is single-goroutine: concurrent calls on the same session return
// ErrInferenceSessionInUse. Different sessions may predict concurrently.
// Predict returns session-owned output that remains valid until the session's
// next prediction or Close. Use PredictInto when results must be retained.
// Close is idempotent.
type InferenceSession struct {
	mutex  sync.Mutex
	owner  *Sequential
	model  *Sequential
	closed bool
	inUse  bool
}

// Predict runs an evaluation forward pass.
func (s *InferenceSession) Predict(input *matrix.Matrix) (output *matrix.Matrix, err error) {
	var sessionModel *Sequential

	if sessionModel, err = s.beginUse(); err != nil {
		return nil, err
	}
	defer s.endUse()

	output, err = sessionModel.Predict(input)
	return output, err
}

// PredictInto copies an evaluation prediction into caller-owned destination.
func (s *InferenceSession) PredictInto(input, destination *matrix.Matrix) (err error) {
	var (
		sessionModel *Sequential
		output       *matrix.Matrix
	)

	if sessionModel, err = s.beginUse(); err != nil {
		return err
	}
	defer s.endUse()

	if output, err = sessionModel.Predict(input); err != nil {
		return err
	}
	if err = destination.CopyFrom(output); err != nil {
		err = fmt.Errorf("model: inference session copy prediction: %w", err)
		return err
	}

	return nil
}

// PredictWithLengths runs an evaluation forward pass with logical sequence lengths.
func (s *InferenceSession) PredictWithLengths(
	input *matrix.Matrix,
	lengths *data.SequenceLengths,
) (output *matrix.Matrix, err error) {
	var sessionModel *Sequential

	if sessionModel, err = s.beginUse(); err != nil {
		return nil, err
	}
	defer s.endUse()

	output, err = sessionModel.PredictWithLengths(input, lengths)
	return output, err
}

// Close releases session-owned host and device references.
func (s *InferenceSession) Close() (err error) {
	var owner *Sequential

	if s == nil {
		return nil
	}
	s.mutex.Lock()
	if s.closed {
		s.mutex.Unlock()
		return nil
	}
	if s.inUse {
		s.mutex.Unlock()
		return ErrInferenceSessionInUse
	}

	owner = s.owner
	s.owner = nil
	s.model = nil
	s.closed = true
	s.mutex.Unlock()
	if owner != nil {
		owner.releaseInferenceSession()
	}

	return nil
}

func (s *InferenceSession) beginUse() (sessionModel *Sequential, err error) {
	if s == nil {
		return nil, ErrInferenceSessionClosed
	}

	s.mutex.Lock()
	defer s.mutex.Unlock()
	if s.closed || s.model == nil {
		return nil, ErrInferenceSessionClosed
	}
	if s.inUse {
		return nil, ErrInferenceSessionInUse
	}

	s.inUse = true
	sessionModel = s.model
	return sessionModel, nil
}

func (s *InferenceSession) endUse() {
	if s == nil {
		return
	}

	s.mutex.Lock()
	s.inUse = false
	s.mutex.Unlock()
}
