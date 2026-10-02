package model

import (
	"errors"
	"testing"
)

func Test_InferenceSession_InUseLifecycle(t *testing.T) {
	var (
		session *InferenceSession
		err     error
	)

	session = &InferenceSession{model: &Sequential{}, inUse: true}
	if err = session.Close(); !errors.Is(err, ErrInferenceSessionInUse) {
		t.Fatalf("Close error = %v, want ErrInferenceSessionInUse", err)
	}
	if _, err = session.beginUse(); !errors.Is(err, ErrInferenceSessionInUse) {
		t.Fatalf("beginUse error = %v, want ErrInferenceSessionInUse", err)
	}
	session.inUse = false
	if err = session.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
}
