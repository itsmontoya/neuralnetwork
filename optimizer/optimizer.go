// Package optimizer defines parameter update contracts and implementations.
package optimizer

// Optimizer updates trainable parameters using their accumulated gradients.
//
// Optimizer implementations reset parameter gradients after a successful
// update.
type Optimizer interface {
	// Update applies one optimization step to parameters.
	Update(parameters []*Parameter) (err error)
	// LearningRate returns the optimizer learning rate.
	LearningRate() (learningRate float32)
	// SetLearningRate updates the optimizer learning rate.
	SetLearningRate(learningRate float32) (err error)
}
