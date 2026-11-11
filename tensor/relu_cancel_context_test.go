package tensor_test

import "context"

func newReLUCancelContext(cancelAfter int) (out *reluCancelContext) {
	var c reluCancelContext

	c.Context, c.cancel = context.WithCancel(context.Background())
	c.cancelAfter = cancelAfter
	return &c
}

type reluCancelContext struct {
	context.Context
	cancel      context.CancelFunc
	checks      int
	cancelAfter int
}

func (c *reluCancelContext) Err() (err error) {
	c.checks++
	if c.checks == c.cancelAfter {
		c.cancel()
	}
	return c.Context.Err()
}
