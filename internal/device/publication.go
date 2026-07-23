package device

// Publication owns the success and failure transitions for one encoded write.
type Publication struct {
	Publish func() (err error)
	Discard func(cause error) (err error)
}

func (p Publication) validate() (err error) {
	if p.Publish == nil || p.Discard == nil {
		err = ErrInvalidPublication
		return err
	}

	return nil
}
