package gpu

// KVAppend is a staged update created by Scope.AppendKV. Copies of this handle
// refer to the same update. Resolve it with Publish or Close.
type KVAppend struct {
	state *kvAppendState
}

type kvAppendState struct {
	cache     *kvCacheState
	scope     *scopeState
	length    int64
	discarded bool
}

// Publish exposes the staged rows to subsequent recordings. It requires a
// successful Wait on the recording scope and returns ErrBusy before that wait.
// After an unsuccessful submission, discarded recording or closed update,
// publication returns ErrInvalid. Call Publish only after validating the
// enclosing sequence step. Publication submits no GPU work.
func (u *KVAppend) Publish() (err error) {
	if u == nil || u.state == nil {
		return ErrClosed
	}
	var (
		update = u.state
		cache  = update.cache
	)
	cache.device.mutex.Lock()
	defer cache.device.mutex.Unlock()
	if cache.closed {
		return ErrClosed
	}
	if update.discarded || cache.pending != update {
		return ErrInvalid
	}
	if update.scope != nil {
		return ErrBusy
	}
	cache.length = update.length
	cache.pending = nil
	return nil
}

// Close abandons the update and leaves the committed history intact. Close
// does not wait. The cache remains busy until the recording scope resolves.
func (u *KVAppend) Close() (err error) {
	if u == nil || u.state == nil {
		return nil
	}
	var (
		update = u.state
		cache  = update.cache
	)
	cache.device.mutex.Lock()
	defer cache.device.mutex.Unlock()
	if cache.pending == update {
		update.discarded = true
		if update.scope == nil {
			cache.pending = nil
		}
	}
	return nil
}

func (u *kvAppendState) finish(ready bool) {
	u.scope = nil
	if u.cache.length > 0 {
		u.cache.slot.state.storage.initialized = true
	}
	if (!ready || u.discarded) && u.cache.pending == u {
		u.cache.pending = nil
	}
}
