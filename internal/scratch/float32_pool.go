package scratch

import "fmt"

// Float32Pool retains float32 buffers for up to four exact lengths in
// least-recently-used order. Its zero value is ready for use.
//
// Returned buffers are dirty. A Float32Pool must serve only one logical
// scratch role, and callers must not request another buffer while a previously
// returned buffer from the same pool is still a live operand. A Float32Pool
// must not be copied after first use.
type Float32Pool struct {
	entries [poolEntryLimit][]float32
	count   int
}

// Get returns a dirty buffer with the requested exact length. Reused reports
// whether the buffer was already retained by the pool.
func (p *Float32Pool) Get(length int) (out []float32, reused bool, err error) {
	if length < 0 {
		err = fmt.Errorf("scratch: length must be non-negative: length=%d", length)
		return nil, false, err
	}

	var index int
	for index = 0; index < p.count; index++ {
		out = p.entries[index]
		if len(out) == length {
			p.moveToMostRecent(index)
			return out, true, nil
		}
	}

	out = make([]float32, length)
	p.retain(out)
	return out, false, nil
}

func (p *Float32Pool) moveToMostRecent(index int) {
	var entry []float32
	entry = p.entries[index]
	copy(p.entries[index:p.count-1], p.entries[index+1:p.count])
	p.entries[p.count-1] = entry
}

func (p *Float32Pool) retain(entry []float32) {
	if p.count < poolEntryLimit {
		p.entries[p.count] = entry
		p.count++
		return
	}

	copy(p.entries[:poolEntryLimit-1], p.entries[1:])
	p.entries[poolEntryLimit-1] = entry
}
