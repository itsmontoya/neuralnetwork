package scratch

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

const poolEntryLimit = 4

// MatrixPool retains matrices for up to four exact shapes in
// least-recently-used order. Its zero value is ready for use.
//
// Returned matrices are dirty. A MatrixPool must serve only one logical
// scratch role, and callers must not request another matrix while a previously
// returned matrix from the same pool is still a live operand. A MatrixPool
// must not be copied after first use.
type MatrixPool struct {
	entries [poolEntryLimit]*matrix.Matrix
	count   int
}

// Release detaches retained device storage and empties the pool.
func (p *MatrixPool) Release() (err error) {
	var (
		entry      *matrix.Matrix
		releaseErr error
		index      int
	)

	if p == nil {
		return nil
	}

	for index = 0; index < p.count; index++ {
		entry = p.entries[index]
		if entry == nil {
			continue
		}
		if releaseErr = entry.CopyFrom(entry); releaseErr != nil {
			releaseErr = fmt.Errorf("scratch: release matrix %d: %w", index, releaseErr)
			err = errors.Join(err, releaseErr)
		}
	}

	clear(p.entries[:p.count])
	p.count = 0
	return err
}

// Get returns a dirty matrix with the requested exact shape. Reused reports
// whether the matrix was already retained by the pool.
func (p *MatrixPool) Get(rows, cols int) (out *matrix.Matrix, reused bool, err error) {
	var index int
	for index = 0; index < p.count; index++ {
		out = p.entries[index]
		if out.Rows() == rows && out.Cols() == cols {
			p.moveToMostRecent(index)
			return out, true, nil
		}
	}

	if out, err = matrix.New(rows, cols); err != nil {
		return nil, false, err
	}

	if err = p.retain(out); err != nil {
		return nil, false, err
	}
	return out, false, nil
}

func (p *MatrixPool) moveToMostRecent(index int) {
	var entry *matrix.Matrix
	entry = p.entries[index]
	copy(p.entries[index:p.count-1], p.entries[index+1:p.count])
	p.entries[p.count-1] = entry
}

func (p *MatrixPool) retain(entry *matrix.Matrix) (err error) {
	if p.count < poolEntryLimit {
		p.entries[p.count] = entry
		p.count++
		return nil
	}

	// A self-copy preserves dirty values while detaching private device storage.
	if err = p.entries[0].CopyFrom(p.entries[0]); err != nil {
		return err
	}
	copy(p.entries[:poolEntryLimit-1], p.entries[1:])
	p.entries[poolEntryLimit-1] = entry
	return nil
}
