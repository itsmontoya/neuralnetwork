package data

import "fmt"

// NewSequenceLengths constructs validated logical sequence lengths by copying values.
func NewSequenceLengths(steps int, values []int) (out *SequenceLengths, err error) {
	var ownedValues []int

	if err = validateSequenceLengthValues(steps, values); err != nil {
		return nil, err
	}

	ownedValues = make([]int, len(values))
	copy(ownedValues, values)
	out, err = newSequenceLengths(steps, ownedValues)
	return out, err
}

// newSequenceLengths stores values that are already owned by the data package.
func newSequenceLengths(steps int, values []int) (out *SequenceLengths, err error) {
	var lengths SequenceLengths

	if err = validateSequenceLengthValues(steps, values); err != nil {
		return nil, err
	}

	lengths.steps = steps
	lengths.values = values
	return &lengths, nil
}

// SequenceLengths owns one positive logical length per padded sequence row.
type SequenceLengths struct {
	steps  int
	values []int
}

// Validate reports whether the sequence lengths contain valid owned values.
func (l *SequenceLengths) Validate() (err error) {
	err = l.validate()
	return err
}

// Steps returns the physical step count shared by every sequence.
func (l *SequenceLengths) Steps() (steps int) {
	if l == nil {
		return 0
	}

	steps = l.steps
	return steps
}

// SampleCount returns the number of logical lengths.
func (l *SequenceLengths) SampleCount() (samples int) {
	if l == nil {
		return 0
	}

	samples = len(l.values)
	return samples
}

// Values returns a caller-owned copy of the logical lengths.
func (l *SequenceLengths) Values() (values []int, err error) {
	if err = l.validate(); err != nil {
		return nil, err
	}

	values = make([]int, len(l.values))
	copy(values, l.values)
	return values, nil
}

// ValuesInto copies every logical length into destination.
//
// The destination must match SampleCount. A valid call fully overwrites the
// caller-owned destination without allocating or retaining it.
func (l *SequenceLengths) ValuesInto(destination []int) (err error) {
	if err = l.validate(); err != nil {
		return err
	}

	if len(destination) != len(l.values) {
		err = fmt.Errorf(
			"data: sequence lengths destination length mismatch: got %d, want %d",
			len(destination),
			len(l.values),
		)
		return err
	}

	copy(destination, l.values)
	return nil
}

// SelectRowsInto copies logical lengths in index order into destination.
//
// Repeated indexes duplicate values. The destination must have len(indexes)
// elements. A valid call allocates no storage and retains neither argument.
func (l *SequenceLengths) SelectRowsInto(indexes, destination []int) (err error) {
	var (
		position  int
		sourceRow int
	)

	if err = l.validate(); err != nil {
		return err
	}

	if len(indexes) == 0 {
		err = fmt.Errorf("data: sequence lengths row indexes are empty")
		return err
	}

	for position, sourceRow = range indexes {
		if sourceRow < 0 || sourceRow >= len(l.values) {
			err = fmt.Errorf(
				"data: sequence lengths row index out of range: index=%d row=%d samples=%d",
				position,
				sourceRow,
				len(l.values),
			)
			return err
		}
	}

	if len(destination) != len(indexes) {
		err = fmt.Errorf(
			"data: sequence lengths selected destination length mismatch: got %d, want %d",
			len(destination),
			len(indexes),
		)
		return err
	}

	for position, sourceRow = range indexes {
		destination[position] = l.values[sourceRow]
	}

	return nil
}

func (l *SequenceLengths) selectRows(indexes []int) (selected *SequenceLengths, err error) {
	var values []int

	values = make([]int, len(indexes))
	if err = l.SelectRowsInto(indexes, values); err != nil {
		return nil, err
	}

	selected, err = newSequenceLengths(l.steps, values)
	return selected, err
}

func (l *SequenceLengths) validate() (err error) {
	if l == nil {
		err = fmt.Errorf("data: sequence lengths are nil")
		return err
	}

	err = validateSequenceLengthValues(l.steps, l.values)
	return err
}

func validateSequenceLengthValues(steps int, values []int) (err error) {
	var (
		row   int
		value int
	)

	if steps <= 0 {
		err = fmt.Errorf("data: sequence lengths steps must be positive: steps=%d", steps)
		return err
	}

	if len(values) == 0 {
		err = fmt.Errorf("data: sequence lengths values are empty")
		return err
	}

	for row, value = range values {
		if value < 1 || value > steps {
			err = fmt.Errorf(
				"data: sequence lengths value out of range: row=%d value=%d steps=%d",
				row,
				value,
				steps,
			)
			return err
		}
	}

	return nil
}
