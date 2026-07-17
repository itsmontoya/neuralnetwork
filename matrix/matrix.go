// Package matrix provides dense row-major float32 matrix primitives.
package matrix

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

const columnReductionAddMinCols = 16

// New constructs a zero-filled Matrix with the provided shape.
func New(rows, cols int) (out *Matrix, err error) {
	var (
		size int
		m    Matrix
	)

	m.rows = rows
	m.cols = cols
	if size, err = m.matrixSize(); err != nil {
		return nil, err
	}

	m.data = make([]float32, size)
	return &m, nil
}

// FromSlice constructs a Matrix by copying values in row-major order.
func FromSlice(rows, cols int, values []float32) (out *Matrix, err error) {
	var (
		size int
		m    Matrix
	)

	m.rows = rows
	m.cols = cols
	if size, err = m.matrixSize(); err != nil {
		return nil, err
	}

	if len(values) != size {
		err = fmt.Errorf("matrix: values length mismatch: got %d, want %d", len(values), size)
		return nil, err
	}

	m.data = make([]float32, size)
	copy(m.data, values)
	return &m, nil
}

// NewRandom constructs a Matrix filled with values from random.Float64.
func NewRandom(rows, cols int, random *rand.Rand) (out *Matrix, err error) {
	out, err = NewUniform(rows, cols, 0, 1, random)
	return out, err
}

// NewUniform constructs a Matrix filled from a uniform distribution in [min, max).
func NewUniform(rows, cols int, min, max float32, random *rand.Rand) (out *Matrix, err error) {
	if random == nil {
		err = errors.New("matrix: random source is nil")
		return nil, err
	}

	if max < min {
		err = fmt.Errorf("matrix: uniform max must be greater than or equal to min: min=%g max=%g", min, max)
		return nil, err
	}

	if out, err = New(rows, cols); err != nil {
		return nil, err
	}

	var (
		index int
		span  float32
	)

	span = max - min
	for index = range out.data {
		out.data[index] = min + span*float32(random.Float64())
	}

	return out, nil
}

// NewNormal constructs a Matrix filled from a normal distribution with mean and stddev.
func NewNormal(rows, cols int, mean, stddev float32, random *rand.Rand) (out *Matrix, err error) {
	if random == nil {
		err = errors.New("matrix: random source is nil")
		return nil, err
	}

	if stddev < 0 {
		err = fmt.Errorf("matrix: normal standard deviation must be non-negative: stddev=%g", stddev)
		return nil, err
	}

	if out, err = New(rows, cols); err != nil {
		return nil, err
	}

	var index int
	for index = range out.data {
		out.data[index] = mean + stddev*float32(random.NormFloat64())
	}

	return out, nil
}

// NewXavierUniform constructs a fanIn by fanOut Matrix using Xavier/Glorot uniform initialization.
// Values are sampled from [-sqrt(6/(fanIn+fanOut)), sqrt(6/(fanIn+fanOut))).
func NewXavierUniform(fanIn, fanOut int, random *rand.Rand) (out *Matrix, err error) {
	var shape Matrix
	var limit float32

	shape.rows = fanIn
	shape.cols = fanOut
	if _, err = shape.matrixSize(); err != nil {
		return nil, err
	}

	limit = float32(math.Sqrt(float64(6 / float32(fanIn+fanOut))))
	out, err = NewUniform(fanIn, fanOut, -limit, limit, random)
	return out, err
}

// NewHeNormal constructs a fanIn by fanOut Matrix using He normal initialization.
// Values are sampled from a normal distribution with mean 0 and stddev sqrt(2/fanIn).
func NewHeNormal(fanIn, fanOut int, random *rand.Rand) (out *Matrix, err error) {
	var shape Matrix
	var stddev float32

	shape.rows = fanIn
	shape.cols = fanOut
	if _, err = shape.matrixSize(); err != nil {
		return nil, err
	}

	stddev = float32(math.Sqrt(float64(2 / float32(fanIn))))
	out, err = NewNormal(fanIn, fanOut, 0, stddev, random)
	return out, err
}

// Matrix stores dense float32 values in row-major order.
type Matrix struct {
	rows int
	cols int
	data []float32
}

// Rows returns the matrix row count.
func (m *Matrix) Rows() (rows int) {
	if m == nil {
		return 0
	}

	rows = m.rows
	return rows
}

// Cols returns the matrix column count.
func (m *Matrix) Cols() (cols int) {
	if m == nil {
		return 0
	}

	cols = m.cols
	return cols
}

// Shape returns the matrix row and column counts.
func (m *Matrix) Shape() (rows, cols int) {
	if m == nil {
		return 0, 0
	}

	rows = m.rows
	cols = m.cols
	return rows, cols
}

// Validate reports whether the matrix has a valid shape and storage length.
func (m *Matrix) Validate() (err error) {
	err = m.validate()
	return err
}

// Values returns a copy of the row-major matrix values.
func (m *Matrix) Values() (values []float32, err error) {
	if err = m.validate(); err != nil {
		return nil, err
	}

	values = make([]float32, len(m.data))
	copy(values, m.data)
	return values, nil
}

// ValuesInto copies row-major matrix values into destination.
//
// The destination length must match the matrix storage length. Values are
// copied, so later destination mutations do not affect m.
func (m *Matrix) ValuesInto(destination []float32) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if len(destination) != len(m.data) {
		err = fmt.Errorf("matrix: destination length mismatch: got %d, want %d", len(destination), len(m.data))
		return err
	}

	copy(destination, m.data)
	return nil
}

// At returns the value at row and col.
func (m *Matrix) At(row, col int) (value float32, err error) {
	if err = m.validate(); err != nil {
		return 0, err
	}

	if err = m.validateIndex(row, col); err != nil {
		return 0, err
	}

	value = m.data[m.index(row, col)]
	return value, nil
}

// Set updates the value at row and col.
func (m *Matrix) Set(row, col int, value float32) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if err = m.validateIndex(row, col); err != nil {
		return err
	}

	m.data[m.index(row, col)] = value
	return nil
}

// Fill sets every matrix value to value.
func (m *Matrix) Fill(value float32) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	var index int
	for index = range m.data {
		m.data[index] = value
	}

	return nil
}

// Clone returns a deep copy of the matrix.
func (m *Matrix) Clone() (clone *Matrix, err error) {
	if err = m.validate(); err != nil {
		return nil, err
	}

	clone = m.newLike()
	copy(clone.data, m.data)
	return clone, nil
}

// CopyFrom copies all values from source into m.
func (m *Matrix) CopyFrom(source *Matrix) (err error) {
	if err = m.sameShape(source); err != nil {
		return err
	}

	copy(m.data, source.data)
	return nil
}

// CopyValuesFrom copies row-major values into m.
//
// The values length must match the matrix storage length. Values are copied, so
// later source-slice mutations do not affect m.
func (m *Matrix) CopyValuesFrom(values []float32) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if len(values) != len(m.data) {
		err = fmt.Errorf("matrix: values length mismatch: got %d, want %d", len(values), len(m.data))
		return err
	}

	copy(m.data, values)
	return nil
}

// SelectRows returns a copy of the rows identified by indexes.
//
// Rows are copied in index order, and repeated indexes duplicate rows in the
// returned matrix. The returned matrix owns its storage.
func (m *Matrix) SelectRows(indexes []int) (result *Matrix, err error) {
	var (
		next        Matrix
		outputRow   int
		sourceRow   int
		sourceStart int
		resultStart int
	)

	if err = m.validate(); err != nil {
		return nil, err
	}

	if len(indexes) == 0 {
		err = errors.New("matrix: row indexes are empty")
		return nil, err
	}

	for _, sourceRow = range indexes {
		if sourceRow < 0 || sourceRow >= m.rows {
			err = fmt.Errorf("matrix: row index out of range: row=%d rows=%d", sourceRow, m.rows)
			return nil, err
		}
	}

	next.rows = len(indexes)
	next.cols = m.cols
	next.data = make([]float32, len(indexes)*m.cols)
	result = &next

	for outputRow, sourceRow = range indexes {
		sourceStart = sourceRow * m.cols
		resultStart = outputRow * m.cols
		copy(result.data[resultStart:resultStart+m.cols], m.data[sourceStart:sourceStart+m.cols])
	}

	return result, nil
}

// Add returns the elementwise sum of m and other.
func (m *Matrix) Add(other *Matrix) (result *Matrix, err error) {
	if err = m.sameShape(other); err != nil {
		return nil, err
	}

	result = m.newLike()

	addInto(m.data, other.data, result.data)
	return result, nil
}

// AddInto writes the elementwise sum of m and other into result.
//
// The destination must match the input shape. The destination is caller-owned
// and may alias either input because each element is read before it is written.
func (m *Matrix) AddInto(other, result *Matrix) (err error) {
	if err = m.sameShape(other); err != nil {
		return err
	}

	if err = result.requireShape("destination", m.rows, m.cols); err != nil {
		return err
	}

	addInto(m.data, other.data, result.data)
	return nil
}

// AddInPlace adds other to m elementwise.
func (m *Matrix) AddInPlace(other *Matrix) (err error) {
	err = m.AddInto(other, m)
	return err
}

// AddScaledInPlace adds scale*other to m elementwise.
//
// The receiver is updated in place. The other matrix is read but not retained.
func (m *Matrix) AddScaledInPlace(other *Matrix, scale float32) (err error) {
	if err = m.sameShape(other); err != nil {
		return err
	}

	addScaledInPlace(m.data, other.data, scale)
	return nil
}

// AddMappedInPlace adds fn applied to each element of other to m.
//
// The receiver is updated in place. The other matrix may alias the receiver
// because each element is read before it is written. Neither argument is
// retained.
func (m *Matrix) AddMappedInPlace(other *Matrix, fn func(float32) float32) (err error) {
	if fn == nil {
		err = errors.New("matrix: add mapped function is nil")
		return err
	}

	if err = m.sameShape(other); err != nil {
		return err
	}

	var index int
	for index = range m.data {
		m.data[index] += fn(other.data[index])
	}

	return nil
}

// AdamUpdateInPlace applies one Adam optimizer update to m.
//
// The gradient, firstMoment, and secondMoment matrices must match m's shape and
// must not alias each other or m. Moment matrices are updated in place before m
// is updated. Correction values are the precomputed bias-correction
// denominators for the current Adam step and must be nonzero. No matrix storage
// is exposed or retained.
func (m *Matrix) AdamUpdateInPlace(
	gradient *Matrix,
	firstMoment *Matrix,
	secondMoment *Matrix,
	learningRate float32,
	beta1 float32,
	beta2 float32,
	epsilon float32,
	firstCorrection float32,
	secondCorrection float32,
) (err error) {
	var (
		gradientValue  float32
		firstEstimate  float32
		secondEstimate float32
		index          int
	)

	if err = m.sameShape(gradient); err != nil {
		return err
	}

	if err = firstMoment.requireShape("first moment", m.rows, m.cols); err != nil {
		return err
	}

	if err = secondMoment.requireShape("second moment", m.rows, m.cols); err != nil {
		return err
	}

	if m == gradient || m == firstMoment || m == secondMoment ||
		gradient == firstMoment || gradient == secondMoment || firstMoment == secondMoment {
		err = errors.New("matrix: adam update matrices must not alias")
		return err
	}

	if firstCorrection == 0 {
		err = errors.New("matrix: adam first correction must be nonzero")
		return err
	}

	if secondCorrection == 0 {
		err = errors.New("matrix: adam second correction must be nonzero")
		return err
	}

	for index = range m.data {
		gradientValue = gradient.data[index]
		firstMoment.data[index] = beta1*firstMoment.data[index] + (1-beta1)*gradientValue
		secondMoment.data[index] = beta2*secondMoment.data[index] + (1-beta2)*gradientValue*gradientValue

		firstEstimate = firstMoment.data[index] / firstCorrection
		secondEstimate = secondMoment.data[index] / secondCorrection
		m.data[index] -= learningRate * firstEstimate / (float32(math.Sqrt(float64(secondEstimate))) + epsilon)
	}

	return nil
}

// MultiplyScalarInPlace multiplies every element of m by value.
//
// The receiver is updated in place and keeps owning its storage.
func (m *Matrix) MultiplyScalarInPlace(value float32) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	multiplyScalarInPlace(m.data, value)
	return nil
}

// Subtract returns the elementwise difference of m and other.
func (m *Matrix) Subtract(other *Matrix) (result *Matrix, err error) {
	if err = m.sameShape(other); err != nil {
		return nil, err
	}

	result = m.newLike()

	subtractInto(m.data, other.data, result.data)
	return result, nil
}

// SubtractInto writes the elementwise difference of m and other into result.
//
// The destination must match the input shape. The destination is caller-owned
// and may alias either input because each element is read before it is written.
func (m *Matrix) SubtractInto(other, result *Matrix) (err error) {
	if err = m.sameShape(other); err != nil {
		return err
	}

	if err = result.requireShape("destination", m.rows, m.cols); err != nil {
		return err
	}

	subtractInto(m.data, other.data, result.data)
	return nil
}

// MultiplyElements returns the elementwise product of m and other.
func (m *Matrix) MultiplyElements(other *Matrix) (result *Matrix, err error) {
	if err = m.sameShape(other); err != nil {
		return nil, err
	}

	result = m.newLike()

	multiplyElementsInto(m.data, other.data, result.data)
	return result, nil
}

// MultiplyElementsInto writes the elementwise product of m and other into result.
//
// The destination must match the input shape. The destination is caller-owned
// and may alias either input because each element is read before it is written.
func (m *Matrix) MultiplyElementsInto(other, result *Matrix) (err error) {
	if err = m.sameShape(other); err != nil {
		return err
	}

	if err = result.requireShape("destination", m.rows, m.cols); err != nil {
		return err
	}

	multiplyElementsInto(m.data, other.data, result.data)
	return nil
}

// DivideElements returns the elementwise quotient of m and other.
func (m *Matrix) DivideElements(other *Matrix) (result *Matrix, err error) {
	if err = m.sameShape(other); err != nil {
		return nil, err
	}

	result = m.newLike()

	var index int
	for index = range result.data {
		if other.data[index] == 0 {
			err = fmt.Errorf("matrix: division by zero at row %d column %d", index/m.cols, index%m.cols)
			return nil, err
		}

		result.data[index] = m.data[index] / other.data[index]
	}

	return result, nil
}

// DivideElementsInto writes the elementwise quotient of m and other into result.
//
// The destination must match the input shape. The destination is caller-owned
// and may alias either input because each element is read before it is written.
func (m *Matrix) DivideElementsInto(other, result *Matrix) (err error) {
	if err = m.sameShape(other); err != nil {
		return err
	}

	if err = result.requireShape("destination", m.rows, m.cols); err != nil {
		return err
	}

	var index int
	for index = range result.data {
		if other.data[index] == 0 {
			err = fmt.Errorf("matrix: division by zero at row %d column %d", index/m.cols, index%m.cols)
			return err
		}

		result.data[index] = m.data[index] / other.data[index]
	}

	return nil
}

// AddScalar returns a matrix with value added to every element.
func (m *Matrix) AddScalar(value float32) (result *Matrix, err error) {
	if err = m.validate(); err != nil {
		return nil, err
	}

	result = m.newLike()

	addScalarInto(m.data, value, result.data)
	return result, nil
}

// AddScalarInto writes m plus value into result.
//
// The destination must match m's shape. The destination is caller-owned and may
// alias m because each element is read before it is written.
func (m *Matrix) AddScalarInto(value float32, result *Matrix) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if err = result.requireShape("destination", m.rows, m.cols); err != nil {
		return err
	}

	addScalarInto(m.data, value, result.data)
	return nil
}

// MultiplyScalar returns a matrix with every element multiplied by value.
func (m *Matrix) MultiplyScalar(value float32) (result *Matrix, err error) {
	if err = m.validate(); err != nil {
		return nil, err
	}

	result = m.newLike()

	multiplyScalarInto(m.data, value, result.data)
	return result, nil
}

// MultiplyScalarInto writes m multiplied by value into result.
//
// The destination must match m's shape. The destination is caller-owned and may
// alias m because each element is read before it is written.
func (m *Matrix) MultiplyScalarInto(value float32, result *Matrix) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if err = result.requireShape("destination", m.rows, m.cols); err != nil {
		return err
	}

	multiplyScalarInto(m.data, value, result.data)
	return nil
}

// DivideScalar returns a matrix with every element divided by value.
func (m *Matrix) DivideScalar(value float32) (result *Matrix, err error) {
	if err = m.validate(); err != nil {
		return nil, err
	}

	if value == 0 {
		err = errors.New("matrix: division by zero scalar")
		return nil, err
	}

	result = m.newLike()

	var index int
	for index = range result.data {
		result.data[index] = m.data[index] / value
	}

	return result, nil
}

// DivideScalarInto writes m divided by value into result.
//
// The destination must match m's shape. The destination is caller-owned and may
// alias m because each element is read before it is written.
func (m *Matrix) DivideScalarInto(value float32, result *Matrix) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if value == 0 {
		err = errors.New("matrix: division by zero scalar")
		return err
	}

	if err = result.requireShape("destination", m.rows, m.cols); err != nil {
		return err
	}

	var index int
	for index = range result.data {
		result.data[index] = m.data[index] / value
	}

	return nil
}

// MatMul returns the matrix product of m and other.
func (m *Matrix) MatMul(other *Matrix) (result *Matrix, err error) {
	var next Matrix

	if err = m.validate(); err != nil {
		return nil, err
	}

	if err = other.validate(); err != nil {
		return nil, err
	}

	if m.cols != other.rows {
		err = fmt.Errorf(
			"matrix: multiplication shape mismatch: left %dx%d, right %dx%d",
			m.rows,
			m.cols,
			other.rows,
			other.cols,
		)
		return nil, err
	}

	next.rows = m.rows
	next.cols = other.cols
	next.data = make([]float32, m.rows*other.cols)
	result = &next

	matMulInto(m, other, result)
	return result, nil
}

// MatMulInto writes the matrix product of m and other into result.
//
// The destination must be shaped [m.Rows(), other.Cols()] and must not be one
// of the input matrices.
func (m *Matrix) MatMulInto(other, result *Matrix) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if err = other.validate(); err != nil {
		return err
	}

	if m.cols != other.rows {
		err = fmt.Errorf(
			"matrix: multiplication shape mismatch: left %dx%d, right %dx%d",
			m.rows,
			m.cols,
			other.rows,
			other.cols,
		)
		return err
	}

	if result == m || result == other {
		err = errors.New("matrix: multiplication destination must not alias inputs")
		return err
	}

	if err = result.requireShape("multiplication destination", m.rows, other.cols); err != nil {
		return err
	}

	matMulInto(m, other, result)
	return nil
}

// MatMulLeftTransposeInto writes the matrix product of m transposed and other into result.
//
// The destination must be shaped [m.Cols(), other.Cols()] and must not be one
// of the input matrices.
func (m *Matrix) MatMulLeftTransposeInto(other, result *Matrix) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if err = other.validate(); err != nil {
		return err
	}

	if m.rows != other.rows {
		err = fmt.Errorf(
			"matrix: left-transpose multiplication shape mismatch: left %dx%d, right %dx%d",
			m.rows,
			m.cols,
			other.rows,
			other.cols,
		)
		return err
	}

	if result == m || result == other {
		err = errors.New("matrix: left-transpose multiplication destination must not alias inputs")
		return err
	}

	if err = result.requireShape("left-transpose multiplication destination", m.cols, other.cols); err != nil {
		return err
	}

	matMulLeftTransposeInto(m, other, result)
	return nil
}

// MatMulRightTransposeInto writes the matrix product of m and other transposed into result.
//
// The destination must be shaped [m.Rows(), other.Rows()] and must not be one
// of the input matrices.
func (m *Matrix) MatMulRightTransposeInto(other, result *Matrix) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if err = other.validate(); err != nil {
		return err
	}

	if m.cols != other.cols {
		err = fmt.Errorf(
			"matrix: right-transpose multiplication shape mismatch: left %dx%d, right %dx%d",
			m.rows,
			m.cols,
			other.rows,
			other.cols,
		)
		return err
	}

	if result == m || result == other {
		err = errors.New("matrix: right-transpose multiplication destination must not alias inputs")
		return err
	}

	if err = result.requireShape("right-transpose multiplication destination", m.rows, other.rows); err != nil {
		return err
	}

	matMulRightTransposeInto(m, other, result)
	return nil
}

// Transpose returns a matrix with rows and columns swapped.
func (m *Matrix) Transpose() (result *Matrix, err error) {
	var next Matrix

	if err = m.validate(); err != nil {
		return nil, err
	}

	next.rows = m.cols
	next.cols = m.rows
	next.data = make([]float32, len(m.data))
	result = &next

	err = m.TransposeInto(result)
	return result, err
}

// TransposeInto writes the transpose of m into result.
//
// The destination must be shaped [m.Cols(), m.Rows()] and must not be m.
func (m *Matrix) TransposeInto(result *Matrix) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if result == m {
		err = errors.New("matrix: transpose destination must not alias input")
		return err
	}

	if err = result.requireShape("transpose destination", m.cols, m.rows); err != nil {
		return err
	}

	var (
		row int
		col int
	)

	for row = 0; row < m.rows; row++ {
		for col = 0; col < m.cols; col++ {
			result.data[col*result.cols+row] = m.data[row*m.cols+col]
		}
	}

	return nil
}

// RowSums returns one sum for each row.
func (m *Matrix) RowSums() (sums []float32, err error) {
	if err = m.validate(); err != nil {
		return nil, err
	}

	sums = make([]float32, m.rows)

	var (
		row int
		col int
	)

	for row = 0; row < m.rows; row++ {
		for col = 0; col < m.cols; col++ {
			sums[row] += m.data[row*m.cols+col]
		}
	}

	return sums, nil
}

// RowSumsInto writes row sums into a [m.Rows(), 1] destination matrix.
//
// The destination is caller-owned and must not alias m.
func (m *Matrix) RowSumsInto(result *Matrix) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if result == m {
		err = errors.New("matrix: row sums destination must not alias input")
		return err
	}

	if err = result.requireShape("row sums destination", m.rows, 1); err != nil {
		return err
	}

	var (
		row int
		col int
	)

	for row = 0; row < m.rows; row++ {
		result.data[row] = 0
		for col = 0; col < m.cols; col++ {
			result.data[row] += m.data[row*m.cols+col]
		}
	}

	return nil
}

// ColumnSums returns one sum for each column.
func (m *Matrix) ColumnSums() (sums []float32, err error) {
	if err = m.validate(); err != nil {
		return nil, err
	}

	sums = make([]float32, m.cols)

	var (
		row int
		col int
	)

	for row = 0; row < m.rows; row++ {
		for col = 0; col < m.cols; col++ {
			sums[col] += m.data[row*m.cols+col]
		}
	}

	return sums, nil
}

// ColumnSumsInto writes column sums into a [1, m.Cols()] destination matrix.
//
// The destination is caller-owned and must not alias m.
func (m *Matrix) ColumnSumsInto(result *Matrix) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if result == m {
		err = errors.New("matrix: column sums destination must not alias input")
		return err
	}

	if err = result.requireShape("column sums destination", 1, m.cols); err != nil {
		return err
	}

	var (
		row      int
		col      int
		rowStart int
	)

	for col = range result.data {
		result.data[col] = 0
	}

	if m.cols >= columnReductionAddMinCols {
		for row = 0; row < m.rows; row++ {
			rowStart = row * m.cols
			addInto(result.data, m.data[rowStart:rowStart+m.cols], result.data)
		}

		return nil
	}

	for row = 0; row < m.rows; row++ {
		for col = 0; col < m.cols; col++ {
			result.data[col] += m.data[row*m.cols+col]
		}
	}

	return nil
}

// AccumulateColumnSumsInto adds column sums to a [1, m.Cols()] destination matrix.
//
// The destination is caller-owned and must not alias m.
func (m *Matrix) AccumulateColumnSumsInto(result *Matrix) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if result == m {
		err = errors.New("matrix: column sums destination must not alias input")
		return err
	}

	if err = result.requireShape("column sums destination", 1, m.cols); err != nil {
		return err
	}

	var (
		row      int
		col      int
		rowStart int
	)

	if m.cols >= columnReductionAddMinCols {
		for row = 0; row < m.rows; row++ {
			rowStart = row * m.cols
			addInto(result.data, m.data[rowStart:rowStart+m.cols], result.data)
		}

		return nil
	}

	for row = 0; row < m.rows; row++ {
		for col = 0; col < m.cols; col++ {
			result.data[col] += m.data[row*m.cols+col]
		}
	}

	return nil
}

// AddRowVectorInPlace adds a [1, m.Cols()] row vector to every row of m.
//
// The receiver is updated in place. The row vector is read but not retained.
func (m *Matrix) AddRowVectorInPlace(rowVector *Matrix) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if err = rowVector.requireShape("row vector", 1, m.cols); err != nil {
		return err
	}

	var (
		row         int
		col         int
		valueOffset int
	)

	for row = 0; row < m.rows; row++ {
		valueOffset = row * m.cols
		for col = 0; col < m.cols; col++ {
			m.data[valueOffset+col] += rowVector.data[col]
		}
	}

	return nil
}

// Apply returns a matrix with fn applied to every element.
func (m *Matrix) Apply(fn func(float32) float32) (result *Matrix, err error) {
	if fn == nil {
		err = errors.New("matrix: apply function is nil")
		return nil, err
	}

	if err = m.validate(); err != nil {
		return nil, err
	}

	result = m.newLike()

	var index int
	for index = range result.data {
		result.data[index] = fn(m.data[index])
	}

	return result, nil
}

// ApplyInto writes fn applied to every element of m into result.
//
// The destination must match m's shape. The destination is caller-owned and may
// alias m because each element is read before it is written.
func (m *Matrix) ApplyInto(fn func(float32) float32, result *Matrix) (err error) {
	if fn == nil {
		err = errors.New("matrix: apply function is nil")
		return err
	}

	if err = m.validate(); err != nil {
		return err
	}

	if err = result.requireShape("destination", m.rows, m.cols); err != nil {
		return err
	}

	var index int
	for index = range result.data {
		result.data[index] = fn(m.data[index])
	}

	return nil
}

// Pairwise visits matching elements of m and other in row-major order.
//
// The callback receives element coordinates and values from each matrix. Matrix
// storage is not exposed to the callback.
func (m *Matrix) Pairwise(other *Matrix, fn func(row, col int, left, right float32) (err error)) (err error) {
	if fn == nil {
		err = errors.New("matrix: pairwise function is nil")
		return err
	}

	if err = m.sameShape(other); err != nil {
		return err
	}

	var (
		row   int
		col   int
		index int
	)

	for row = 0; row < m.rows; row++ {
		for col = 0; col < m.cols; col++ {
			index = row*m.cols + col
			if err = fn(row, col, m.data[index], other.data[index]); err != nil {
				return err
			}
		}
	}

	return nil
}

// PairwiseInto writes callback results for matching elements of m and other into result.
//
// The destination must have the same shape as both inputs. Matrix storage is not
// exposed to the callback. The destination is caller-owned and may alias either
// input because each element is read before it is written.
func (m *Matrix) PairwiseInto(
	other, result *Matrix,
	fn func(row, col int, left, right float32) (value float32, err error),
) (err error) {
	if fn == nil {
		err = errors.New("matrix: pairwise function is nil")
		return err
	}

	if err = m.sameShape(other); err != nil {
		return err
	}

	if err = result.requireShape("destination", m.rows, m.cols); err != nil {
		return err
	}

	var (
		row   int
		col   int
		index int
	)

	for row = 0; row < m.rows; row++ {
		for col = 0; col < m.cols; col++ {
			index = row*m.cols + col
			result.data[index], err = fn(row, col, m.data[index], other.data[index])
			if err != nil {
				return err
			}
		}
	}

	return nil
}

func (m *Matrix) newLike() (result *Matrix) {
	var next Matrix

	next.rows = m.rows
	next.cols = m.cols
	next.data = make([]float32, len(m.data))
	return &next
}

func (m *Matrix) validate() (err error) {
	if m == nil {
		err = errors.New("matrix: matrix is nil")
		return err
	}

	var size int
	if size, err = m.matrixSize(); err != nil {
		return err
	}

	if len(m.data) != size {
		err = fmt.Errorf("matrix: storage length mismatch: got %d, want %d", len(m.data), size)
		return err
	}

	return nil
}

func (m *Matrix) validateIndex(row, col int) (err error) {
	if row < 0 || row >= m.rows {
		err = fmt.Errorf("matrix: row index out of range: row=%d rows=%d", row, m.rows)
		return err
	}

	if col < 0 || col >= m.cols {
		err = fmt.Errorf("matrix: column index out of range: col=%d cols=%d", col, m.cols)
		return err
	}

	return nil
}

func (m *Matrix) index(row, col int) (index int) {
	index = row*m.cols + col
	return index
}

func (m *Matrix) sameShape(other *Matrix) (err error) {
	if err = m.validate(); err != nil {
		return err
	}

	if err = other.validate(); err != nil {
		return err
	}

	if m.rows != other.rows || m.cols != other.cols {
		err = fmt.Errorf(
			"matrix: shape mismatch: left %dx%d, right %dx%d",
			m.rows,
			m.cols,
			other.rows,
			other.cols,
		)
		return err
	}

	return nil
}

func (m *Matrix) requireShape(name string, rows, cols int) (err error) {
	var (
		matrixRows int
		matrixCols int
	)

	if err = m.validate(); err != nil {
		return err
	}

	matrixRows, matrixCols = m.Shape()
	if matrixRows != rows || matrixCols != cols {
		err = fmt.Errorf("matrix: %s shape mismatch: got %dx%d, want %dx%d", name, matrixRows, matrixCols, rows, cols)
		return err
	}

	return nil
}

func (m *Matrix) matrixSize() (size int, err error) {
	if m == nil {
		err = errors.New("matrix: matrix is nil")
		return 0, err
	}

	if m.rows <= 0 || m.cols <= 0 {
		err = fmt.Errorf("matrix: dimensions must be positive: rows=%d cols=%d", m.rows, m.cols)
		return 0, err
	}

	var maxInt int
	maxInt = int(^uint(0) >> 1)

	if m.rows > maxInt/m.cols {
		err = fmt.Errorf("matrix: dimensions are too large: rows=%d cols=%d", m.rows, m.cols)
		return 0, err
	}

	size = m.rows * m.cols
	return size, nil
}
