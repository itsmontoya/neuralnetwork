package metric

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// NewBinaryConfusionMatrix constructs a binary confusion matrix using the
// default threshold of 0.5.
func NewBinaryConfusionMatrix(predictions, targets *matrix.Matrix) (out *ConfusionMatrix, err error) {
	out, err = NewBinaryConfusionMatrixWithThreshold(predictions, targets, defaultBinaryThreshold)
	return out, err
}

// NewBinaryConfusionMatrixWithThreshold constructs a binary confusion matrix
// with a finite threshold.
func NewBinaryConfusionMatrixWithThreshold(predictions, targets *matrix.Matrix, threshold float32) (out *ConfusionMatrix, err error) {
	var (
		predictedClasses []int
		targetClasses    []int
	)

	if predictedClasses, targetClasses, err = binaryClassValues(predictions, targets, threshold); err != nil {
		return nil, err
	}

	out, err = newConfusionMatrix(2, predictedClasses, targetClasses)
	return out, err
}

// NewCategoricalConfusionMatrix constructs a confusion matrix for one-hot targets.
//
// Predicted classes use each prediction row's first maximum value.
func NewCategoricalConfusionMatrix(predictions, targets *matrix.Matrix) (out *ConfusionMatrix, err error) {
	var (
		classCount       int
		predictedClasses []int
		targetClasses    []int
	)

	if classCount, predictedClasses, targetClasses, err = categoricalClassValues(predictions, targets); err != nil {
		return nil, err
	}

	out, err = newConfusionMatrix(classCount, predictedClasses, targetClasses)
	return out, err
}

// ConfusionMatrix stores classification counts as target rows by predicted columns.
type ConfusionMatrix struct {
	counts [][]int
	total  int
}

func newConfusionMatrix(classCount int, predictedClasses, targetClasses []int) (out *ConfusionMatrix, err error) {
	var (
		index          int
		predictedClass int
		targetClass    int
		c              ConfusionMatrix
	)

	if classCount <= 0 {
		err = fmt.Errorf("metric: confusion matrix class count must be positive: classCount=%d", classCount)
		return nil, err
	}

	if len(predictedClasses) != len(targetClasses) {
		err = fmt.Errorf("metric: confusion matrix class length mismatch: predictions=%d targets=%d", len(predictedClasses), len(targetClasses))
		return nil, err
	}

	c.counts = make([][]int, classCount)
	c.total = len(predictedClasses)
	for index = range c.counts {
		c.counts[index] = make([]int, classCount)
	}

	for index, predictedClass = range predictedClasses {
		targetClass = targetClasses[index]
		if err = validateClassIndex("predicted", predictedClass, classCount); err != nil {
			return nil, err
		}

		if err = validateClassIndex("target", targetClass, classCount); err != nil {
			return nil, err
		}

		c.counts[targetClass][predictedClass]++
	}

	return &c, nil
}

// ClassCount returns the number of classes represented by the matrix.
func (c *ConfusionMatrix) ClassCount() (classCount int) {
	if c == nil {
		return 0
	}

	classCount = len(c.counts)
	return classCount
}

// Total returns the number of samples counted by the matrix.
func (c *ConfusionMatrix) Total() (total int) {
	if c == nil {
		return 0
	}

	total = c.total
	return total
}

// Counts returns a copy of target-row by predicted-column counts.
func (c *ConfusionMatrix) Counts() (counts [][]int, err error) {
	var (
		row int
	)

	if err = c.validate(); err != nil {
		return nil, err
	}

	counts = make([][]int, len(c.counts))
	for row = range c.counts {
		counts[row] = append([]int(nil), c.counts[row]...)
	}

	return counts, nil
}

// At returns the count for targetClass and predictedClass.
func (c *ConfusionMatrix) At(targetClass, predictedClass int) (count int, err error) {
	if err = c.validate(); err != nil {
		return 0, err
	}

	if err = validateClassIndex("target", targetClass, len(c.counts)); err != nil {
		return 0, err
	}

	if err = validateClassIndex("predicted", predictedClass, len(c.counts)); err != nil {
		return 0, err
	}

	count = c.counts[targetClass][predictedClass]
	return count, nil
}

// Accuracy returns the fraction of samples on the matrix diagonal.
func (c *ConfusionMatrix) Accuracy() (value float32, err error) {
	var (
		classIndex int
		correct    int
	)

	if err = c.validate(); err != nil {
		return 0, err
	}

	if c.total == 0 {
		return 0, nil
	}

	for classIndex = range c.counts {
		correct += c.counts[classIndex][classIndex]
	}

	value = float32(correct) / float32(c.total)
	return value, nil
}

// Precision returns precision for classIndex.
func (c *ConfusionMatrix) Precision(classIndex int) (value float32, err error) {
	var predictedTotal int

	if err = c.validateClass(classIndex); err != nil {
		return 0, err
	}

	predictedTotal = c.predictedTotal(classIndex)
	if predictedTotal == 0 {
		return 0, nil
	}

	value = float32(c.counts[classIndex][classIndex]) / float32(predictedTotal)
	return value, nil
}

// Recall returns recall for classIndex.
func (c *ConfusionMatrix) Recall(classIndex int) (value float32, err error) {
	var targetTotal int

	if err = c.validateClass(classIndex); err != nil {
		return 0, err
	}

	targetTotal = c.targetTotal(classIndex)
	if targetTotal == 0 {
		return 0, nil
	}

	value = float32(c.counts[classIndex][classIndex]) / float32(targetTotal)
	return value, nil
}

// F1 returns the harmonic mean of precision and recall for classIndex.
func (c *ConfusionMatrix) F1(classIndex int) (value float32, err error) {
	var (
		precision float32
		recall    float32
	)

	if precision, err = c.Precision(classIndex); err != nil {
		return 0, err
	}

	if recall, err = c.Recall(classIndex); err != nil {
		return 0, err
	}

	if precision+recall == 0 {
		return 0, nil
	}

	value = 2 * precision * recall / (precision + recall)
	return value, nil
}

// MacroPrecision returns the unweighted mean precision across classes.
func (c *ConfusionMatrix) MacroPrecision() (value float32, err error) {
	value, err = c.macro(c.Precision)
	return value, err
}

// MacroRecall returns the unweighted mean recall across classes.
func (c *ConfusionMatrix) MacroRecall() (value float32, err error) {
	value, err = c.macro(c.Recall)
	return value, err
}

// MacroF1 returns the unweighted mean F1 across classes.
func (c *ConfusionMatrix) MacroF1() (value float32, err error) {
	value, err = c.macro(c.F1)
	return value, err
}

func (c *ConfusionMatrix) macro(fn func(int) (float32, error)) (value float32, err error) {
	var (
		classIndex int
		next       float32
	)

	if err = c.validate(); err != nil {
		return 0, err
	}

	for classIndex = range c.counts {
		if next, err = fn(classIndex); err != nil {
			return 0, err
		}

		value += next
	}

	value /= float32(len(c.counts))
	return value, nil
}

func (c *ConfusionMatrix) targetTotal(classIndex int) (total int) {
	var predictedClass int

	for predictedClass = range c.counts[classIndex] {
		total += c.counts[classIndex][predictedClass]
	}

	return total
}

func (c *ConfusionMatrix) predictedTotal(classIndex int) (total int) {
	var targetClass int

	for targetClass = range c.counts {
		total += c.counts[targetClass][classIndex]
	}

	return total
}

func (c *ConfusionMatrix) validateClass(classIndex int) (err error) {
	if err = c.validate(); err != nil {
		return err
	}

	err = validateClassIndex("class", classIndex, len(c.counts))
	return err
}

func (c *ConfusionMatrix) validate() (err error) {
	var row int

	if c == nil {
		err = errors.New("metric: confusion matrix is nil")
		return err
	}

	if len(c.counts) == 0 {
		err = errors.New("metric: confusion matrix has no classes")
		return err
	}

	for row = range c.counts {
		if len(c.counts[row]) != len(c.counts) {
			err = fmt.Errorf("metric: confusion matrix row %d length mismatch: got %d, want %d", row, len(c.counts[row]), len(c.counts))
			return err
		}
	}

	return nil
}

func validateClassIndex(name string, classIndex, classCount int) (err error) {
	if classIndex < 0 || classIndex >= classCount {
		err = fmt.Errorf("metric: %s class index out of range: class=%d classes=%d", name, classIndex, classCount)
		return err
	}

	return nil
}
