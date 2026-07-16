package metric

import (
	"errors"
	"fmt"

	"github.com/itsmontoya/neuralnetwork/internal/f32"
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
	var confusionMatrix ConfusionMatrix

	if confusionMatrix, err = binaryConfusionMatrix(predictions, targets, threshold); err != nil {
		return nil, err
	}

	out = &confusionMatrix
	return out, nil
}

// NewCategoricalConfusionMatrix constructs a confusion matrix for one-hot targets.
//
// Predicted classes use each prediction row's first maximum value.
func NewCategoricalConfusionMatrix(predictions, targets *matrix.Matrix) (out *ConfusionMatrix, err error) {
	var confusionMatrix ConfusionMatrix

	if confusionMatrix, err = categoricalConfusionMatrix(predictions, targets); err != nil {
		return nil, err
	}

	out = &confusionMatrix
	return out, nil
}

func binaryConfusionMatrix(predictions, targets *matrix.Matrix, threshold float32) (out ConfusionMatrix, err error) {
	var (
		truePositive      int
		predictedPositive int
		targetPositive    int
	)

	if f32.IsNaN(threshold) || f32.IsInf(threshold, 0) {
		err = fmt.Errorf("metric: binary classification threshold must be finite: threshold=%g", threshold)
		return out, err
	}

	if out.total, truePositive, predictedPositive, targetPositive, err = binaryPositiveTotals(
		predictions,
		targets,
		threshold,
		"binary classification",
	); err != nil {
		return out, err
	}

	out.classCount = 2
	out.counts = make([]int, out.classCount*out.classCount)
	out.counts[0] = out.total - predictedPositive - targetPositive + truePositive
	out.counts[1] = predictedPositive - truePositive
	out.counts[2] = targetPositive - truePositive
	out.counts[3] = truePositive
	return out, nil
}

func categoricalConfusionMatrix(predictions, targets *matrix.Matrix) (out ConfusionMatrix, err error) {
	if out.total, out.classCount, err = matrixShapePair(predictions, targets); err != nil {
		return out, err
	}

	out.counts = make([]int, out.classCount*out.classCount)
	if _, err = categoricalClassSummary(
		predictions,
		targets,
		out.classCount,
		out.counts,
	); err != nil {
		return out, err
	}

	return out, nil
}

// ConfusionMatrix stores classification counts as target rows by predicted columns.
type ConfusionMatrix struct {
	counts     []int
	classCount int
	total      int
}

// ClassCount returns the number of classes represented by the matrix.
func (c *ConfusionMatrix) ClassCount() (classCount int) {
	if c == nil {
		return 0
	}

	classCount = c.classCount
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
		row    int
		offset int
	)

	if err = c.validate(); err != nil {
		return nil, err
	}

	counts = make([][]int, c.classCount)
	for row = 0; row < c.classCount; row++ {
		offset = row * c.classCount
		counts[row] = make([]int, c.classCount)
		copy(counts[row], c.counts[offset:offset+c.classCount])
	}

	return counts, nil
}

// At returns the count for targetClass and predictedClass.
func (c *ConfusionMatrix) At(targetClass, predictedClass int) (count int, err error) {
	if err = c.validate(); err != nil {
		return 0, err
	}

	if err = validateClassIndex("target", targetClass, c.classCount); err != nil {
		return 0, err
	}

	if err = validateClassIndex("predicted", predictedClass, c.classCount); err != nil {
		return 0, err
	}

	count = c.counts[targetClass*c.classCount+predictedClass]
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

	for classIndex = 0; classIndex < c.classCount; classIndex++ {
		correct += c.counts[classIndex*c.classCount+classIndex]
	}

	value = float32(correct) / float32(c.total)
	return value, nil
}

// Precision returns precision for classIndex.
func (c *ConfusionMatrix) Precision(classIndex int) (value float32, err error) {
	if err = c.validateClass(classIndex); err != nil {
		return 0, err
	}

	value = c.precision(classIndex)
	return value, nil
}

// Recall returns recall for classIndex.
func (c *ConfusionMatrix) Recall(classIndex int) (value float32, err error) {
	if err = c.validateClass(classIndex); err != nil {
		return 0, err
	}

	value = c.recall(classIndex)
	return value, nil
}

// F1 returns the harmonic mean of precision and recall for classIndex.
func (c *ConfusionMatrix) F1(classIndex int) (value float32, err error) {
	if err = c.validateClass(classIndex); err != nil {
		return 0, err
	}

	value = c.f1(classIndex)
	return value, nil
}

// MacroPrecision returns the unweighted mean precision across classes.
func (c *ConfusionMatrix) MacroPrecision() (value float32, err error) {
	var classIndex int

	if err = c.validate(); err != nil {
		return 0, err
	}

	for classIndex = 0; classIndex < c.classCount; classIndex++ {
		value += c.precision(classIndex)
	}

	value /= float32(c.classCount)
	return value, nil
}

// MacroRecall returns the unweighted mean recall across classes.
func (c *ConfusionMatrix) MacroRecall() (value float32, err error) {
	var classIndex int

	if err = c.validate(); err != nil {
		return 0, err
	}

	for classIndex = 0; classIndex < c.classCount; classIndex++ {
		value += c.recall(classIndex)
	}

	value /= float32(c.classCount)
	return value, nil
}

// MacroF1 returns the unweighted mean F1 across classes.
func (c *ConfusionMatrix) MacroF1() (value float32, err error) {
	var classIndex int

	if err = c.validate(); err != nil {
		return 0, err
	}

	for classIndex = 0; classIndex < c.classCount; classIndex++ {
		value += c.f1(classIndex)
	}

	value /= float32(c.classCount)
	return value, nil
}

func (c *ConfusionMatrix) precision(classIndex int) (value float32) {
	var predictedTotal int

	predictedTotal = c.predictedTotal(classIndex)
	value = precisionValue(c.counts[classIndex*c.classCount+classIndex], predictedTotal)
	return value
}

func (c *ConfusionMatrix) recall(classIndex int) (value float32) {
	var targetTotal int

	targetTotal = c.targetTotal(classIndex)
	value = recallValue(c.counts[classIndex*c.classCount+classIndex], targetTotal)
	return value
}

func (c *ConfusionMatrix) f1(classIndex int) (value float32) {
	var (
		truePositive      int
		predictedPositive int
		targetPositive    int
	)

	truePositive = c.counts[classIndex*c.classCount+classIndex]
	predictedPositive = c.predictedTotal(classIndex)
	targetPositive = c.targetTotal(classIndex)
	value = f1Value(truePositive, predictedPositive, targetPositive)
	return value
}

func (c *ConfusionMatrix) targetTotal(classIndex int) (total int) {
	var (
		predictedClass int
		offset         int
	)

	offset = classIndex * c.classCount
	for predictedClass = 0; predictedClass < c.classCount; predictedClass++ {
		total += c.counts[offset+predictedClass]
	}

	return total
}

func (c *ConfusionMatrix) predictedTotal(classIndex int) (total int) {
	var targetClass int

	for targetClass = 0; targetClass < c.classCount; targetClass++ {
		total += c.counts[targetClass*c.classCount+classIndex]
	}

	return total
}

func (c *ConfusionMatrix) validateClass(classIndex int) (err error) {
	if err = c.validate(); err != nil {
		return err
	}

	err = validateClassIndex("class", classIndex, c.classCount)
	return err
}

func (c *ConfusionMatrix) validate() (err error) {
	if c == nil {
		err = errors.New("metric: confusion matrix is nil")
		return err
	}

	if c.classCount <= 0 {
		err = errors.New("metric: confusion matrix has no classes")
		return err
	}

	if len(c.counts) != c.classCount*c.classCount {
		err = fmt.Errorf(
			"metric: confusion matrix storage length mismatch: got %d, want %d",
			len(c.counts),
			c.classCount*c.classCount,
		)
		return err
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
