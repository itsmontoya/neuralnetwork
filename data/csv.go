package data

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

// CSVConfig configures LoadCSV.
type CSVConfig struct {
	// InputColumns is the number of input feature columns at the start of each row.
	InputColumns int
	// TargetColumns is the number of target value columns after the input columns.
	TargetColumns int
	// HasHeader skips the first CSV record when true.
	HasHeader bool
}

func (c CSVConfig) validate() (err error) {
	if c.InputColumns <= 0 {
		err = fmt.Errorf("data: csv input columns must be positive: inputColumns=%d", c.InputColumns)
		return err
	}

	if c.TargetColumns <= 0 {
		err = fmt.Errorf("data: csv target columns must be positive: targetColumns=%d", c.TargetColumns)
		return err
	}

	return nil
}

// LoadCSV reads a supervised dataset from CSV data.
//
// Each data row must contain InputColumns input values followed by
// TargetColumns target values. Values are parsed as float32.
func LoadCSV(reader io.Reader, config CSVConfig) (out *Dataset, err error) {
	var (
		csvReader    *csv.Reader
		record       []string
		inputValues  []float32
		targetValues []float32
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
		rows         int
		recordNumber int
	)

	if reader == nil {
		err = errors.New("data: csv reader is nil")
		return nil, err
	}

	if err = config.validate(); err != nil {
		return nil, err
	}

	csvReader = csv.NewReader(reader)
	csvReader.FieldsPerRecord = -1
	csvReader.TrimLeadingSpace = true

	if config.HasHeader {
		if _, err = csvReader.Read(); err != nil {
			err = fmt.Errorf("data: csv header read failed: %w", err)
			return nil, err
		}
	}

	for {
		record, err = csvReader.Read()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			err = fmt.Errorf("data: csv record read failed: record=%d: %w", recordNumber+1, err)
			return nil, err
		}

		recordNumber++
		if isBlankCSVRecord(record) {
			continue
		}

		if inputValues, targetValues, err = appendCSVRecord(
			inputValues,
			targetValues,
			record,
			config,
			recordNumber,
		); err != nil {
			return nil, err
		}

		rows++
	}

	if rows == 0 {
		err = errors.New("data: csv contains no data rows")
		return nil, err
	}

	if inputs, err = matrix.FromSlice(rows, config.InputColumns, inputValues); err != nil {
		err = fmt.Errorf("data: csv inputs matrix failed: %w", err)
		return nil, err
	}

	if targets, err = matrix.FromSlice(rows, config.TargetColumns, targetValues); err != nil {
		err = fmt.Errorf("data: csv targets matrix failed: %w", err)
		return nil, err
	}

	out, err = NewDataset(inputs, targets)
	return out, err
}

func appendCSVRecord(
	inputValues []float32,
	targetValues []float32,
	record []string,
	config CSVConfig,
	recordNumber int,
) (nextInputs, nextTargets []float32, err error) {
	var (
		expectedColumns int
		column          int
		value           float32
	)

	expectedColumns = config.InputColumns + config.TargetColumns
	if len(record) != expectedColumns {
		err = fmt.Errorf(
			"data: csv column count mismatch: record=%d columns=%d want=%d",
			recordNumber,
			len(record),
			expectedColumns,
		)
		return nil, nil, err
	}

	nextInputs = inputValues
	nextTargets = targetValues

	for column = 0; column < config.InputColumns; column++ {
		if value, err = parseCSVFloat(record[column], recordNumber, column+1); err != nil {
			return nil, nil, err
		}

		nextInputs = append(nextInputs, value)
	}

	for column = 0; column < config.TargetColumns; column++ {
		if value, err = parseCSVFloat(
			record[config.InputColumns+column],
			recordNumber,
			config.InputColumns+column+1,
		); err != nil {
			return nil, nil, err
		}

		nextTargets = append(nextTargets, value)
	}

	return nextInputs, nextTargets, nil
}

func parseCSVFloat(field string, recordNumber, column int) (value float32, err error) {
	var parsed float64

	field = strings.TrimSpace(field)
	if field == "" {
		err = fmt.Errorf("data: csv value is empty: record=%d column=%d", recordNumber, column)
		return 0, err
	}

	if parsed, err = strconv.ParseFloat(field, 32); err != nil {
		err = fmt.Errorf("data: csv value parse failed: record=%d column=%d value=%q: %w", recordNumber, column, field, err)
		return 0, err
	}

	value = float32(parsed)
	return value, nil
}

func isBlankCSVRecord(record []string) (blank bool) {
	if len(record) != 1 {
		return false
	}

	blank = strings.TrimSpace(record[0]) == ""
	return blank
}
