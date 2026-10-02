package model

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"io"
	"math"
)

const (
	binarySerializationVersion   uint16 = 1
	binarySerializationBuffer           = 32 * 1024
	binarySerializationMaxLayers        = 4096
	binarySerializationMaxWidths        = 4096
	binarySerializationMaxString        = 256
	binarySerializationMaxValues uint64 = 1 << 27
	binarySerializationMagic            = "NNSEQBIN"
)

func encodeSequentialBinary(writer io.Writer, s *Sequential) (checksum ModelChecksum, err error) {
	var (
		document      sequentialDocument
		digest        hash.Hash
		payloadWriter io.Writer
	)

	if document, err = sequentialDocumentFromModel(s); err != nil {
		return checksum, err
	}
	if err = validateBinaryDocument(document); err != nil {
		return checksum, err
	}

	if err = writeBinaryBytes(writer, []byte(binarySerializationMagic)); err != nil {
		err = fmt.Errorf("write magic: %w", err)
		return checksum, err
	}
	if err = writeBinaryUint16(writer, binarySerializationVersion); err != nil {
		err = fmt.Errorf("write version: %w", err)
		return checksum, err
	}

	digest = sha256.New()
	payloadWriter = io.MultiWriter(writer, digest)
	if err = encodeBinaryDocument(payloadWriter, document); err != nil {
		return checksum, err
	}
	copy(checksum[:], digest.Sum(nil))
	if err = writeBinaryBytes(writer, checksum[:]); err != nil {
		err = fmt.Errorf("write checksum: %w", err)
		return ModelChecksum{}, err
	}

	return checksum, nil
}

func decodeSequentialBinary(reader io.Reader) (s *Sequential, checksum ModelChecksum, err error) {
	var (
		magic          [len(binarySerializationMagic)]byte
		storedChecksum ModelChecksum
		document       sequentialDocument
		digest         hash.Hash
		payloadReader  io.Reader
		version        uint16
	)

	if _, err = io.ReadFull(reader, magic[:]); err != nil {
		err = fmt.Errorf("read magic: %w", err)
		return nil, checksum, err
	}
	if string(magic[:]) != binarySerializationMagic {
		err = fmt.Errorf("unsupported magic %q", magic)
		return nil, checksum, err
	}
	if version, err = readBinaryUint16(reader); err != nil {
		err = fmt.Errorf("read version: %w", err)
		return nil, checksum, err
	}
	if version != binarySerializationVersion {
		err = fmt.Errorf("unsupported version %d", version)
		return nil, checksum, err
	}

	digest = sha256.New()
	payloadReader = io.TeeReader(reader, digest)
	if document, err = decodeBinaryDocument(payloadReader); err != nil {
		return nil, checksum, err
	}
	copy(checksum[:], digest.Sum(nil))
	if _, err = io.ReadFull(reader, storedChecksum[:]); err != nil {
		err = fmt.Errorf("read checksum: %w", err)
		return nil, ModelChecksum{}, err
	}
	if storedChecksum != checksum {
		err = errors.New("checksum mismatch")
		return nil, ModelChecksum{}, err
	}
	if err = ensureBinaryEOF(reader); err != nil {
		return nil, ModelChecksum{}, err
	}

	if s, err = document.model(); err != nil {
		return nil, ModelChecksum{}, err
	}
	return s, checksum, nil
}

func validateBinaryDocument(document sequentialDocument) (err error) {
	var (
		current       serializedLayer
		matrices      []*serializedMatrix
		currentMatrix *serializedMatrix
		totalValues   uint64
		valueCount    uint64
	)

	if len(document.Layers) > binarySerializationMaxLayers {
		err = fmt.Errorf("layer count %d exceeds limit %d", len(document.Layers), binarySerializationMaxLayers)
		return err
	}
	for _, current = range document.Layers {
		if len(current.Type) > binarySerializationMaxString || len(current.Activation) > binarySerializationMaxString {
			err = errors.New("binary model string exceeds limit")
			return err
		}
		if len(current.ActivationWidths) > binarySerializationMaxWidths {
			err = fmt.Errorf(
				"activation width count %d exceeds limit %d",
				len(current.ActivationWidths),
				binarySerializationMaxWidths,
			)
			return err
		}
		matrices = []*serializedMatrix{
			current.InputWeights,
			current.RecurrentWeights,
			current.Weights,
			current.Biases,
			current.Gamma,
			current.Beta,
			current.RunningMean,
			current.RunningVariance,
		}
		for _, currentMatrix = range matrices {
			if currentMatrix == nil {
				continue
			}
			if err = currentMatrix.validate(); err != nil {
				return err
			}
			valueCount = uint64(len(currentMatrix.Values))
			if valueCount > binarySerializationMaxValues || totalValues > binarySerializationMaxValues-valueCount {
				err = fmt.Errorf("matrix values exceed limit %d", binarySerializationMaxValues)
				return err
			}
			totalValues += valueCount
		}
	}

	return nil
}

func encodeBinaryDocument(writer io.Writer, document sequentialDocument) (err error) {
	var (
		buffer  [binarySerializationBuffer]byte
		current serializedLayer
		index   int
	)

	if err = writeBinaryUint(writer, uint64(len(document.Layers))); err != nil {
		err = fmt.Errorf("write layer count: %w", err)
		return err
	}
	for index, current = range document.Layers {
		if err = encodeBinaryLayer(writer, current, buffer[:]); err != nil {
			err = fmt.Errorf("write layer %d: %w", index, err)
			return err
		}
	}

	return nil
}

func decodeBinaryDocument(reader io.Reader) (document sequentialDocument, err error) {
	var (
		buffer      [binarySerializationBuffer]byte
		layerCount  uint64
		totalValues uint64
		index       int
	)

	if layerCount, err = readBinaryUint(reader); err != nil {
		err = fmt.Errorf("read layer count: %w", err)
		return document, err
	}
	if layerCount > binarySerializationMaxLayers {
		err = fmt.Errorf("layer count %d exceeds limit %d", layerCount, binarySerializationMaxLayers)
		return document, err
	}

	document.Format = serializationFormatSequential
	document.Version = serializationVersion
	document.Layers = make([]serializedLayer, int(layerCount))
	for index = range document.Layers {
		if document.Layers[index], err = decodeBinaryLayer(reader, buffer[:], &totalValues); err != nil {
			err = fmt.Errorf("read layer %d: %w", index, err)
			return sequentialDocument{}, err
		}
	}

	return document, nil
}

func encodeBinaryLayer(writer io.Writer, current serializedLayer, buffer []byte) (err error) {
	var (
		integers      []int
		matrices      []*serializedMatrix
		value         int
		width         int
		windowHeight  int
		windowWidth   int
		currentMatrix *serializedMatrix
	)

	if err = writeBinaryString(writer, current.Type); err != nil {
		return err
	}
	windowHeight = current.WindowHeight
	windowWidth = current.WindowWidth
	if current.Type == serializationLayerAdaptiveAveragePool2D {
		windowHeight = current.OutputHeight
		windowWidth = current.OutputWidth
	}
	integers = []int{
		current.InputSize,
		current.OutputSize,
		current.Steps,
		current.FeatureSize,
		current.HiddenSize,
	}
	for _, value = range integers {
		if err = writeBinaryNonnegativeInt(writer, value); err != nil {
			return err
		}
	}
	if err = writeBinaryString(writer, current.Activation); err != nil {
		return err
	}
	if err = writeBinaryUint(writer, uint64(len(current.ActivationWidths))); err != nil {
		return err
	}
	for _, width = range current.ActivationWidths {
		if err = writeBinaryNonnegativeInt(writer, width); err != nil {
			return err
		}
	}
	if err = writeBinaryFloat32(writer, current.Rate); err != nil {
		return err
	}
	if err = writeBinaryFloat32(writer, current.Momentum); err != nil {
		return err
	}
	if err = writeBinaryFloat32(writer, current.Epsilon); err != nil {
		return err
	}

	matrices = []*serializedMatrix{
		current.InputWeights,
		current.RecurrentWeights,
		current.Weights,
		current.Biases,
		current.Gamma,
		current.Beta,
		current.RunningMean,
		current.RunningVariance,
	}
	for _, currentMatrix = range matrices {
		if err = encodeBinaryMatrix(writer, currentMatrix, buffer); err != nil {
			return err
		}
	}

	integers = []int{
		current.InputChannels,
		current.InputHeight,
		current.InputWidth,
		current.OutputChannels,
		current.KernelHeight,
		current.KernelWidth,
		current.StrideHeight,
		current.StrideWidth,
		current.PaddingHeight,
		current.PaddingWidth,
		windowHeight,
		windowWidth,
	}
	for _, value = range integers {
		if err = writeBinaryNonnegativeInt(writer, value); err != nil {
			return err
		}
	}

	return nil
}

func decodeBinaryLayer(reader io.Reader, buffer []byte, totalValues *uint64) (current serializedLayer, err error) {
	var (
		integers         []*int
		matrices         []**serializedMatrix
		activationWidths uint64
		index            int
	)

	if current.Type, err = readBinaryString(reader); err != nil {
		return current, err
	}
	integers = []*int{
		&current.InputSize,
		&current.OutputSize,
		&current.Steps,
		&current.FeatureSize,
		&current.HiddenSize,
	}
	for _, destination := range integers {
		if *destination, err = readBinaryInt(reader); err != nil {
			return current, err
		}
	}
	if current.Activation, err = readBinaryString(reader); err != nil {
		return current, err
	}
	if activationWidths, err = readBinaryUint(reader); err != nil {
		return current, err
	}
	if activationWidths > binarySerializationMaxWidths {
		err = fmt.Errorf("activation width count %d exceeds limit %d", activationWidths, binarySerializationMaxWidths)
		return current, err
	}
	current.ActivationWidths = make([]int, int(activationWidths))
	for index = range current.ActivationWidths {
		if current.ActivationWidths[index], err = readBinaryInt(reader); err != nil {
			return current, err
		}
	}
	if current.Rate, err = readBinaryFloat32(reader); err != nil {
		return current, err
	}
	if current.Momentum, err = readBinaryFloat32(reader); err != nil {
		return current, err
	}
	if current.Epsilon, err = readBinaryFloat32(reader); err != nil {
		return current, err
	}

	matrices = []**serializedMatrix{
		&current.InputWeights,
		&current.RecurrentWeights,
		&current.Weights,
		&current.Biases,
		&current.Gamma,
		&current.Beta,
		&current.RunningMean,
		&current.RunningVariance,
	}
	for _, destination := range matrices {
		if *destination, err = decodeBinaryMatrix(reader, buffer, totalValues); err != nil {
			return current, err
		}
	}

	integers = []*int{
		&current.InputChannels,
		&current.InputHeight,
		&current.InputWidth,
		&current.OutputChannels,
		&current.KernelHeight,
		&current.KernelWidth,
		&current.StrideHeight,
		&current.StrideWidth,
		&current.PaddingHeight,
		&current.PaddingWidth,
		&current.WindowHeight,
		&current.WindowWidth,
	}
	for _, destination := range integers {
		if *destination, err = readBinaryInt(reader); err != nil {
			return current, err
		}
	}
	if current.Type == serializationLayerAdaptiveAveragePool2D {
		current.OutputHeight = current.WindowHeight
		current.OutputWidth = current.WindowWidth
		current.WindowHeight = 0
		current.WindowWidth = 0
	}

	return current, nil
}

func encodeBinaryMatrix(writer io.Writer, current *serializedMatrix, buffer []byte) (err error) {
	var (
		bufferIndex int
		valueIndex  int
	)

	if current == nil {
		err = writeBinaryBytes(writer, []byte{0})
		return err
	}
	if err = current.validate(); err != nil {
		return err
	}
	if err = writeBinaryBytes(writer, []byte{1}); err != nil {
		return err
	}
	if err = writeBinaryNonnegativeInt(writer, current.Rows); err != nil {
		return err
	}
	if err = writeBinaryNonnegativeInt(writer, current.Cols); err != nil {
		return err
	}

	for valueIndex < len(current.Values) {
		bufferIndex = 0
		for valueIndex < len(current.Values) && bufferIndex+4 <= len(buffer) {
			binary.LittleEndian.PutUint32(buffer[bufferIndex:bufferIndex+4], math.Float32bits(current.Values[valueIndex]))
			bufferIndex += 4
			valueIndex++
		}
		if err = writeBinaryBytes(writer, buffer[:bufferIndex]); err != nil {
			return err
		}
	}

	return nil
}

func decodeBinaryMatrix(
	reader io.Reader,
	buffer []byte,
	totalValues *uint64,
) (current *serializedMatrix, err error) {
	var (
		presence    [1]byte
		rows        int
		cols        int
		valueCount  uint64
		chunkValues int
		bufferIndex int
		valuesRead  uint64
		value       float32
	)

	if _, err = io.ReadFull(reader, presence[:]); err != nil {
		return nil, err
	}
	if presence[0] == 0 {
		return nil, nil
	}
	if presence[0] != 1 {
		err = fmt.Errorf("invalid matrix presence marker %d", presence[0])
		return nil, err
	}
	if rows, err = readBinaryInt(reader); err != nil {
		return nil, err
	}
	if cols, err = readBinaryInt(reader); err != nil {
		return nil, err
	}
	if rows <= 0 || cols <= 0 {
		err = fmt.Errorf("matrix dimensions must be positive: rows=%d cols=%d", rows, cols)
		return nil, err
	}
	if uint64(rows) > math.MaxUint64/uint64(cols) {
		err = fmt.Errorf("matrix dimensions are too large: rows=%d cols=%d", rows, cols)
		return nil, err
	}
	valueCount = uint64(rows) * uint64(cols)
	if valueCount > binarySerializationMaxValues || *totalValues > binarySerializationMaxValues-valueCount {
		err = fmt.Errorf("matrix values exceed limit %d", binarySerializationMaxValues)
		return nil, err
	}
	*totalValues += valueCount

	current = &serializedMatrix{
		Rows:   rows,
		Cols:   cols,
		Values: make([]float32, 0, min(int(valueCount), len(buffer)/4)),
	}
	for valuesRead < valueCount {
		chunkValues = len(buffer) / 4
		if remaining := valueCount - valuesRead; remaining < uint64(chunkValues) {
			chunkValues = int(remaining)
		}
		if _, err = io.ReadFull(reader, buffer[:chunkValues*4]); err != nil {
			return nil, err
		}
		for bufferIndex = 0; bufferIndex < chunkValues; bufferIndex++ {
			value = math.Float32frombits(
				binary.LittleEndian.Uint32(buffer[bufferIndex*4 : bufferIndex*4+4]),
			)
			current.Values = append(current.Values, value)
		}
		valuesRead += uint64(chunkValues)
	}

	return current, nil
}

func writeBinaryString(writer io.Writer, value string) (err error) {
	if len(value) > binarySerializationMaxString {
		err = fmt.Errorf("string length %d exceeds limit %d", len(value), binarySerializationMaxString)
		return err
	}
	if err = writeBinaryUint(writer, uint64(len(value))); err != nil {
		return err
	}
	err = writeBinaryBytes(writer, []byte(value))
	return err
}

func readBinaryString(reader io.Reader) (value string, err error) {
	var (
		length uint64
		bytes  []byte
	)

	if length, err = readBinaryUint(reader); err != nil {
		return "", err
	}
	if length > binarySerializationMaxString {
		err = fmt.Errorf("string length %d exceeds limit %d", length, binarySerializationMaxString)
		return "", err
	}
	bytes = make([]byte, int(length))
	if _, err = io.ReadFull(reader, bytes); err != nil {
		return "", err
	}
	value = string(bytes)
	return value, nil
}

func writeBinaryNonnegativeInt(writer io.Writer, value int) (err error) {
	if value < 0 {
		err = fmt.Errorf("negative integer %d", value)
		return err
	}
	err = writeBinaryUint(writer, uint64(value))
	return err
}

func readBinaryInt(reader io.Reader) (value int, err error) {
	var encoded uint64

	if encoded, err = readBinaryUint(reader); err != nil {
		return 0, err
	}
	if encoded > uint64(maxIntValue()) {
		err = fmt.Errorf("integer %d exceeds platform limit", encoded)
		return 0, err
	}
	value = int(encoded)
	return value, nil
}

func writeBinaryUint(writer io.Writer, value uint64) (err error) {
	var (
		buffer [binary.MaxVarintLen64]byte
		length int
	)

	length = binary.PutUvarint(buffer[:], value)
	err = writeBinaryBytes(writer, buffer[:length])
	return err
}

func readBinaryUint(reader io.Reader) (value uint64, err error) {
	var (
		current [1]byte
		shift   uint
		index   int
	)

	for index = 0; index < binary.MaxVarintLen64; index++ {
		if _, err = io.ReadFull(reader, current[:]); err != nil {
			return 0, err
		}
		if current[0] < 0x80 {
			if index == binary.MaxVarintLen64-1 && current[0] > 1 {
				err = errors.New("varint overflows uint64")
				return 0, err
			}
			if index > 0 && current[0] == 0 {
				err = errors.New("varint is not canonical")
				return 0, err
			}
			value |= uint64(current[0]) << shift
			return value, nil
		}
		value |= uint64(current[0]&0x7f) << shift
		shift += 7
	}

	err = errors.New("varint overflows uint64")
	return 0, err
}

func writeBinaryFloat32(writer io.Writer, value float32) (err error) {
	var buffer [4]byte

	binary.LittleEndian.PutUint32(buffer[:], math.Float32bits(value))
	err = writeBinaryBytes(writer, buffer[:])
	return err
}

func readBinaryFloat32(reader io.Reader) (value float32, err error) {
	var buffer [4]byte

	if _, err = io.ReadFull(reader, buffer[:]); err != nil {
		return 0, err
	}
	value = math.Float32frombits(binary.LittleEndian.Uint32(buffer[:]))
	return value, nil
}

func writeBinaryUint16(writer io.Writer, value uint16) (err error) {
	var buffer [2]byte

	binary.LittleEndian.PutUint16(buffer[:], value)
	err = writeBinaryBytes(writer, buffer[:])
	return err
}

func readBinaryUint16(reader io.Reader) (value uint16, err error) {
	var buffer [2]byte

	if _, err = io.ReadFull(reader, buffer[:]); err != nil {
		return 0, err
	}
	value = binary.LittleEndian.Uint16(buffer[:])
	return value, nil
}

func writeBinaryBytes(writer io.Writer, value []byte) (err error) {
	var written int

	if written, err = writer.Write(value); err != nil {
		return err
	}
	if written != len(value) {
		err = io.ErrShortWrite
		return err
	}
	return nil
}

func ensureBinaryEOF(reader io.Reader) (err error) {
	var (
		extra [1]byte
		read  int
	)

	if read, err = io.ReadFull(reader, extra[:]); read > 0 {
		err = errors.New("binary model contains trailing data")
		return err
	}
	if errors.Is(err, io.EOF) {
		return nil
	}
	err = fmt.Errorf("read trailing data: %w", err)
	return err
}

func maxIntValue() (value int) {
	value = int(^uint(0) >> 1)
	return value
}
