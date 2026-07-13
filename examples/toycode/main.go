package main

import (
	"fmt"
	"log"
	"math/rand"
	"strings"

	"github.com/itsmontoya/neuralnetwork/activation"
	"github.com/itsmontoya/neuralnetwork/data"
	"github.com/itsmontoya/neuralnetwork/internal/f32"
	"github.com/itsmontoya/neuralnetwork/layer"
	"github.com/itsmontoya/neuralnetwork/loss"
	"github.com/itsmontoya/neuralnetwork/matrix"
	"github.com/itsmontoya/neuralnetwork/metric"
	"github.com/itsmontoya/neuralnetwork/model"
	"github.com/itsmontoya/neuralnetwork/optimizer"
)

const (
	padToken        = "<pad>"
	endToken        = "<end>"
	contextSize     = 6
	epochs          = 900
	logInterval     = 150
	batchSize       = 24
	hiddenSize      = 72
	learningRate    = 0.02
	generationLimit = 48
	temperature     = 0.75
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	var (
		tokens         []string
		tokenIndexes   map[string]int
		sequences      [][]int
		trainingData   *data.Dataset
		network        *model.Sequential
		optimizerRule  optimizer.Optimizer
		shuffleRandom  *rand.Rand
		modelRandom    *rand.Rand
		samplingRandom *rand.Rand
		history        model.TrainingHistory
		accuracyMetric metric.CategoricalAccuracy
		finalMetrics   model.EpochMetrics
		generated      []string
	)

	tokens = vocabulary()
	tokenIndexes = indexTokens(tokens)

	if sequences, err = encodedPrograms(tokenIndexes); err != nil {
		return err
	}

	if trainingData, err = newNextTokenDataset(sequences, len(tokens), tokenIndexes[padToken], tokenIndexes[endToken]); err != nil {
		return err
	}

	modelRandom = rand.New(rand.NewSource(101))
	if network, err = newToyCodeModel(len(tokens), modelRandom); err != nil {
		return err
	}

	if optimizerRule, err = optimizer.NewAdam(learningRate); err != nil {
		return err
	}

	shuffleRandom = rand.New(rand.NewSource(103))
	history, err = network.Fit(trainingData, model.FitConfig{
		Epochs:    epochs,
		BatchSize: batchSize,
		Shuffle:   true,
		Random:    shuffleRandom,
		Optimizer: optimizerRule,
		Loss:      loss.CategoricalCrossEntropy{},
		Accuracy:  accuracyMetric.Value,
		Callback:  printEpochMetrics,
	})
	if err != nil {
		return err
	}

	finalMetrics = history.Epochs[len(history.Epochs)-1]
	fmt.Printf("final next-token loss %.6f accuracy %.3f\n", finalMetrics.Loss, finalMetrics.Accuracy)
	fmt.Println()

	samplingRandom = rand.New(rand.NewSource(107))
	if generated, err = generateProgram(network, tokens, tokenIndexes, samplingRandom); err != nil {
		return err
	}

	fmt.Println("generated toy code:")
	fmt.Print(formatToyCode(generated))
	return nil
}

func vocabulary() (tokens []string) {
	tokens = []string{
		padToken,
		endToken,
		"fn",
		"(",
		")",
		"{",
		"}",
		",",
		";",
		"let",
		"ret",
		"if",
		"out",
		"a",
		"b",
		"x",
		"one",
		"two",
		"add",
		"sub",
		"mul",
		"inc",
		"dec",
		"double",
		"square",
		"max",
		"min",
		"=",
		"+",
		"-",
		"*",
		">",
		"<",
	}
	return tokens
}

func indexTokens(tokens []string) (indexes map[string]int) {
	var (
		index int
		token string
	)

	indexes = make(map[string]int, len(tokens))
	for index, token = range tokens {
		indexes[token] = index
	}

	return indexes
}

func encodedPrograms(tokenIndexes map[string]int) (sequences [][]int, err error) {
	var (
		program  []string
		programs [][]string
		sequence []int
	)

	programs = toyPrograms()
	sequences = make([][]int, 0, len(programs))
	for _, program = range programs {
		if sequence, err = encodeProgram(program, tokenIndexes); err != nil {
			return nil, err
		}

		sequences = append(sequences, sequence)
	}

	return sequences, nil
}

func encodeProgram(program []string, tokenIndexes map[string]int) (sequence []int, err error) {
	var (
		token string
		index int
		ok    bool
	)

	sequence = make([]int, 0, len(program))
	for _, token = range program {
		index, ok = tokenIndexes[token]
		if !ok {
			err = fmt.Errorf("toycode: unknown token %q", token)
			return nil, err
		}

		sequence = append(sequence, index)
	}

	return sequence, nil
}

func toyPrograms() (programs [][]string) {
	programs = [][]string{
		{"fn", "add", "(", "a", ",", "b", ")", "{", "let", "out", "=", "a", "+", "b", ";", "ret", "out", ";", "}"},
		{"fn", "sub", "(", "a", ",", "b", ")", "{", "let", "out", "=", "a", "-", "b", ";", "ret", "out", ";", "}"},
		{"fn", "mul", "(", "a", ",", "b", ")", "{", "let", "out", "=", "a", "*", "b", ";", "ret", "out", ";", "}"},
		{"fn", "inc", "(", "x", ")", "{", "let", "out", "=", "x", "+", "one", ";", "ret", "out", ";", "}"},
		{"fn", "dec", "(", "x", ")", "{", "let", "out", "=", "x", "-", "one", ";", "ret", "out", ";", "}"},
		{"fn", "double", "(", "x", ")", "{", "let", "out", "=", "x", "*", "two", ";", "ret", "out", ";", "}"},
		{"fn", "square", "(", "x", ")", "{", "let", "out", "=", "x", "*", "x", ";", "ret", "out", ";", "}"},
		{"fn", "max", "(", "a", ",", "b", ")", "{", "if", "a", ">", "b", "{", "ret", "a", ";", "}", "ret", "b", ";", "}"},
		{"fn", "min", "(", "a", ",", "b", ")", "{", "if", "a", "<", "b", "{", "ret", "a", ";", "}", "ret", "b", ";", "}"},
	}
	return programs
}

func newNextTokenDataset(sequences [][]int, vocabSize, padIndex, endIndex int) (dataset *data.Dataset, err error) {
	var (
		inputValues  []float32
		targetValues []float32
		context      []int
		sequence     []int
		token        int
		rowCount     int
		inputs       *matrix.Matrix
		targets      *matrix.Matrix
	)

	for _, sequence = range sequences {
		context = paddedContext(padIndex)
		for _, token = range sequence {
			inputValues = appendContext(inputValues, context, vocabSize)
			targetValues = appendOneHot(targetValues, token, vocabSize)
			context = shiftedContext(context, token)
			rowCount++
		}

		inputValues = appendContext(inputValues, context, vocabSize)
		targetValues = appendOneHot(targetValues, endIndex, vocabSize)
		rowCount++
	}

	if inputs, err = matrix.FromSlice(rowCount, contextSize*vocabSize, inputValues); err != nil {
		return nil, err
	}

	if targets, err = matrix.FromSlice(rowCount, vocabSize, targetValues); err != nil {
		return nil, err
	}

	dataset, err = data.NewDataset(inputs, targets)
	return dataset, err
}

func paddedContext(padIndex int) (context []int) {
	var index int

	context = make([]int, contextSize)
	for index = range context {
		context[index] = padIndex
	}

	return context
}

func shiftedContext(context []int, token int) (next []int) {
	next = make([]int, len(context))
	copy(next, context[1:])
	next[len(next)-1] = token
	return next
}

func appendContext(values []float32, context []int, vocabSize int) (next []float32) {
	var (
		token int
		index int
	)

	next = values
	for _, token = range context {
		for index = 0; index < vocabSize; index++ {
			if index == token {
				next = append(next, 1)
				continue
			}

			next = append(next, 0)
		}
	}

	return next
}

func appendOneHot(values []float32, token, vocabSize int) (next []float32) {
	var index int

	next = values
	for index = 0; index < vocabSize; index++ {
		if index == token {
			next = append(next, 1)
			continue
		}

		next = append(next, 0)
	}

	return next
}

func newToyCodeModel(vocabSize int, random *rand.Rand) (network *model.Sequential, err error) {
	var (
		first            *layer.Dense
		firstActivation  *layer.Activation
		second           *layer.Dense
		secondActivation *layer.Activation
		output           *layer.Dense
		outputActivation *layer.Activation
	)

	if first, err = layer.NewDense(contextSize*vocabSize, hiddenSize, layer.XavierUniformWeights(random)); err != nil {
		return nil, err
	}

	if firstActivation, err = layer.NewActivation(activation.Tanh{}); err != nil {
		return nil, err
	}

	if second, err = layer.NewDense(hiddenSize, hiddenSize, layer.XavierUniformWeights(random)); err != nil {
		return nil, err
	}

	if secondActivation, err = layer.NewActivation(activation.Tanh{}); err != nil {
		return nil, err
	}

	if output, err = layer.NewDense(hiddenSize, vocabSize, layer.XavierUniformWeights(random)); err != nil {
		return nil, err
	}

	if outputActivation, err = layer.NewActivation(activation.Softmax{}); err != nil {
		return nil, err
	}

	network, err = model.NewSequential(first, firstActivation, second, secondActivation, output, outputActivation)
	return network, err
}

func printEpochMetrics(metrics model.EpochMetrics) (err error) {
	if metrics.Epoch == 1 || metrics.Epoch%logInterval == 0 || metrics.Epoch == epochs {
		fmt.Printf("epoch %4d next-token loss %.6f accuracy %.3f\n", metrics.Epoch, metrics.Loss, metrics.Accuracy)
	}

	return nil
}

func generateProgram(network *model.Sequential, tokens []string, tokenIndexes map[string]int, random *rand.Rand) (generated []string, err error) {
	var (
		context          []int
		inputValues      []float32
		input            *matrix.Matrix
		predictions      *matrix.Matrix
		predictionValues []float32
		nextToken        int
		step             int
	)

	context = paddedContext(tokenIndexes[padToken])
	nextToken = tokenIndexes["fn"]
	generated = []string{"fn"}
	context = shiftedContext(context, nextToken)

	for step = 0; step < generationLimit; step++ {
		inputValues = appendContext(nil, context, len(tokens))
		if input, err = matrix.FromSlice(1, contextSize*len(tokens), inputValues); err != nil {
			return nil, err
		}

		if predictions, err = network.Predict(input); err != nil {
			return nil, err
		}

		if predictionValues, err = predictions.Values(); err != nil {
			return nil, err
		}

		nextToken = sampleToken(predictionValues, tokenIndexes[padToken], random)
		if tokens[nextToken] == endToken {
			return generated, nil
		}

		generated = append(generated, tokens[nextToken])
		context = shiftedContext(context, nextToken)
	}

	return generated, nil
}

func sampleToken(probabilities []float32, blockedToken int, random *rand.Rand) (token int) {
	var (
		index       int
		probability float32
		scaled      []float32
		total       float32
		threshold   float32
		running     float32
	)

	scaled = make([]float32, len(probabilities))
	for index, probability = range probabilities {
		if index == blockedToken || probability <= 0 {
			continue
		}

		scaled[index] = f32.Pow(probability, 1/temperature)
		total += scaled[index]
	}

	if total == 0 {
		return argmax(probabilities)
	}

	threshold = float32(random.Float64()) * total
	for index, probability = range scaled {
		running += probability
		if running >= threshold {
			return index
		}
	}

	token = argmax(probabilities)
	return token
}

func argmax(values []float32) (index int) {
	var (
		current int
		value   float32
		best    float32
	)

	best = values[0]
	for current = 1; current < len(values); current++ {
		value = values[current]
		if value <= best {
			continue
		}

		best = value
		index = current
	}

	return index
}

func formatToyCode(tokens []string) (code string) {
	var (
		builder strings.Builder
		line    []string
		indent  int
		token   string
	)

	for _, token = range tokens {
		switch token {
		case "{":
			writeLine(&builder, indent, append(line, "{"))
			line = nil
			indent++
		case "}":
			if len(line) > 0 {
				writeLine(&builder, indent, line)
				line = nil
			}

			if indent > 0 {
				indent--
			}

			writeLine(&builder, indent, []string{"}"})
		case ";":
			writeLine(&builder, indent, append(line, ";"))
			line = nil
		default:
			line = append(line, token)
		}
	}

	if len(line) > 0 {
		writeLine(&builder, indent, line)
	}

	code = builder.String()
	return code
}

func writeLine(builder *strings.Builder, indent int, tokens []string) {
	var index int

	for index = 0; index < indent; index++ {
		builder.WriteString("  ")
	}

	builder.WriteString(formatLine(tokens))
	builder.WriteString("\n")
}

func formatLine(tokens []string) (line string) {
	line = strings.Join(tokens, " ")
	line = strings.ReplaceAll(line, "( ", "(")
	line = strings.ReplaceAll(line, " )", ")")
	line = strings.ReplaceAll(line, " ,", ",")
	line = strings.ReplaceAll(line, " ;", ";")
	return line
}
