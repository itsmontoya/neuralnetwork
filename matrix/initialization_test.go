package matrix_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/itsmontoya/neuralnetwork/matrix"
)

func Test_NewUniform(t *testing.T) {
	var (
		random         *rand.Rand
		expectedRandom *rand.Rand
		got            *matrix.Matrix
		expected       []float64
		index          int
		err            error
	)

	random = rand.New(rand.NewSource(11))
	expectedRandom = rand.New(rand.NewSource(11))
	got, err = matrix.NewUniform(2, 3, -0.25, 0.75, random)
	if err != nil {
		t.Fatalf("NewUniform returned error: %v", err)
	}

	expected = make([]float64, 6)
	for index = range expected {
		expected[index] = -0.25 + expectedRandom.Float64()
	}

	requireMatrixValues(t, got, expected)
}

func Test_NewNormal(t *testing.T) {
	var (
		random         *rand.Rand
		expectedRandom *rand.Rand
		got            *matrix.Matrix
		expected       []float64
		index          int
		err            error
	)

	random = rand.New(rand.NewSource(13))
	expectedRandom = rand.New(rand.NewSource(13))
	got, err = matrix.NewNormal(2, 3, 1.5, 0.25, random)
	if err != nil {
		t.Fatalf("NewNormal returned error: %v", err)
	}

	expected = make([]float64, 6)
	for index = range expected {
		expected[index] = 1.5 + 0.25*expectedRandom.NormFloat64()
	}

	requireMatrixValues(t, got, expected)
}

func Test_NewXavierUniform(t *testing.T) {
	var (
		random         *rand.Rand
		expectedRandom *rand.Rand
		got            *matrix.Matrix
		expected       []float64
		limit          float64
		index          int
		err            error
	)

	random = rand.New(rand.NewSource(17))
	expectedRandom = rand.New(rand.NewSource(17))
	got, err = matrix.NewXavierUniform(2, 3, random)
	if err != nil {
		t.Fatalf("NewXavierUniform returned error: %v", err)
	}

	if got.Rows() != 2 {
		t.Fatalf("Rows() = %d, want 2", got.Rows())
	}

	if got.Cols() != 3 {
		t.Fatalf("Cols() = %d, want 3", got.Cols())
	}

	limit = math.Sqrt(6 / float64(2+3))
	expected = make([]float64, 6)
	for index = range expected {
		expected[index] = -limit + (2 * limit * expectedRandom.Float64())
	}

	requireMatrixValues(t, got, expected)
}

func Test_NewHeNormal(t *testing.T) {
	var (
		random         *rand.Rand
		expectedRandom *rand.Rand
		got            *matrix.Matrix
		expected       []float64
		stddev         float64
		index          int
		err            error
	)

	random = rand.New(rand.NewSource(19))
	expectedRandom = rand.New(rand.NewSource(19))
	got, err = matrix.NewHeNormal(4, 3, random)
	if err != nil {
		t.Fatalf("NewHeNormal returned error: %v", err)
	}

	if got.Rows() != 4 {
		t.Fatalf("Rows() = %d, want 4", got.Rows())
	}

	if got.Cols() != 3 {
		t.Fatalf("Cols() = %d, want 3", got.Cols())
	}

	stddev = math.Sqrt(2 / float64(4))
	expected = make([]float64, 12)
	for index = range expected {
		expected[index] = stddev * expectedRandom.NormFloat64()
	}

	requireMatrixValues(t, got, expected)
}

func Test_RandomInitializers_ValidateRandomSource(t *testing.T) {
	type testcase struct {
		name  string
		build func() (m *matrix.Matrix, err error)
	}

	tests := []testcase{
		{
			name: "uniform",
			build: func() (m *matrix.Matrix, err error) {
				m, err = matrix.NewUniform(1, 1, 0, 1, nil)
				return m, err
			},
		},
		{
			name: "normal",
			build: func() (m *matrix.Matrix, err error) {
				m, err = matrix.NewNormal(1, 1, 0, 1, nil)
				return m, err
			},
		},
		{
			name: "xavier uniform",
			build: func() (m *matrix.Matrix, err error) {
				m, err = matrix.NewXavierUniform(1, 1, nil)
				return m, err
			},
		},
		{
			name: "he normal",
			build: func() (m *matrix.Matrix, err error) {
				m, err = matrix.NewHeNormal(1, 1, nil)
				return m, err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				got *matrix.Matrix
				err error
			)

			got, err = tt.build()
			if err == nil {
				t.Fatal("initializer error = nil, want error")
			}

			if got != nil {
				t.Fatal("initializer returned matrix on error")
			}
		})
	}
}

func Test_RandomInitializers_ValidateConfig(t *testing.T) {
	type testcase struct {
		name  string
		build func(random *rand.Rand) (m *matrix.Matrix, err error)
	}

	tests := []testcase{
		{
			name: "uniform range",
			build: func(random *rand.Rand) (m *matrix.Matrix, err error) {
				m, err = matrix.NewUniform(1, 1, 1, 0, random)
				return m, err
			},
		},
		{
			name: "normal standard deviation",
			build: func(random *rand.Rand) (m *matrix.Matrix, err error) {
				m, err = matrix.NewNormal(1, 1, 0, -1, random)
				return m, err
			},
		},
		{
			name: "xavier fan in",
			build: func(random *rand.Rand) (m *matrix.Matrix, err error) {
				m, err = matrix.NewXavierUniform(0, 1, random)
				return m, err
			},
		},
		{
			name: "he fan out",
			build: func(random *rand.Rand) (m *matrix.Matrix, err error) {
				m, err = matrix.NewHeNormal(1, 0, random)
				return m, err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var (
				random *rand.Rand
				got    *matrix.Matrix
				err    error
			)

			random = rand.New(rand.NewSource(23))
			got, err = tt.build(random)
			if err == nil {
				t.Fatal("initializer error = nil, want error")
			}

			if got != nil {
				t.Fatal("initializer returned matrix on error")
			}
		})
	}
}
