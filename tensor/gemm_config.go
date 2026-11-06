package tensor

// GemmConfig specifies mathematical scaling and transposition for GemmInto.
// Its zero value has zero scaling coefficients; callers supply any defaults.
type GemmConfig struct {
	TransposeA bool
	TransposeB bool
	Alpha      float32
	Beta       float32
}
