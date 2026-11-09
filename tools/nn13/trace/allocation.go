package trace

// allocation retains a numeric buffer during its measured ownership scope.
type allocation struct {
	base  uintptr
	bytes int64
	frame int
	site  string
	hold  any
}
