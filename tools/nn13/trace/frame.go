package trace

// frame identifies one function ownership scope in the serial observer.
type frame struct {
	id   int
	site string
}
