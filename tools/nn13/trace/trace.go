package trace

import (
	"reflect"
	"unsafe"
)

var (
	enabled     bool
	sequence    int
	frames      []frame
	allocations map[uintptr]allocation
	live        int64
	result      Snapshot
)

// Begin starts an observation, including preexisting input/constant storage.
// It must not overlap another observation or run concurrently with any tracing.
func Begin(roots ...any) {
	if enabled {
		panic("nn13trace: overlapping observation")
	}

	sequence, live = 0, 0
	frames = []frame{{id: 0, site: "roots"}}
	allocations = make(map[uintptr]allocation)
	result = Snapshot{Copies: make(map[string]int64), Peak: make(map[string]int64)}
	enabled = true
	visitRoots(roots, func(value reflect.Value, name string) {
		if payloadField(name) && value.Len() != 0 {
			register(value.Pointer(), int64(value.Cap())*int64(value.Type().Elem().Size()), value)
		}
	})
	result.AllocatedBytes = 0
}

// End completes one observation. No instrumented stack frame may remain.
func End() (out Snapshot) {
	if !enabled || len(frames) != 1 {
		panic("nn13trace: unbalanced observation")
	}

	enabled = false
	out = result
	allocations, frames = nil, nil
	return out
}

// Enter records a source function's allocation ownership scope.
func Enter(site string) {
	if !enabled {
		return
	}

	sequence++
	frames = append(frames, frame{id: sequence, site: site})
}

// Leave preserves allocations reachable through results or caller-owned
// arguments/receivers and releases the remaining function-local payloads.
func Leave(roots ...any) {
	if !enabled {
		return
	}

	if len(frames) < 2 {
		panic("nn13trace: stack underflow")
	}

	current := frames[len(frames)-1].id
	parent := frames[len(frames)-2].id
	retained := make(map[uintptr]bool)
	visitRoots(roots, func(value reflect.Value, name string) {
		if value.Len() != 0 {
			if base, ok := find(value.Pointer()); ok {
				retained[base] = true
			}
		}
	})
	for base, item := range allocations {
		if item.frame != current {
			continue
		}

		if retained[base] {
			item.frame = parent
			allocations[base] = item
		} else {
			live -= item.bytes
			delete(allocations, base)
		}
	}

	frames = frames[:len(frames)-1]
}

// Make instruments a numeric payload allocation. Metadata allocations do not
// use this helper. A scope retains backing capacity, including for subslices.
func Make[T any](length int, capacity ...int) (values []T) {
	if len(capacity) == 0 {
		values = make([]T, length)
	} else {
		values = make([]T, length, capacity[0])
	}
	Track(values)
	return values
}

// Track registers a numeric buffer when metadata becomes a Shape result.
func Track[T any](values []T) {
	if !enabled || cap(values) == 0 {
		return
	}

	var element T
	register(uintptr(unsafe.Pointer(unsafe.SliceData(values))), int64(cap(values))*int64(unsafe.Sizeof(element)), values)
}

// Copy counts actual copied payload elements. Shape metadata copies are excluded
// because their destination was not registered as numeric payload storage.
func Copy[T any](destination, source []T) (count int) {
	count = copy(destination, source)
	if !enabled || count == 0 {
		return count
	}

	if _, ok := find(uintptr(unsafe.Pointer(unsafe.SliceData(destination)))); ok {
		var element T
		bytes := int64(count) * int64(unsafe.Sizeof(element))
		result.CopiedBytes += bytes
		result.Copies[frames[len(frames)-1].site] += bytes
	}

	return count
}

func register(base uintptr, bytes int64, hold any) {
	if bytes == 0 {
		return
	}

	if _, ok := find(base); ok {
		return
	}

	current := frames[len(frames)-1]
	allocations[base] = allocation{base: base, bytes: bytes, frame: current.id, site: current.site, hold: hold}
	live += bytes
	result.AllocatedBytes += bytes
	if live > result.PeakBytes {
		result.PeakBytes = live
		result.Peak = make(map[string]int64)
		for _, item := range allocations {
			result.Peak[item.site] += item.bytes
		}
	}
}

func find(pointer uintptr) (base uintptr, found bool) {
	for start, item := range allocations {
		if pointer >= start && uint64(pointer-start) < uint64(item.bytes) {
			return start, true
		}
	}
	return 0, false
}

func payloadField(name string) (payload bool) {
	return name == "floats" || name == "integers" || name == "data" || name == "shapeValues" || name == "FloatData" || name == "Int64Data"
}

func visitRoots(roots []any, visit func(reflect.Value, string)) {
	seen := make(map[uintptr]bool)
	var walk func(reflect.Value, string)
	walk = func(value reflect.Value, name string) {
		if !value.IsValid() {
			return
		}

		switch value.Kind() {
		case reflect.Interface:
			if !value.IsNil() {
				walk(value.Elem(), name)
			}
		case reflect.Pointer:
			if !value.IsNil() && !seen[value.Pointer()] {
				seen[value.Pointer()] = true
				walk(value.Elem(), name)
			}
		case reflect.Struct:
			for index := 0; index < value.NumField(); index++ {
				walk(value.Field(index), value.Type().Field(index).Name)
			}
		case reflect.Slice:
			if value.Type().Elem().Kind() == reflect.Float32 || value.Type().Elem().Kind() == reflect.Int64 {
				visit(value, name)
				return
			}

			for index := 0; index < value.Len(); index++ {
				walk(value.Index(index), name)
			}
		case reflect.Array:
			for index := 0; index < value.Len(); index++ {
				walk(value.Index(index), name)
			}
		case reflect.Map:
			iterator := value.MapRange()
			for iterator.Next() {
				walk(iterator.Value(), name)
			}
		}
	}
	for _, root := range roots {
		walk(reflect.ValueOf(root), "")
	}
}

// ValueCopy counts scalar assignments in explicit transpose materialization.
// Arithmetic/kernel result writes are not counted as payload copies.
func ValueCopy[T any](value T) (out T) {
	if enabled {
		bytes := int64(unsafe.Sizeof(value))
		result.CopiedBytes += bytes
		result.Copies[frames[len(frames)-1].site] += bytes
	}

	return value
}
