package trace

import "testing"

func TestScopeOwnershipAndCopy(t *testing.T) {
	var owner struct{ floats []float32 }
	owner.floats = make([]float32, 3)
	Begin(&owner)
	Enter("caller")
	Enter("callee")
	values := Make[float32](5)
	Copy(values[1:3], owner.floats)
	_ = Make[float32](7)
	Leave(values)
	Copy(values, owner.floats)
	Leave()
	out := End()
	if out.CopiedBytes != 20 || out.PeakBytes != 60 || out.AllocatedBytes != 48 {
		t.Fatalf("unexpected scope/copy metrics: %+v", out)
	}
}

func TestReceiverRetentionAndAliasDeduplication(t *testing.T) {
	var owner struct{ integers []int64 }
	Begin()
	Enter("caller")
	Enter("mutate receiver")
	owner.integers = Make[int64](4)
	Track(owner.integers[1:])
	Leave(&owner)
	Enter("later call")
	_ = Make[float32](2)
	Leave()
	Leave(&owner)
	out := End()
	if out.PeakBytes != 40 || out.AllocatedBytes != 40 {
		t.Fatalf("receiver storage was lost or alias double counted: %+v", out)
	}
}

func TestMetadataExclusionAndScalarTransposeCopy(t *testing.T) {
	Begin()
	Enter("transpose")
	metadata := make([]int64, 2)
	Copy(metadata, []int64{1, 2})
	values := Make[float32](1)
	values[0] = ValueCopy(float32(3))
	Leave(values)
	out := End()
	if out.CopiedBytes != 4 || out.PeakBytes != 4 || out.AllocatedBytes != 4 {
		t.Fatalf("metadata counted as payload or scalar copy missed: %+v", out)
	}
}
