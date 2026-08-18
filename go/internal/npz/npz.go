// Package npz reads NumPy .npz archives (zip of .npy members) — just enough
// for the nina canvas/phrase packs: C-order float32/float64 arrays of rank
// 1-2, little-endian.
package npz

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"regexp"
	"strings"
)

// Array is a dense row-major float32 matrix (rank-1 arrays get Cols=1... no:
// rank-1 keeps Rows=len, Cols=1 semantics documented per accessor).
type Array struct {
	Shape []int
	Data  []float32 // C-order
}

func (a *Array) Rows() int {
	if len(a.Shape) == 0 {
		return 0
	}
	return a.Shape[0]
}

func (a *Array) Cols() int {
	if len(a.Shape) < 2 {
		return 1
	}
	return a.Shape[1]
}

// Row returns a view of row i (rank-2) — no copy.
func (a *Array) Row(i int) []float32 {
	c := a.Cols()
	return a.Data[i*c : (i+1)*c]
}

var headerRe = regexp.MustCompile(
	`'descr':\s*'([^']+)'.*'fortran_order':\s*(\w+).*'shape':\s*\(([^)]*)\)`)

func parseNpy(r io.Reader) (*Array, error) {
	var magic [8]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, err
	}
	if !bytes.Equal(magic[:6], []byte("\x93NUMPY")) {
		return nil, fmt.Errorf("npz: bad npy magic")
	}
	var hlen uint32
	if magic[6] == 1 {
		var h16 uint16
		if err := binary.Read(r, binary.LittleEndian, &h16); err != nil {
			return nil, err
		}
		hlen = uint32(h16)
	} else {
		if err := binary.Read(r, binary.LittleEndian, &hlen); err != nil {
			return nil, err
		}
	}
	hdr := make([]byte, hlen)
	if _, err := io.ReadFull(r, hdr); err != nil {
		return nil, err
	}
	m := headerRe.FindStringSubmatch(string(hdr))
	if m == nil {
		return nil, fmt.Errorf("npz: unparsable npy header: %s", hdr)
	}
	descr, fortran, shapeStr := m[1], m[2], m[3]
	if fortran != "False" {
		return nil, fmt.Errorf("npz: fortran-order arrays unsupported")
	}
	shape := []int{}
	n := 1
	for _, tok := range strings.Split(shapeStr, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		var d int
		if _, err := fmt.Sscanf(tok, "%d", &d); err != nil {
			return nil, err
		}
		shape = append(shape, d)
		n *= d
	}
	out := &Array{Shape: shape, Data: make([]float32, n)}
	switch descr {
	case "<f4":
		raw := make([]byte, n*4)
		if _, err := io.ReadFull(r, raw); err != nil {
			return nil, err
		}
		for i := 0; i < n; i++ {
			out.Data[i] = math.Float32frombits(
				binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case "<f8":
		raw := make([]byte, n*8)
		if _, err := io.ReadFull(r, raw); err != nil {
			return nil, err
		}
		for i := 0; i < n; i++ {
			out.Data[i] = float32(math.Float64frombits(
				binary.LittleEndian.Uint64(raw[i*8:])))
		}
	default:
		return nil, fmt.Errorf("npz: dtype %s unsupported", descr)
	}
	return out, nil
}

// Load reads all members of an .npz (member name minus ".npy" → array).
func Load(path string) (map[string]*Array, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, err
	}
	defer zr.Close()
	out := map[string]*Array{}
	for _, f := range zr.File {
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		a, err := parseNpy(rc)
		rc.Close()
		if err != nil {
			return nil, fmt.Errorf("%s/%s: %w", path, f.Name, err)
		}
		out[strings.TrimSuffix(f.Name, ".npy")] = a
	}
	return out, nil
}
