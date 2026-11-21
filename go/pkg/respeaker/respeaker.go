// Package respeaker reads and sets the ReSpeaker 4 Mic Array's XVF-3000 parameters (its AGC, noise
// suppression, echo canceller, high-pass filter) over USB vendor control transfers, the protocol of
// Seeed's tuning.py, in plain Go: the usbdevfs ioctl on /dev/bus/usb/BBB/DDD, no libusb, no cgo.
//
// The parameters live in the array's RAM: a power cycle or replug resets them to the firmware's
// defaults, so whoever needs them applies them at start (fsttm-go does, from its config's
// `respeaker:` block). Reading is harmless; the device node must be readable/writable by the user
// (deploy/nuc/audio/99-respeaker.rules).
package respeaker

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"unsafe"
)

const (
	VendorID  = 0x2886
	ProductID = 0x0018
)

// Param is one XVF-3000 parameter: its id and offset in the control protocol, its type and range.
type Param struct {
	ID, Offset int
	Int        bool
	Max, Min   float64
	Writable   bool
	What       string
}

// Params is tuning.py's table (the ids and offsets the firmware answers to).
var Params = map[string]Param{
	"AECFREEZEONOFF":       {18, 7, true, 1, 0, true, "adaptive echo canceller updates inhibited"},
	"AECNORM":              {18, 19, false, 16, 0.25, true, "echo canceller step size"},
	"AECPATHCHANGE":        {18, 25, true, 1, 0, false, "echo path change detected"},
	"RT60":                 {18, 26, false, 0.9, 0.25, false, "estimated reverberation time"},
	"HPFONOFF":             {18, 27, true, 3, 0, true, "high-pass filter (0 off, 1 70 Hz, 2 125 Hz, 3 180 Hz)"},
	"RT60ONOFF":            {18, 28, true, 1, 0, true, "RT60 estimation"},
	"AECSILENCELEVEL":      {18, 30, false, 1, 1e-9, true, "echo canceller silence threshold"},
	"AECSILENCEMODE":       {18, 31, true, 1, 0, false, "echo canceller sees silence"},
	"AGCONOFF":             {19, 0, true, 1, 0, true, "automatic gain control"},
	"AGCMAXGAIN":           {19, 1, false, 1000, 1, true, "AGC maximum gain"},
	"AGCDESIREDLEVEL":      {19, 2, false, 0.99, 1e-8, true, "AGC target level"},
	"AGCGAIN":              {19, 3, false, 1000, 1, true, "AGC current gain"},
	"AGCTIME":              {19, 4, false, 1, 0.1, true, "AGC time constant (s)"},
	"CNIONOFF":             {19, 5, true, 1, 0, true, "comfort noise insertion"},
	"FREEZEONOFF":          {19, 6, true, 1, 0, true, "beamformer adaptation frozen"},
	"STATNOISEONOFF":       {19, 8, true, 1, 0, true, "stationary noise suppression"},
	"GAMMA_NS":             {19, 9, false, 3, 0, true, "stationary noise over-subtraction"},
	"MIN_NS":               {19, 10, false, 1, 0, true, "stationary noise gain floor"},
	"NONSTATNOISEONOFF":    {19, 11, true, 1, 0, true, "non-stationary noise suppression"},
	"GAMMA_NN":             {19, 12, false, 3, 0, true, "non-stationary noise over-subtraction"},
	"MIN_NN":               {19, 13, false, 1, 0, true, "non-stationary noise gain floor"},
	"ECHOONOFF":            {19, 14, true, 1, 0, true, "echo suppression"},
	"GAMMA_E":              {19, 15, false, 3, 0, true, "direct echo over-subtraction"},
	"GAMMA_ETAIL":          {19, 16, false, 3, 0, true, "echo tail over-subtraction"},
	"GAMMA_ENL":            {19, 17, false, 5, 0, true, "non-linear echo over-subtraction"},
	"NLATTENONOFF":         {19, 18, true, 1, 0, true, "non-linear echo attenuation"},
	"NLAEC_MODE":           {19, 20, true, 2, 0, true, "non-linear AEC mode"},
	"SPEECHDETECTED":       {19, 22, true, 1, 0, false, "speech detected (ASR channel)"},
	"FSBUPDATED":           {19, 23, true, 1, 0, false, "beamformer updated"},
	"FSBPATHCHANGE":        {19, 24, true, 1, 0, false, "beamformer path change"},
	"TRANSIENTONOFF":       {19, 29, true, 1, 0, true, "transient suppression"},
	"VOICEACTIVITY":        {19, 32, true, 1, 0, false, "voice activity (on-chip VAD)"},
	"STATNOISEONOFF_SR":    {19, 33, true, 1, 0, true, "stationary noise suppression, ASR channel"},
	"NONSTATNOISEONOFF_SR": {19, 34, true, 1, 0, true, "non-stationary noise suppression, ASR channel"},
	"GAMMA_NS_SR":          {19, 35, false, 3, 0, true, "stationary over-subtraction, ASR channel"},
	"GAMMA_NN_SR":          {19, 36, false, 3, 0, true, "non-stationary over-subtraction, ASR channel"},
	"MIN_NS_SR":            {19, 37, false, 1, 0, true, "stationary gain floor, ASR channel"},
	"MIN_NN_SR":            {19, 38, false, 1, 0, true, "non-stationary gain floor, ASR channel"},
	"GAMMAVAD_SR":          {19, 39, false, 1000, 0, true, "on-chip VAD threshold, ASR channel"},
	"DOAANGLE":             {21, 0, true, 359, 0, false, "direction of arrival (degrees)"},
}

// Names is the table's names in order.
func Names() []string {
	n := make([]string, 0, len(Params))
	for k := range Params {
		n = append(n, k)
	}
	sort.Strings(n)
	return n
}

// Device is an open array.
type Device struct{ f *os.File }

// Find returns the device node of the first array on the bus (/dev/bus/usb/BBB/DDD).
func Find() (string, error) {
	ms, _ := filepath.Glob("/sys/bus/usb/devices/*/idVendor")
	for _, m := range ms {
		dir := filepath.Dir(m)
		if hexIs(dir+"/idVendor", VendorID) && hexIs(dir+"/idProduct", ProductID) {
			bus, e1 := readInt(dir + "/busnum")
			dev, e2 := readInt(dir + "/devnum")
			if e1 == nil && e2 == nil {
				return fmt.Sprintf("/dev/bus/usb/%03d/%03d", bus, dev), nil
			}
		}
	}
	return "", fmt.Errorf("no ReSpeaker 4 Mic Array (%04x:%04x) on USB", VendorID, ProductID)
}

func hexIs(path string, want int) bool {
	b, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	v, err := strconv.ParseInt(strings.TrimSpace(string(b)), 16, 32)
	return err == nil && int(v) == want
}

func readInt(path string) (int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(strings.TrimSpace(string(b)))
}

// Open opens the first array found.
func Open() (*Device, error) {
	node, err := Find()
	if err != nil {
		return nil, err
	}
	f, err := os.OpenFile(node, os.O_RDWR, 0)
	if err != nil {
		return nil, fmt.Errorf("%s: %w (deploy/nuc/audio/99-respeaker.rules gives the user access)", node, err)
	}
	return &Device{f: f}, nil
}

func (d *Device) Close() error { return d.f.Close() }

// usbdevfs_ctrltransfer (linux/usbdevice_fs.h), 64-bit layout.
type ctrl struct {
	RequestType uint8
	Request     uint8
	Value       uint16
	Index       uint16
	Length      uint16
	Timeout     uint32
	Data        uintptr
}

const usbdevfsControl = 0xC0185500 // _IOWR('U', 0, struct usbdevfs_ctrltransfer) on 64-bit

func (d *Device) control(reqType uint8, value, index uint16, data []byte) error {
	c := ctrl{RequestType: reqType, Request: 0, Value: value, Index: index, Length: uint16(len(data)), Timeout: 1000}
	if len(data) > 0 {
		c.Data = uintptr(unsafe.Pointer(&data[0]))
	}
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, d.f.Fd(), usbdevfsControl, uintptr(unsafe.Pointer(&c)))
	if errno != 0 {
		return errno
	}
	return nil
}

// Read reads one parameter.
func (d *Device) Read(name string) (float64, error) {
	p, ok := Params[name]
	if !ok {
		return 0, fmt.Errorf("no parameter %s", name)
	}
	cmd := 0x80 | p.Offset
	if p.Int {
		cmd |= 0x40
	}
	buf := make([]byte, 8)
	if err := d.control(0xC0, uint16(cmd), uint16(p.ID), buf); err != nil {
		return 0, fmt.Errorf("read %s: %w", name, err)
	}
	a := int32(binary.LittleEndian.Uint32(buf[0:]))
	b := int32(binary.LittleEndian.Uint32(buf[4:]))
	if p.Int {
		return float64(a), nil
	}
	return float64(a) * math.Pow(2, float64(b)), nil
}

// Write sets one parameter (until the array is power-cycled).
func (d *Device) Write(name string, v float64) error {
	p, ok := Params[name]
	if !ok {
		return fmt.Errorf("no parameter %s", name)
	}
	if !p.Writable {
		return fmt.Errorf("%s is read-only", name)
	}
	if v < p.Min || v > p.Max {
		return fmt.Errorf("%s takes %g..%g, not %g", name, p.Min, p.Max, v)
	}
	buf := make([]byte, 12)
	binary.LittleEndian.PutUint32(buf[0:], uint32(p.Offset))
	if p.Int {
		binary.LittleEndian.PutUint32(buf[4:], uint32(int32(v)))
		binary.LittleEndian.PutUint32(buf[8:], 1)
	} else {
		binary.LittleEndian.PutUint32(buf[4:], math.Float32bits(float32(v)))
		binary.LittleEndian.PutUint32(buf[8:], 0)
	}
	if err := d.control(0x40, 0, uint16(p.ID), buf); err != nil {
		return fmt.Errorf("write %s: %w", name, err)
	}
	return nil
}

// Apply sets each parameter of settings and returns one line per setting as read back.
func (d *Device) Apply(settings map[string]float64) ([]string, error) {
	keys := make([]string, 0, len(settings))
	for k := range settings {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var out []string
	for _, k := range keys {
		if err := d.Write(k, settings[k]); err != nil {
			return out, err
		}
		v, err := d.Read(k)
		if err != nil {
			return out, err
		}
		out = append(out, fmt.Sprintf("%s=%.4g", k, v))
	}
	return out, nil
}
