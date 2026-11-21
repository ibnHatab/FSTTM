package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"path"
	"strconv"
	"strings"
)

// Config is the `hass` block (fsttm/feeds/hass.py's DEFAULTS; the token is a file).
type Config struct {
	URL        string   `yaml:"url"`
	TokenFile  string   `yaml:"token_file"`
	Domains    []string `yaml:"domains"`
	Exclude    []string `yaml:"exclude"`    // shell patterns on entity_id
	Attributes []string `yaml:"attributes"` // the attributes worth saying, in this order
	// Dump false: the raw entity dump is off, metis-home's semantic world serves ENVIRONMENT (one source,
	// no key twice in the LLM's context); fsttm-hass then idles, kept as the fallback (true, the default)
	Dump *bool `yaml:"dump"`
}

// Dumps: the raw entity dump is on (dump unset or true).
func (c *Config) Dumps() bool { return c.Dump == nil || *c.Dump }

func Defaults() Config {
	return Config{
		URL:       "ws://127.0.0.1:8123/api/websocket",
		TokenFile: "~/.config/fsttm/hass.token",
		Domains: []string{"fan", "light", "switch", "media_player", "climate", "cover", "lock",
			"sensor", "binary_sensor", "vacuum", "water_heater"},
		Exclude: []string{"sensor.sun_*", "*_display_brightness", "sensor.backup_*", "event.*"},
		Attributes: []string{"percentage", "preset_mode", "brightness", "temperature", "current_temperature",
			"hvac_action", "media_title", "source", "volume_level", "unit_of_measurement"},
	}
}

// Wanted: the entity's domain is listed and no exclude pattern matches it.
func (c *Config) Wanted(entityID string) bool {
	domain, _, _ := strings.Cut(entityID, ".")
	ok := false
	for _, d := range c.Domains {
		ok = ok || d == domain
	}
	if !ok {
		return false
	}
	for _, p := range c.Exclude {
		if m, _ := path.Match(p, entityID); m {
			return false
		}
	}
	return true
}

// State is one entity's state as Home Assistant sends it; the attribute values stay raw
// JSON, so a number reads as Home Assistant wrote it (21.5, 40), as Python's str() does.
type State struct {
	EntityID   string                     `json:"entity_id"`
	State      *string                    `json:"state"`
	Attributes map[string]json.RawMessage `json:"attributes"`
}

// pyStr is Python's "%s" of a JSON value; empty says "absent" (None, "", []).
func pyStr(raw json.RawMessage) (string, bool) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" || string(raw) == `""` || string(raw) == "[]" {
		return "", false
	}
	switch raw[0] {
	case '"':
		var s string
		_ = json.Unmarshal(raw, &s)
		return s, true
	case 't':
		return "True", true
	case 'f':
		return "False", true
	case '[', '{':
		var v any
		_ = json.Unmarshal(raw, &v)
		return pyRepr(v), true
	}
	return pyNum(string(raw)), true
}

// pyNum: a JSON number as Python prints the int or float json.loads makes of it.
func pyNum(s string) string {
	if !strings.ContainsAny(s, ".eE") {
		return s
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return s
	}
	if f == math.Trunc(f) && math.Abs(f) < 1e16 {
		return strconv.FormatFloat(f, 'f', 1, 64)
	}
	return strconv.FormatFloat(f, 'g', -1, 64)
}

func pyRepr(v any) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case string:
		return "'" + strings.ReplaceAll(x, "'", `\'`) + "'"
	case float64:
		return pyNum(strconv.FormatFloat(x, 'g', -1, 64))
	case []any:
		p := make([]string, len(x))
		for i, e := range x {
			p[i] = pyRepr(e)
		}
		return "[" + strings.Join(p, ", ") + "]"
	case map[string]any:
		var p []string
		for k, e := range x {
			p = append(p, pyRepr(k)+": "+pyRepr(e))
		}
		return "{" + strings.Join(p, ", ") + "}"
	}
	return fmt.Sprint(v)
}

// Words is an entity's state as one short line: its name, its state, the attributes that
// matter ("lab Supply fan: on (percentage 40%, preset mode night)").
func (c *Config) Words(st State) string {
	a := st.Attributes
	name := st.EntityID
	if s, ok := pyStr(a["friendly_name"]); ok {
		name = s
	}
	value := "None"
	if st.State != nil {
		value = *st.State
	}
	head := name + ": " + value
	if unit, ok := pyStr(a["unit_of_measurement"]); ok && value != "unavailable" && value != "unknown" {
		head += " " + unit
	}
	var extra []string
	for _, k := range c.Attributes {
		raw, present := a[k]
		v, ok := pyStr(raw)
		if k == "unit_of_measurement" || !present || !ok {
			continue
		}
		switch k {
		case "brightness":
			if f, err := strconv.ParseFloat(string(bytes.TrimSpace(raw)), 64); err == nil {
				v = strconv.Itoa(int(math.RoundToEven(f*100/255))) + "%"
			}
		case "percentage":
			v += "%"
		}
		extra = append(extra, strings.ReplaceAll(k, "_", " ")+" "+v)
	}
	if len(extra) > 0 {
		head += " (" + strings.Join(extra, ", ") + ")"
	}
	return head
}
