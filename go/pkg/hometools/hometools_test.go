package hometools

import (
	"encoding/json"
	"testing"
)

type rules struct{ got []string }

func (r *rules) Tools() []Tool {
	return []Tool{{Type: "function", Function: Function{Name: "home_air", Description: "air",
		Parameters: json.RawMessage(`{"type":"object"}`)}}}
}

func (r *rules) Call(domain string, args map[string]any) Outcome {
	r.got = append(r.got, domain+":"+args["intent"].(string))
	return Outcome{Status: "refused", Text: "the lab ventilation is stopped."}
}

func TestClientServer(t *testing.T) {
	r := &rules{}
	ln, err := Serve("127.0.0.1:0", r)
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	c := &Client{Addr: ln.Addr().String()}
	tl := c.Tools()
	if len(tl) != 1 {
		t.Fatalf("tools %s", tl)
	}
	o, line := c.Call("home_air", map[string]any{"intent": "NIGHT", "device": "lab ventilation"})
	if o.Status != "refused" || r.got[0] != "air:NIGHT" || line == "" {
		t.Fatalf("%+v %q %v", o, line, r.got)
	}
	ln.Close()
	if o, _ := c.Call("air", map[string]any{"intent": "ON"}); o.Status != "error" {
		t.Fatalf("closed port: %+v", o)
	}
}
