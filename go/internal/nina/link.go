// RobotLink — the jsonl event protocol between fsttm-go (all realtime) and
// nina_relay (a ~100-line event-driven rclpy translator that spawns us).
//
// OUT (our stdout, one JSON object per line, everything else goes to stderr):
//
//	{"ev":"action","name":"STAND_UP"}                            → /nina/action (String)
//	{"ev":"nav_goal","x":4.2,"y":1.7,"z":0.0,"yaw":1.57}         → /nina/nav_goal (map)
//	{"ev":"cmd_vel","wz":0.4}                                    → /cmd_vel (Twist)
//	{"ev":"intent","intent":{...},"voice":"..."}                 → /nina/intent
//	{"ev":"state","dialog":"AWAKE"}                              → /nina/dialog_state
//
// IN (our stdin):
//
//	{"ev":"say","text":"Battery low."}                           ← /nina/say
//
// The relay publishes NOTHING but these topics — the structural enforcement
// of the arbiter doctrine (see canvas.go header).
package nina

import (
	"bufio"
	"encoding/json"
	"io"
	"log"
	"os"
	"sync"
)

// RobotLink is the transport seam: the engine's only mouth toward the
// robot. Two implementations:
//   - Link (this file): jsonl on stdout — laptop/SIL transport, consumed by
//     any relay or test harness;
//   - RosLink (roslink.go, build tag `ros`): rclgo-native — the engine IS
//     the nina_speak ROS node, publishing the contracted topics directly
//     (MerlinDrones/rclgo v0.5.x, Humble).
//
// Whatever the transport, the topic set is fixed by the contract table —
// the arbiter doctrine holds: no publisher beyond the contracted topics
// exists in either implementation.
type RobotLink interface {
	// Action publishes a voice-action NAME on /nina/action. The
	// cmd_arbiter is the ONLY author on /api/sport/request — it maps the
	// name to an api-id, gates on armed state, and refuses anything
	// outside its voice-safe table (stand-up incident 2026-08-19: this
	// seam must never carry a sport request, only a word).
	Action(name string)
	NavGoal(x, y, z, yaw float64)
	CmdVel(wz float64)
	Intent(intentJSON any, voice string)
	DialogState(state string)
}

type Link struct {
	mu  sync.Mutex
	out io.Writer
}

var _ RobotLink = (*Link)(nil)

// NewLink emits events on w (production: os.Stdout — the relay's pipe).
func NewLink(w io.Writer) *Link {
	if w == nil {
		w = os.Stdout
	}
	return &Link{out: w}
}

func (l *Link) emit(v map[string]any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	_, _ = l.out.Write(append(b, '\n'))
}

func (l *Link) Action(name string) {
	l.emit(map[string]any{"ev": "action", "name": name})
}

func (l *Link) NavGoal(x, y, z, yaw float64) {
	l.emit(map[string]any{"ev": "nav_goal", "x": x, "y": y, "z": z, "yaw": yaw})
}

func (l *Link) CmdVel(wz float64) {
	l.emit(map[string]any{"ev": "cmd_vel", "wz": wz})
}

func (l *Link) Intent(intentJSON any, voice string) {
	l.emit(map[string]any{"ev": "intent", "intent": intentJSON, "voice": voice})
}

func (l *Link) DialogState(state string) {
	l.emit(map[string]any{"ev": "state", "dialog": state})
}

// ReadSay consumes stdin-side events; each /nina/say text is handed to
// `announce` (Engine.Announce — spoken only on a free floor). Blocks; run
// as a goroutine. Returns on EOF (relay gone).
func (l *Link) ReadSay(r io.Reader, announce func(string)) {
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		var ev struct {
			Ev   string `json:"ev"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			continue
		}
		if ev.Ev == "say" && ev.Text != "" {
			log.Printf("[nina] /nina/say → announce %q", ev.Text)
			announce(ev.Text)
		}
	}
}
