// RobotLink — the jsonl event protocol between fsttm-go (all realtime) and
// nina_relay (a ~100-line event-driven rclpy translator that spawns us).
//
// OUT (our stdout, one JSON object per line, everything else goes to stderr):
//
//	{"ev":"sport","api_id":1004,"name":"STAND_UP","params":{}}   → /api/sport/request
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
	Sport(apiID int, name string, params map[string]any)
	NavGoal(x, y, z, yaw float64)
	CmdVel(wz float64)
	Intent(intentJSON any, voice string)
	DialogState(state string)
}

type Link struct {
	mu            sync.Mutex
	out           io.Writer
	motionInhibit bool // gate sport/cmd_vel/nav_goal (see RosLink)
}

var _ RobotLink = (*Link)(nil)

// NewLink emits events on w (production: os.Stdout — the relay's pipe).
// armMotion=false (the default via NewLinkSafe) inhibits actuator events.
func NewLink(w io.Writer) *Link { return NewLinkArmed(w, true) }

// NewLinkArmed lets the caller choose whether actuator events are emitted.
func NewLinkArmed(w io.Writer, armMotion bool) *Link {
	if w == nil {
		w = os.Stdout
	}
	return &Link{out: w, motionInhibit: !armMotion}
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

func (l *Link) Sport(apiID int, name string, params map[string]any) {
	if l.motionInhibit {
		log.Printf("[nina] (inhibited) sport %s api_id=%d", name, apiID)
		return
	}
	ev := map[string]any{"ev": "sport", "api_id": apiID, "name": name}
	if params != nil {
		ev["params"] = params
	}
	l.emit(ev)
}

func (l *Link) NavGoal(x, y, z, yaw float64) {
	if l.motionInhibit {
		log.Printf("[nina] (inhibited) nav_goal (%.2f, %.2f)", x, y)
		return
	}
	l.emit(map[string]any{"ev": "nav_goal", "x": x, "y": y, "z": z, "yaw": yaw})
}

func (l *Link) CmdVel(wz float64) {
	if l.motionInhibit {
		log.Printf("[nina] (inhibited) cmd_vel wz=%.2f", wz)
		return
	}
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
