// The three Nina robot seams (ActionBackend / SemanticMemory /
// NavigationBackend) — Go port of backends_nina.py, emitting through the
// jsonl RobotLink. See canvas.go for the arbiter doctrine.
package nina

import (
	"log"
	"math"
	"time"

	"github.com/ibnHatab/fsttm/go/internal/intent"
)

// H2 firmware contract (h2-command-contract.md; confirmed by the nina_ws
// agent 2026-08-18). ClassicWalk 2049 {data:true} exists in the table but is
// mode control, not a voice action.
var sportAPI = map[string]int{
	"STAND_UP": 1004, "SIT_DOWN": 1009, "LIE_DOWN": 1005,
	"STRETCH": 1017, "SHAKE": 1016, "STOP": 1003, "BALANCE": 1002,
}

// Voice-safe subset: the LLM's enum is wider than what a spoken command may
// trigger — no JUMP/POUNCE from speech.
var voiceSafe = map[string]bool{
	"STAND_UP": true, "SIT_DOWN": true, "LIE_DOWN": true,
	"STRETCH": true, "SHAKE": true,
}

// Actions — LOCAL_ACTION → sport api; TURN → bounded /cmd_vel burst.
type Actions struct {
	Link RobotLink
}

func (a *Actions) Execute(c *intent.Command) {
	if c.Action == "TURN" && c.AngleDeg != 0 {
		a.turn(c.AngleDeg*math.Pi/180, c.Direction)
		return
	}
	if !voiceSafe[c.Action] {
		log.Printf("[nina] action %s refused (voice-safe set only)", c.Action)
		return
	}
	a.Link.Sport(sportAPI[c.Action], c.Action, nil)
	log.Printf("[nina] sport %s sent", c.Action)
}

// turn: bounded in-place rotation through /cmd_vel — the cmd_arbiter owns
// shaping (0.4 rad/s, 10 Hz ticks for the computed window, then a zero
// Twist). This is the ONLY velocity this engine ever emits, and only as a
// finite burst — never a stream.
func (a *Actions) turn(angleRad float64, direction string) {
	sign := 1.0
	if direction == "RIGHT" {
		sign = -1.0
	}
	wz := 0.4 * sign
	deadline := time.Now().Add(
		time.Duration(math.Abs(angleRad)/0.4*1000) * time.Millisecond)
	log.Printf("[nina] turn %.0f° via /cmd_vel (bounded burst)",
		angleRad*180/math.Pi*sign)
	for time.Now().Before(deadline) {
		a.Link.CmdVel(wz)
		time.Sleep(100 * time.Millisecond)
	}
	a.Link.CmdVel(0)
}

func (a *Actions) Velocity(v *intent.Command) {
	log.Print("[nina] VELOCITY intents are planner territory (spec §3) — ignored")
}

func (a *Actions) Stop() { // spec §14: immediate, bypasses planning
	a.Link.Sport(sportAPI["STOP"], "STOP", nil)
	log.Print("[nina] STOP → StopMove")
}

func (a *Actions) Cancel() { a.Stop() }

// Nav — navigation goals on /nina/nav_goal (map frame). EXPLORE/FOLLOW stay
// deferred exactly as in the Python reference (UC2 planner / person tracker).
type Nav struct {
	Link RobotLink
}

func (n *Nav) Navigate(pos [3]float64, instanceID int) {
	n.Link.NavGoal(pos[0], pos[1], pos[2], 0)
	log.Printf("[nina] nav goal (%.2f, %.2f) published (instance %d)",
		pos[0], pos[1], instanceID)
}

func (n *Nav) Explore(t intent.SemanticTarget) {
	log.Printf("[nina] EXPLORE %q deferred to the UC2 planner — not wired yet",
		t.Description)
}

func (n *Nav) Follow(t intent.SemanticTarget) {
	log.Printf("[nina] FOLLOW %q needs the person tracker — not wired yet",
		t.Description)
}

func (n *Nav) Cancel() {
	n.Link.Sport(sportAPI["STOP"], "STOP", nil)
	log.Print("[nina] nav cancel → StopMove")
}

// NewDispatcher wires the dog-intent dispatcher to the Nina seams. Canvas
// may be nil (no pack yet) — the memory then reports nothing found, and
// FIND degrades to the EXPLORE log per the spec fallback.
func NewDispatcher(link RobotLink, canvas *Canvas) *intent.Dispatcher {
	d := intent.NewLogging()
	d.Actions = &Actions{Link: link}
	d.Nav = &Nav{Link: link}
	if canvas != nil {
		d.Memory = canvas
	}
	return d
}
