//go:build ros

// RosLink — the rclgo-NATIVE RobotLink: the engine IS the nina_speak ROS 2
// node, publishing the contracted topics directly (MerlinDrones/rclgo
// v0.5.x, Humble; built on a ROS-sourced host — see go/build.sh ROS=1).
//
// The topic set is identical to the jsonl transport, enforced here by
// construction: these five publishers and the /nina/say subscription are
// the node's complete interface — the arbiter doctrine in one screen.
package nina

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"

	geometry_msgs_msg "github.com/merlindrones/rclgo/pkg/msgs/geometry_msgs/msg"
	std_msgs_msg "github.com/merlindrones/rclgo/pkg/msgs/std_msgs/msg"
	"github.com/merlindrones/rclgo/pkg/rclgo"

	unitree_api_msg "github.com/ibnHatab/fsttm/go/internal/rosmsgs/unitree_api/msg"
)

type RosLink struct {
	node   *rclgo.Node
	sport  *unitree_api_msg.RequestPublisher
	goal   *geometry_msgs_msg.PoseStampedPublisher
	cmdVel *geometry_msgs_msg.TwistPublisher
	intent *std_msgs_msg.StringPublisher
	state  *std_msgs_msg.StringPublisher
	reqID  int64

	// motionInhibit gates the ACTUATOR publishers (sport / cmd_vel /
	// nav_goal): when true they are logged, not published, so the physical
	// robot never moves. Observation topics (intent, dialog_state) stay
	// live. DEFAULT true — a bare launch or a functional test can never
	// drive the legs; the operator arms motion explicitly. See NewRosLink.
	motionInhibit bool
}

var _ RobotLink = (*RosLink)(nil)

// NewRosLink initializes rclgo, creates the nina_speak node with the five
// contracted publishers, subscribes /nina/say → announce, and spins the
// node until ctx is done.
func NewRosLink(ctx context.Context, announce func(string), armMotion bool) (*RosLink, error) {
	if err := rclgo.Init(nil); err != nil {
		return nil, fmt.Errorf("roslink: %w", err)
	}
	node, err := rclgo.NewNode("nina_speak", "")
	if err != nil {
		return nil, fmt.Errorf("roslink: %w", err)
	}
	l := &RosLink{node: node, motionInhibit: !armMotion}
	if l.sport, err = unitree_api_msg.NewRequestPublisher(
		node, "/api/sport/request", nil); err != nil {
		return nil, err
	}
	if l.goal, err = geometry_msgs_msg.NewPoseStampedPublisher(
		node, "/nina/nav_goal", nil); err != nil {
		return nil, err
	}
	if l.cmdVel, err = geometry_msgs_msg.NewTwistPublisher(
		node, "/cmd_vel", nil); err != nil {
		return nil, err
	}
	if l.intent, err = std_msgs_msg.NewStringPublisher(
		node, "/nina/intent", nil); err != nil {
		return nil, err
	}
	if l.state, err = std_msgs_msg.NewStringPublisher(
		node, "/nina/dialog_state", nil); err != nil {
		return nil, err
	}
	_, err = std_msgs_msg.NewStringSubscription(node, "/nina/say", nil,
		func(msg *std_msgs_msg.String, _ *rclgo.MessageInfo, err error) {
			if err == nil && msg.Data != "" {
				log.Printf("[nina] /nina/say → announce %q", msg.Data)
				announce(msg.Data)
			}
		})
	if err != nil {
		return nil, err
	}
	go func() {
		if err := node.Spin(ctx); err != nil && ctx.Err() == nil {
			log.Printf("[roslink] spin: %v", err)
		}
	}()
	if l.motionInhibit {
		log.Print("[nina] ⚠ MOTION INHIBITED — sport/cmd_vel/nav_goal are " +
			"logged, not published (set nina.arm_motion: true to drive the robot)")
	} else {
		log.Print("[nina] ⚠ MOTION ARMED — sport/cmd_vel/nav_goal reach the robot")
	}
	log.Print("[nina] rclgo native node up: nina_speak")
	return l, nil
}

func (l *RosLink) Sport(apiID int, name string, params map[string]any) {
	if l.motionInhibit {
		log.Printf("[nina] (inhibited) sport %s api_id=%d", name, apiID)
		return
	}
	r := unitree_api_msg.NewRequest()
	l.reqID++
	r.Header.Identity.Id = l.reqID
	r.Header.Identity.ApiId = int64(apiID)
	if params != nil {
		b, _ := json.Marshal(params)
		r.Parameter = string(b)
	}
	if err := l.sport.Publish(r); err != nil {
		log.Printf("[roslink] sport publish: %v", err)
	}
}

func (l *RosLink) NavGoal(x, y, z, yaw float64) {
	if l.motionInhibit {
		log.Printf("[nina] (inhibited) nav_goal (%.2f, %.2f)", x, y)
		return
	}
	ps := geometry_msgs_msg.NewPoseStamped()
	ps.Header.FrameId = "map"
	ps.Pose.Position.X, ps.Pose.Position.Y, ps.Pose.Position.Z = x, y, z
	ps.Pose.Orientation.Z = math.Sin(yaw / 2)
	ps.Pose.Orientation.W = math.Cos(yaw / 2)
	if err := l.goal.Publish(ps); err != nil {
		log.Printf("[roslink] goal publish: %v", err)
	}
}

func (l *RosLink) CmdVel(wz float64) {
	if l.motionInhibit {
		log.Printf("[nina] (inhibited) cmd_vel wz=%.2f", wz)
		return
	}
	tw := geometry_msgs_msg.NewTwist()
	tw.Angular.Z = wz
	if err := l.cmdVel.Publish(tw); err != nil {
		log.Printf("[roslink] cmd_vel publish: %v", err)
	}
}

func (l *RosLink) Intent(intentJSON any, voice string) {
	b, err := json.Marshal(intentJSON)
	if err != nil {
		return
	}
	m := std_msgs_msg.NewString()
	m.Data = string(b)
	_ = l.intent.Publish(m)
}

func (l *RosLink) DialogState(state string) {
	m := std_msgs_msg.NewString()
	m.Data = state
	_ = l.state.Publish(m)
}
