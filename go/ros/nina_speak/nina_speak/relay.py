"""nina_speak relay — the ROS2 main after the Go override.

Spawns fsttm-go (which owns ALL realtime: audio, VAD, STT, LLM, TTS,
voice-id) and translates its jsonl events to the contracted topics. This
node is purely event-driven — a handful of messages per voice interaction,
~0% CPU — and it is the ONLY ROS mouth of the voice stack:

    /api/sport/request  unitree_api/Request   (H2 api-id table)
    /nina/nav_goal      geometry_msgs/PoseStamped (map frame)
    /cmd_vel            geometry_msgs/Twist   (bounded turn bursts ONLY —
                        the cmd_arbiter is the single motion author; this
                        relay publishes no other motion, ever)
    /nina/intent        std_msgs/String       (intent JSON per turn)   [new]
    /nina/dialog_state  std_msgs/String       (ASLEEP/AWAKE/…)         [new]
    /nina/say           std_msgs/String  SUB  (announce requests)      [new]

Config: ROS param `engine_cmd` (default: fsttm-go -config <share>/config
next to this package's config.dog.sample.yaml with nina.enabled: true).
"""
import json
import math
import subprocess
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String

try:
    from unitree_api.msg import Request
    HAVE_UNITREE = True
except ImportError:                      # SIL on a laptop without unitree msgs
    HAVE_UNITREE = False


class NinaRelay(Node):
    def __init__(self):
        super().__init__("nina_speak")
        self.declare_parameter("engine_cmd", "fsttm-go -config config.dog.yaml")
        cmd = self.get_parameter("engine_cmd").value.split()

        if HAVE_UNITREE:
            self.sport = self.create_publisher(Request, "/api/sport/request", 5)
        self.goal = self.create_publisher(PoseStamped, "/nina/nav_goal", 5)
        self.cmd_vel = self.create_publisher(Twist, "/cmd_vel", 5)
        self.intent = self.create_publisher(String, "/nina/intent", 5)
        self.state = self.create_publisher(String, "/nina/dialog_state", 5)
        self.create_subscription(String, "/nina/say", self.on_say, 5)

        self._req_id = 0
        self.get_logger().info(f"spawning engine: {' '.join(cmd)}")
        self.child = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True,
                                      bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()

    # ── engine → ROS ─────────────────────────────────────────────────────
    def _pump(self):
        for line in self.child.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("ev")
            if kind == "sport":
                self._sport(int(ev["api_id"]), ev.get("params"))
            elif kind == "nav_goal":
                self._nav(ev)
            elif kind == "cmd_vel":
                tw = Twist()
                tw.angular.z = float(ev.get("wz", 0.0))
                self.cmd_vel.publish(tw)
            elif kind == "intent":
                self.intent.publish(String(data=json.dumps(ev.get("intent"))))
            elif kind == "state":
                self.state.publish(String(data=ev.get("dialog", "")))
        self.get_logger().warn("engine exited")
        rclpy.shutdown()

    def _sport(self, api_id, params):
        if not HAVE_UNITREE:
            self.get_logger().info(f"[sil] sport api {api_id}")
            return
        r = Request()
        self._req_id += 1
        r.header.identity.id = self._req_id
        r.header.identity.api_id = api_id
        r.parameter = json.dumps(params) if params else ""
        self.sport.publish(r)

    def _nav(self, ev):
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(ev["x"])
        ps.pose.position.y = float(ev["y"])
        ps.pose.position.z = float(ev.get("z", 0.0))
        yaw = float(ev.get("yaw", 0.0))
        ps.pose.orientation.z = math.sin(yaw / 2)
        ps.pose.orientation.w = math.cos(yaw / 2)
        self.goal.publish(ps)

    # ── ROS → engine ─────────────────────────────────────────────────────
    def on_say(self, msg: String):
        try:
            self.child.stdin.write(
                json.dumps({"ev": "say", "text": msg.data}) + "\n")
            self.child.stdin.flush()
        except BrokenPipeError:
            self.get_logger().warn("engine stdin gone")


def main():
    rclpy.init()
    node = NinaRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.child.poll() is None:
            node.child.terminate()


if __name__ == "__main__":
    main()
