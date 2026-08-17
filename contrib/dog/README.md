# fsttm-dog

Robot-dog (Unitree Go2) intent domain for the [FSTTM](../../README.md) spoken
dialog engine. Implements the natural-language → intent → semantic-query →
navigation-goal architecture described in [spec.md](spec.md). Ships typed
action/query seams (`ActionBackend`, `SemanticMemory`, `NavigationBackend`)
with logging stubs — no ROS2 dependency; the real nav2 / DINOv3 wiring plugs
in on the robot.

```bash
pip install -e .            # registers the "dog" domain entry point
fsttm-headless --config config.dog.sample.yaml --intent --no-aec
```
