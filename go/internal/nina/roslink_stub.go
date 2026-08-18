//go:build !ros

package nina

import (
	"context"
	"errors"
)

// NewRosLink without the `ros` build tag: unavailable. Build with
// `ROS=1 ./build.sh` on a ROS-sourced host (Humble + rclgo v0.5.x).
func NewRosLink(_ context.Context, _ func(string)) (RobotLink, error) {
	return nil, errors.New("nina: built without ROS support (ROS=1 ./build.sh)")
}
