module github.com/ibnHatab/fsttm/go

go 1.23

require (
	github.com/gen2brain/malgo v0.11.26
	github.com/ggerganov/whisper.cpp/bindings/go v0.0.0
	github.com/maxhawkins/go-webrtcvad v0.0.0-20210121163624-be60036f3083
	gopkg.in/yaml.v3 v3.0.1
)

replace github.com/ggerganov/whisper.cpp/bindings/go => ../../whisper.cpp/bindings/go
