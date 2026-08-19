package go2audio

// pion WebRTC driver — opens the peer connection, the "data" control
// channel (validation + heartbeat), and the live audio sender track, then
// enables the robot's audio channel. TTS PCM pushed to the track is heard on
// the Go2 speaker in real time. Port of webrtc_driver + webrtc_datachannel +
// webrtc_audio (the paths we need for output; lidar/video/cloud dropped).

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

// Conn is a live speaker connection to a Go2.
type Conn struct {
	pc        *webrtc.PeerConnection
	data      *webrtc.DataChannel
	track     *webrtc.TrackLocalStaticSample
	validated chan struct{}
	valOnce   sync.Once
	hbStop    chan struct{}
	closeOnce sync.Once
	dead      chan struct{} // closed when the peer connection drops
	deadOnce  sync.Once
}

type Config struct {
	IP      string // robot LAN IP
	AES128  string // per-device key (32 hex) — needed on newer firmware
	Timeout time.Duration
}

// dataMsg is the control-channel envelope (msgs/pub_sub).
type dataMsg struct {
	Type  string          `json:"type"`
	Topic string          `json:"topic"`
	Data  json.RawMessage `json:"data,omitempty"`
}

// Connect performs signaling + validation and returns a ready Conn whose
// audio track feeds the speaker.
func Connect(ctx context.Context, cfg Config) (*Conn, error) {
	if cfg.Timeout == 0 {
		cfg.Timeout = 15 * time.Second
	}
	api := webrtc.NewAPI()
	pc, err := api.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		return nil, err
	}
	c := &Conn{pc: pc, validated: make(chan struct{}),
		hbStop: make(chan struct{}), dead: make(chan struct{})}

	// mark the connection dead on any terminal peer state so the sink can
	// redial — WriteSample never errors on a dropped peer, so this is the
	// only signal we get that the robot went away (reboot, wifi, slot loss).
	pc.OnConnectionStateChange(func(st webrtc.PeerConnectionState) {
		switch st {
		case webrtc.PeerConnectionStateFailed,
			webrtc.PeerConnectionStateDisconnected,
			webrtc.PeerConnectionStateClosed:
			c.deadOnce.Do(func() { close(c.dead) })
		}
	})

	// audio sender track (Opus, the WebRTC-mandatory codec) — sendrecv so we
	// match the robot's transceiver even though we only send.
	c.track, err = webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus},
		"audio", "fsttm-tts")
	if err != nil {
		return nil, err
	}
	if _, err = pc.AddTransceiverFromTrack(c.track,
		webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionSendrecv}); err != nil {
		return nil, err
	}

	// video recvonly transceiver: we never consume it, but the PROVEN
	// uc11 offer (aiortc reference, fw 1.1.9 field test 2026-08-19)
	// carried audio sendrecv + video recvonly + data — keep the offer
	// shape identical so the robot's answerer sees what it expects.
	if _, err = pc.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo,
		webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}); err != nil {
		return nil, err
	}

	// control data channel
	c.data, err = pc.CreateDataChannel("data", nil)
	if err != nil {
		return nil, err
	}
	c.data.OnMessage(c.onData)

	// offer → LAN signaling → answer
	offer, err := pc.CreateOffer(nil)
	if err != nil {
		return nil, err
	}
	gatherComplete := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(offer); err != nil {
		return nil, err
	}
	<-gatherComplete

	offerJSON, _ := json.Marshal(map[string]string{
		"id": "STA_localNetwork", "sdp": pc.LocalDescription().SDP,
		"type": "offer", "token": "",
	})
	answerJSON, err := sendSDPLocal(cfg.IP, string(offerJSON), cfg.AES128)
	if err != nil {
		return nil, err
	}
	var ans struct{ Sdp, Type string }
	if err := json.Unmarshal([]byte(answerJSON), &ans); err != nil {
		return nil, fmt.Errorf("go2audio: answer json: %w", err)
	}
	if ans.Sdp == "reject" {
		return nil, fmt.Errorf("go2audio: robot busy (another WebRTC peer connected)")
	}
	if err := pc.SetRemoteDescription(webrtc.SessionDescription{
		Type: webrtc.SDPTypeAnswer, SDP: ans.Sdp}); err != nil {
		return nil, err
	}

	// wait for validation (data channel open + challenge answered)
	select {
	case <-c.validated:
	case <-time.After(cfg.Timeout):
		_ = pc.Close()
		return nil, fmt.Errorf("go2audio: validation timed out (state=%s)",
			pc.ConnectionState())
	case <-ctx.Done():
		_ = pc.Close()
		return nil, ctx.Err()
	}
	c.switchAudio(true) // enable the robot's audio channel
	go c.heartbeat()
	log.Printf("[go2audio] speaker connected: %s", cfg.IP)
	return c, nil
}

func (c *Conn) onData(msg webrtc.DataChannelMessage) {
	var m dataMsg
	if err := json.Unmarshal(msg.Data, &m); err != nil {
		return
	}
	switch m.Type {
	case "validation":
		var s string
		_ = json.Unmarshal(m.Data, &s)
		if s == "Validation Ok." {
			c.valOnce.Do(func() { close(c.validated) })
			return
		}
		// challenge: echo md5("UnitreeGo2_"+key) as base64
		c.send(dataMsg{Type: "validation",
			Data: mustJSON(validationKey(s))})
	case "err":
		// robot asks to re-validate → resend last challenge answer is
		// handled by the next validation message; nothing to do here.
	}
}

func (c *Conn) send(m dataMsg) {
	if m.Topic == "" {
		m.Topic = ""
	}
	b, _ := json.Marshal(m)
	_ = c.data.SendText(string(b))
}

func (c *Conn) switchAudio(on bool) {
	v := "off"
	if on {
		v = "on"
	}
	c.send(dataMsg{Type: "aud", Data: mustJSON(v)})
}

func (c *Conn) heartbeat() {
	t := time.NewTicker(2 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-c.hbStop:
			return
		case <-t.C:
			c.send(dataMsg{Type: "heartbeat",
				Data: mustJSON(map[string]int64{
					"timeInStr": time.Now().UnixMilli()})})
		}
	}
}

// WriteOpus pushes one encoded Opus frame (a media sample) to the speaker.
func (c *Conn) WriteOpus(frame []byte, dur time.Duration) error {
	return c.track.WriteSample(media.Sample{Data: frame, Duration: dur})
}

// Dead is closed when the peer connection drops (or on Close). Alive is a
// non-blocking check of the same.
func (c *Conn) Dead() <-chan struct{} { return c.dead }
func (c *Conn) Alive() bool {
	select {
	case <-c.dead:
		return false
	default:
		return true
	}
}

func (c *Conn) Close() {
	c.closeOnce.Do(func() {
		c.deadOnce.Do(func() { close(c.dead) })
		close(c.hbStop)
		c.switchAudio(false)
		_ = c.pc.Close()
		// give the DTLS/ICE close a moment to reach the robot so it frees
		// its single WebRTC slot immediately instead of waiting to time out.
		time.Sleep(300 * time.Millisecond)
	})
}

func mustJSON(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}
