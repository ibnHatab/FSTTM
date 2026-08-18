package go2audio

// LAN signaling — the con_notify / con_ing handshake (newer firmware, port
// 9991) and the legacy /offer flow (pre-1.1.11 Go2, port 8081). Ported from
// unitree_auth.send_sdp_to_local_peer.

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"
)

// staticGcmKey — AESGCMUtil.keyBytes from the Unitree apk (data2 === 2).
var staticGcmKey = []byte{232, 86, 130, 189, 22, 84, 155, 0, 142, 4, 166,
	104, 43, 179, 235, 227}

func probeTCP(ip string, port int, timeout time.Duration) bool {
	c, err := net.DialTimeout("tcp",
		fmt.Sprintf("%s:%d", ip, port), timeout)
	if err != nil {
		return false
	}
	_ = c.Close()
	return true
}

// sendSDPLocal picks the signaling flow by which port answers and returns
// the SDP answer JSON. aes128Key (32 hex chars) is required when con_notify
// replies data2 === 3 (Go2 ≥ 1.1.15 / G1 ≥ 1.5.1 / all R1).
func sendSDPLocal(ip, sdpOffer, aes128Key string) (string, error) {
	if probeTCP(ip, 9991, 1500*time.Millisecond) {
		return sendSDPConNotify(ip, sdpOffer, aes128Key)
	}
	if probeTCP(ip, 8081, 1500*time.Millisecond) {
		return sendSDPLegacy(ip, sdpOffer)
	}
	return "", fmt.Errorf("go2audio: no signaling port open on %s (:9991/:8081)", ip)
}

func sendSDPLegacy(ip, sdpOffer string) (string, error) {
	resp, err := http.Post(fmt.Sprintf("http://%s:8081/offer", ip),
		"application/json", bytes.NewReader([]byte(sdpOffer)))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("go2audio: legacy /offer status %d", resp.StatusCode)
	}
	return string(b), nil
}

func sendSDPConNotify(ip, sdpOffer, aes128Key string) (string, error) {
	// 1. GET the robot's per-session key material
	resp, err := http.Post(fmt.Sprintf("http://%s:9991/con_notify", ip),
		"", nil)
	if err != nil {
		return "", err
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	decoded, err := base64.StdEncoding.DecodeString(string(body))
	if err != nil {
		return "", fmt.Errorf("go2audio: con_notify base64: %w", err)
	}
	var notify struct {
		Data1 string `json:"data1"`
		Data2 int    `json:"data2"`
	}
	if err := json.Unmarshal(decoded, &notify); err != nil {
		return "", fmt.Errorf("go2audio: con_notify json: %w", err)
	}

	data1 := notify.Data1
	switch notify.Data2 {
	case 2:
		if data1, err = gcmDecryptLegacy(data1, staticGcmKey); err != nil {
			return "", err
		}
	case 3:
		if aes128Key == "" {
			return "", fmt.Errorf("go2audio: robot needs the per-device " +
				"AES-128 key (data2=3); fetch it once via the cloud and pass aes_128_key")
		}
		key, err := hexKey(aes128Key)
		if err != nil {
			return "", err
		}
		if data1, err = gcmDecrypt(data1, key); err != nil {
			return "", err
		}
	}
	// data2 == 1 or absent → data1 already plaintext

	if len(data1) < 20 {
		return "", fmt.Errorf("go2audio: con_notify data1 too short")
	}
	pubKeyBody := data1[10 : len(data1)-10]
	pathEnding := localPathEnding(data1)

	pub, err := loadPublicKey(pubKeyBody)
	if err != nil {
		return "", fmt.Errorf("go2audio: robot public key: %w", err)
	}
	aesKey := generateAesKey()
	encSDP, err := aesEcbEncrypt(sdpOffer, aesKey)
	if err != nil {
		return "", err
	}
	wrappedKey, err := rsaEncrypt(aesKey, pub)
	if err != nil {
		return "", err
	}
	reqBody, _ := json.Marshal(map[string]string{
		"data1": encSDP, "data2": wrappedKey,
	})

	// 2. POST the encrypted offer; response is the AES-encrypted answer
	url := fmt.Sprintf("http://%s:9991/con_ing_%s", ip, pathEnding)
	resp2, err := http.Post(url, "application/x-www-form-urlencoded",
		bytes.NewReader(reqBody))
	if err != nil {
		return "", err
	}
	defer resp2.Body.Close()
	ans, _ := io.ReadAll(resp2.Body)
	return aesEcbDecrypt(string(ans), aesKey)
}

func hexKey(h string) ([]byte, error) {
	if len(h) != 32 {
		return nil, fmt.Errorf("go2audio: aes_128_key must be 32 hex chars, got %d", len(h))
	}
	out := make([]byte, 16)
	for i := 0; i < 16; i++ {
		var b byte
		if _, err := fmt.Sscanf(h[2*i:2*i+2], "%02x", &b); err != nil {
			return nil, err
		}
		out[i] = b
	}
	return out, nil
}
