// Package go2audio streams TTS to a Unitree Go2 speaker over WebRTC — a Go
// port of the audio path in legion1581/unitree_webrtc_connect (Python /
// aiortc). We reach the robot's LIVE audio transceiver (sendrecv) and push
// Opus frames, so a spoken reply is heard immediately — no file upload, no
// MP3 round-trip (the audiohub upload path is for pre-baked clips).
//
// This file: the crypto used by the LAN signaling handshake and the data-
// channel validation, ported byte-for-byte from the Python driver and pinned
// against it (crypto_vectors_test.go).
package go2audio

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/md5"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"fmt"
)

// validationKey reproduces WebRTCDataChannelValidaton.encrypt_key:
// md5("UnitreeGo2_" + key) hex → bytes → base64. The robot sends `key` in
// the first validation message; we echo this back to prove we know the
// scheme.
func validationKey(key string) string {
	// md5 hex digest → bytes.fromhex → base64. The Python hex_to_base64
	// decodes the hex back to the 16 raw md5 bytes before base64 — so this
	// is base64 of the raw digest, NOT of the 32-char hex string.
	sum := md5.Sum([]byte("UnitreeGo2_" + key))
	return base64.StdEncoding.EncodeToString(sum[:])
}

// pkcs7pad / pkcs7unpad — the Python `pad`/`unpad` (PKCS#5/7 on 16-byte
// blocks; the value of each pad byte is the pad length).
func pkcs7pad(data []byte) []byte {
	n := aes.BlockSize - len(data)%aes.BlockSize
	out := make([]byte, len(data)+n)
	copy(out, data)
	for i := len(data); i < len(out); i++ {
		out[i] = byte(n)
	}
	return out
}

func pkcs7unpad(data []byte) ([]byte, error) {
	if len(data) == 0 || len(data)%aes.BlockSize != 0 {
		return nil, fmt.Errorf("go2audio: bad padded length %d", len(data))
	}
	n := int(data[len(data)-1])
	if n == 0 || n > aes.BlockSize || n > len(data) {
		return nil, fmt.Errorf("go2audio: bad pad byte %d", n)
	}
	return data[:len(data)-n], nil
}

// aesEcbEncrypt / aesEcbDecrypt — AES-ECB with PKCS7, base64 in/out. The
// Python driver uses a 32-char hex UUID as the "key" string; its UTF-8
// bytes (32 bytes) key AES-256 directly (NOT hex-decoded — matches
// aes_encrypt's `key.encode('utf-8')`).
func aesEcbEncrypt(plain, key string) (string, error) {
	block, err := aes.NewCipher([]byte(key))
	if err != nil {
		return "", err
	}
	src := pkcs7pad([]byte(plain))
	dst := make([]byte, len(src))
	for i := 0; i < len(src); i += aes.BlockSize {
		block.Encrypt(dst[i:i+aes.BlockSize], src[i:i+aes.BlockSize])
	}
	return base64.StdEncoding.EncodeToString(dst), nil
}

func aesEcbDecrypt(b64, key string) (string, error) {
	block, err := aes.NewCipher([]byte(key))
	if err != nil {
		return "", err
	}
	src, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return "", err
	}
	if len(src)%aes.BlockSize != 0 {
		return "", fmt.Errorf("go2audio: ciphertext not block-aligned")
	}
	dst := make([]byte, len(src))
	for i := 0; i < len(src); i += aes.BlockSize {
		block.Decrypt(dst[i:i+aes.BlockSize], src[i:i+aes.BlockSize])
	}
	out, err := pkcs7unpad(dst)
	return string(out), err
}

// rsaEncrypt — PKCS1v15, chunked at (keySize-11), base64 of the concatenated
// ciphertext (matches rsa_encrypt in the Python driver).
func rsaEncrypt(data string, pub *rsa.PublicKey) (string, error) {
	maxChunk := pub.Size() - 11
	var out []byte
	for i := 0; i < len(data); i += maxChunk {
		end := i + maxChunk
		if end > len(data) {
			end = len(data)
		}
		enc, err := rsa.EncryptPKCS1v15(rand.Reader, pub, []byte(data[i:end]))
		if err != nil {
			return "", err
		}
		out = append(out, enc...)
	}
	return base64.StdEncoding.EncodeToString(out), nil
}

// loadPublicKey parses the robot's per-session RSA public key. The Python
// driver base64-decodes the PEM body then RSA.import_key; the decoded bytes
// are DER (PKIX or PKCS1).
func loadPublicKey(pemBody string) (*rsa.PublicKey, error) {
	der, err := base64.StdEncoding.DecodeString(pemBody)
	if err != nil {
		return nil, err
	}
	if k, err := x509.ParsePKIXPublicKey(der); err == nil {
		if rk, ok := k.(*rsa.PublicKey); ok {
			return rk, nil
		}
	}
	return x509.ParsePKCS1PublicKey(der)
}

// generateAesKey — 32-char hex UUID (16 random bytes hex-encoded), exactly
// _generate_uuid() in the Python driver.
func generateAesKey() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

// gcmDecryptLegacy — `data2 === 2`: AES-GCM with the static apk key
// (12-byte nonce prefix). Used to decrypt con_notify data1 on Go2 < 1.1.15.
func gcmDecryptLegacy(b64 string, staticKey []byte) (string, error) {
	return gcmDecrypt(b64, staticKey)
}

// gcmDecrypt — AESGCM(key).decrypt(nonce, ciphertext+tag): first 12 bytes
// nonce, remainder ciphertext‖tag.
func gcmDecrypt(b64 string, key []byte) (string, error) {
	raw, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		return "", err
	}
	if len(raw) < 12+16 {
		return "", fmt.Errorf("go2audio: gcm blob too short")
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return "", err
	}
	g, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}
	out, err := g.Open(nil, raw[:12], raw[12:], nil)
	if err != nil {
		return "", fmt.Errorf("go2audio: gcm auth failed (wrong AES-128 key?): %w", err)
	}
	return string(out), nil
}

// localPathEnding reproduces _calc_local_path_ending: last 10 chars of the
// decrypted data1, pairs, map each pair's 2nd char A–J → its index.
func localPathEnding(data1 string) string {
	if len(data1) < 10 {
		return ""
	}
	last10 := data1[len(data1)-10:]
	var out []byte
	for i := 0; i+1 < len(last10); i += 2 {
		c := last10[i+1]
		if c >= 'A' && c <= 'J' {
			out = append(out, '0'+(c-'A'))
		}
	}
	return string(out)
}
