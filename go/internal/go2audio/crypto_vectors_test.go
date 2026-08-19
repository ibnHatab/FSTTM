package go2audio

// Crypto pinned against the Python reference (legion1581/unitree_webrtc_
// connect). Vectors generated from its encryption.py / validation scheme —
// if the robot changes the handshake, these break loudly.

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"testing"
)

func TestValidationKeyVector(t *testing.T) {
	// md5("UnitreeGo2_abc123XYZ") → hex → base64
	if got := validationKey("abc123XYZ"); got != "2raWR7HTXomcQv27DtNkhg==" {
		t.Fatalf("validationKey = %q", got)
	}
}

func TestAesEcbVector(t *testing.T) {
	key := "0123456789abcdef0123456789abcdef"
	want := "7NlFxPDec4cBG3Cg5WgexuibZQIx4pSDROgGHcYvOaM="
	got, err := aesEcbEncrypt("hello go2 speaker", key)
	if err != nil || got != want {
		t.Fatalf("aesEcbEncrypt = %q (%v)", got, err)
	}
	rt, err := aesEcbDecrypt(got, key)
	if err != nil || rt != "hello go2 speaker" {
		t.Fatalf("roundtrip = %q (%v)", rt, err)
	}
}

func TestGcmLegacyDecryptVector(t *testing.T) {
	// AESGCM(static apk key), SUFFIX layout [ct‖nonce(0..11)‖tag] — the
	// robot's real wire format (fw 1.1.9 field-confirmed), not nonce-prefix
	staticKey := []byte{232, 86, 130, 189, 22, 84, 155, 0, 142, 4, 166,
		104, 43, 179, 235, 227}
	blob := "2cfMwkss8E69OWXgzOvtnS544227M1QAAQIDBAUGBwgJCgs4Fu4F3ttBDxwWQZJVgueN"
	got, err := gcmDecryptLegacy(blob, staticKey)
	if err != nil || got != "session-public-key-body" {
		t.Fatalf("gcmDecryptLegacy = %q (%v)", got, err)
	}
}

func TestLocalPathEnding(t *testing.T) {
	if got := localPathEnding("XXXXXXXXXXaAbBcCdDeE"); got != "01234" {
		t.Fatalf("localPathEnding = %q", got)
	}
}

// RSA round-trip: our chunked PKCS1v15 encrypt is decryptable by a standard
// private key (the robot side), proving the wire format.
func TestRsaEncryptDecryptable(t *testing.T) {
	priv, _ := rsa.GenerateKey(rand.Reader, 2048)
	pubDER, _ := x509.MarshalPKIXPublicKey(&priv.PublicKey)
	pub, err := loadPublicKey(base64.StdEncoding.EncodeToString(pubDER))
	if err != nil {
		t.Fatal(err)
	}
	aesKey := generateAesKey()
	if len(aesKey) != 32 {
		t.Fatalf("aes key must be 32 hex chars, got %d", len(aesKey))
	}
	enc, err := rsaEncrypt(aesKey, pub)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := base64.StdEncoding.DecodeString(enc)
	dec, err := rsa.DecryptPKCS1v15(rand.Reader, priv, raw)
	if err != nil || string(dec) != aesKey {
		t.Fatalf("rsa round-trip failed: %q (%v)", dec, err)
	}
}
