# HANDOFF — fsttm maintainer agent (Jetson)

You are maintaining **fsttm**, a local voice assistant (STT→LLM→TTS) with N09-1071
finite-state turn-taking, running on this Jetson AGX Xavier. The system is at
**v1.0** and works. Your job is **light tuning of intents and the TUI** — NOT
changing the core engine.

---

## 0. The constraint that shapes everything: NO NETWORK

This session the Jetson has **no internet / no GitLab access**. You cannot
`git push` to origin and cannot `git fetch`. So the workflow is:

1. **Edit and test directly on the Jetson** (here, `/home/nvidia/repo/vox/fsttm`).
2. When changes are good, **rsync them back to the host** (which HAS network).
3. The **host commits and pushes** to GitLab.

You drive step 1 and 2. A human (or the host-side agent) does step 3.

### Topology
| | path | git |
|---|---|---|
| **This Jetson** | `nvidia@<this box>:/home/nvidia/repo/vox/fsttm` | origin = GitLab (UNREACHABLE this session) |
| **Host** | `axadmin@boxter:/home/axadmin/repo/vox/fsttm` | origin = GitLab (reachable); also has remote `jetson-live` → this box |

Branch is **`main`** on both. Current HEAD: `5410b73` (v1.0 tag is on origin).

**SSH to this Jetson** (from the host — key auth, no password needed). This is the
`~/.ssh/config` entry on the host so `ssh jetson` / rsync resolves to the board:

```sshconfig
Host jetson
	HostName 10.0.0.67
	User nvidia
	ForwardX11 yes
```
(Login is by SSH key; the host's public key is already authorized on the Jetson —
you do not need a password. If key auth ever breaks, the operator re-adds the key;
do not hardcode credentials in this repo.)

### Rsync changed source back to the host
From the Jetson, push only the source you changed (NOT models/, .venv/, *.npz,
*.gguf, srv.out, fsttm.log, the board-local config*.yaml):

```bash
rsync -avz --relative \
  fsttm/intents/ fsttm/tui.py \
  axadmin@boxter:/home/axadmin/repo/vox/fsttm/
```
Then tell the operator: "synced to host — please review `git diff` and commit/push."
The HOST side does:
```bash
git add -p && git commit && git push origin main
```
> If you can't reach the host either, just describe the diff in your final message
> and leave the files edited on the Jetson; the operator will rsync/commit.

### DO NOT do git operations against origin on the Jetson
No `git pull`, `git fetch`, `git push` — they'll hang on the dead network. Local
`git diff` / `git status` / `git stash` are fine.

---

---

## 0b. Starting the maintainer agent (Claude Code)

Claude Code is **already installed on this Jetson** (done in the v1.0 session):
- Node.js **v20** and npm are present (Claude Code needs Node ≥18). aarch64 — fine,
  the package is pure JS.
- `claude` is installed globally (`sudo npm install -g @anthropic-ai/claude-code`);
  check with `claude --version`.

### Start the agent
From `~/repo/vox/fsttm`:
```bash
claude --dangerously-skip-permissions
```
`--dangerously-skip-permissions` lets the agent run tool calls without per-action
prompts — appropriate for this autonomous maintenance box. Read `HANDOFF.md` first.

### If you must (re)install — REQUIRES NETWORK (won't work offline)
Node 20 is already here, so normally you only need the global package:
```bash
sudo npm install -g @anthropic-ai/claude-code
claude --version
```
If Node were missing/old (Ubuntu's apt `nodejs` can be stale), install Node 22 via
nvm first, then the package:
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc && nvm install 22 && nvm use 22
npm install -g @anthropic-ai/claude-code     # no sudo under nvm
```
If `claude` isn't found after install, the npm global bin isn't on PATH — reopen
the shell or add `$(npm prefix -g)/bin` to PATH. **Do all of this while the box is
online; the offline session cannot npm-install anything.**

### API key
Auth is via `ANTHROPIC_API_KEY` (or an interactive `claude` login). Set it in the
shell before starting — do NOT commit it to this repo:
```bash
export ANTHROPIC_API_KEY="<key>"
```

## 1. What you MAY change (the tuning surface)

- **`fsttm/intents/__init__.py`** — the prompt: header guidance, few-shot examples
  (`_FEWSHOT`, `_FEWSHOT_EXTRA`), prompt variants. This is where you fix
  misclassifications (e.g. "X said → wrong intent"). 90% of intent tuning is here.
- **`fsttm/intents/{climate,lights,body,manual}.py`** — per-domain intent tables
  (trigger phrases) and `translate()` (intent JSON → car command). Edit a domain's
  prompt table to teach new phrasings; edit its translate to fix the emitted command.
- **`fsttm/intents/base.py`** — the schema/grammar assembler + the meta-intent enum
  (`TIME, DATE, STATUS, CHITCHAT, UNKNOWN`). Add a meta intent here if needed.
- **`fsttm/tui.py`** — the 3-panel Rich TUI (Chat / Intents / State·Perf). Layout,
  colors, what's shown. Pure presentation.
- **`docs/USER_MANUAL.md`** — keep in sync when you change intents/phrases.
- **config** (board-local, see §3) — toggles, not code.

## 2. What you must NOT touch (the core engine)

Do not modify these unless the operator explicitly asks — they are stable and
subtle:
- **`fsttm/fsttm.py`** — the N09-1071 FSM (turn-taking state machine). Off-limits.
- **`fsttm/server.py`** — the reactive graph: barge-in, narrator/checkpoints,
  KV-prefix reuse, floor transitions, RAG/chitchat routing. Very subtle; many past
  bugs lived here. Only the operator touches the engine.
- **`fsttm/two_pass.py`** — the intent two-pass + **KV prefix reuse / pre-warm**
  (the perf-critical path). Do not change the eval/generate strategy.
- **`fsttm/whisper.py`, `fsttm/piper.py`, `fsttm/perception.py`, `fsttm/aec.py`** —
  audio drivers. Stable.

If a fix seems to need a core-engine change, STOP and report it to the operator
rather than editing the engine.

---

## 3. Running it (what's deployed now)

Run in tmux (window `fsttm-tui:server`). The launcher needs CUDA libs and (for
intent mode) the AEC off-or-on toggle.

```bash
cd ~/repo/vox/fsttm && source .venv/bin/activate
# AEC ON (echo-cancel; current default):
unset FSTTM_NO_AEC
LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH \
  python -u -m fsttm.server --config config.jetson.yaml --tui
# Headless (no TUI, timestamped stdout — best for profiling): drop --tui, add ` 2>&1 | tee srv.out`
```

**Config knobs** (`config.jetson.yaml`, board-local — see gotcha below):
- `system.hvac_intent: true|false` — intent mode vs plain chat
- `system.intent_domains: [climate, lights, body, manual]`
- `system.manual: true` + `manual_store` — RAG (Taycan manual ingested)
- `system.prompt_variant: one-shot|few-shot|few-shot-extra` (currently few-shot-extra)
- `system.attention: true` (wake "Nina") + `sleep_intent: true` ("Hey Nina, voice off")
- `aec.enabled: true, method: speex` (webrtc is broken on this Jetson's PulseAudio)

### Logs
- `fsttm.log` — timestamped (ms), all `fsttm.*` loggers. Your main diagnostic.
- Headless `srv.out` (when run with `tee srv.out`) — timestamped console mirror.
- Useful greps: `intent OK:`, `approach_a timing:` (shows `warm=True/False`),
  `[manual] … passages`, `[asleep] ignored`.

---

## 4. GOTCHAS (learned the hard way)

- **`config.jetson.yaml` is git-tracked but board-modified.** `git status` shows it
  as ` M`. Do NOT `git checkout`/`stash` it away — those are the live deployment
  settings. There are also untracked `config.jetson*.bak`, `intent_*.py/out` on the
  board; leave them. When rsyncing source to host, EXCLUDE config/models/logs.
- **jetson_clocks does NOT survive reboot.** After any reboot, GPU drops to the
  dynamic governor and inference gets slow. Re-pin with: `sudo jetson_clocks`
  (MAXN power mode persists; the clock pinning does not). Verify GPU min==max==
  1377000000 at `/sys/devices/17000000.gv11b/devfreq/.../min_freq`.
- **Prefix pre-warm + reuse:** intent prompt (~3800 tok) is eval'd ONCE at startup
  (~11s, in `two_pass._prime_prefix` via AddSystem) and kept in the KV cache, so
  each command is ~150ms eval. A chitchat/RAG/chat `create_completion` RESETS the
  KV and wipes it; we re-warm in the idle gap after. If you see `approach_a timing:
  … warm=False prime=~11000ms`, the prefix got wiped — that's known, not a new bug.
- **RAG embedder is on CPU** (`manual_embed_gpu: false`). Do NOT set it to GPU — it
  OOMs the shared VRAM against the Phi-3 LLM (`CUDA error: out of memory`).
- **Replug the Jabra → restart the server** (it holds a stale device handle; audio
  goes silent until restart). The device itself is fine; just relaunch.
- **TTS sink:** with AEC on, TTS routes to `fsttm_ec_sink`; the server falls back to
  the default sink if that's missing. If audio is silent, check `pactl list short
  sinks | grep fsttm_ec_sink` exists.
- **STT is imperfect** (whisper base.en + mic): "how to" → "I hope to", etc. Many
  "wrong intent" reports are actually STT mishears — check the `transcript:` line in
  the log before blaming the intent prompt.
- **tmux dies on reboot.** Recreate: `tmux new-session -d -s fsttm-tui -n server`.

---

## 5. Typical task: fix a misclassification

1. Reproduce; read `fsttm.log` for the `transcript:` (was STT correct?) and the
   `intent OK: … intent={…}` (what it classified).
2. If the transcript was right but the intent wrong: edit the relevant
   `fsttm/intents/*.py` few-shot/prompt (add the phrasing, tighten guidance).
3. Restart the server, retest from the TUI (wake "Nina", say it).
4. (Optional) validate accuracy headless via `scripts/opt_intent.py` if the embed/
   GPU is free — but the live TUI test is the source of truth here.
5. rsync the changed `fsttm/intents/` to the host; ask the operator to commit/push.
   Update `docs/USER_MANUAL.md` if phrases changed.

Keep changes minimal and intent/TUI-scoped. When in doubt, report rather than edit
the engine.
