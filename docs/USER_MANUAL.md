# Nina — Voice Assistant User Manual

Nina is the in-car voice assistant. She controls climate, lights, doors,
windows and seats, answers questions from the vehicle manual, tells the time and
date, and chats. This guide covers how to wake and dismiss her, and every command
she understands, with examples.

---

## 1. Activating and deactivating Nina

Nina starts **asleep**. While asleep she keeps listening but ignores everything
except her name.

### Wake her
Say her name, optionally with a command right after:

| You say | Result |
|---|---|
| **"Nina"** / **"Hey Nina"** / **"Hi Nina"** | Wakes; now listening for commands |
| **"Hey Nina, set temperature to 22"** | Wakes **and** runs the command in one go |

Once awake she stays awake — you do **not** repeat "Nina" before every command.
Just speak naturally.

### Dismiss her (put her back to sleep / mute)
To turn the assistant off you **must address her by name**, then ask to stop.
This is deliberate: a bare command can never accidentally disable voice control.

| You say | Result |
|---|---|
| **"Hey Nina, voice off"** | Mutes / sleeps the assistant |
| **"Hey Nina, go to sleep"** | Sleeps |
| **"Hey Nina, stop listening"** | Sleeps |
| **"Hey Nina, that's all, goodbye"** | Sleeps |

> A request to stop a **car function** is *not* a dismissal. "Stop the climate"
> or "turn it off" is treated as a normal command, even if you said "Nina" first.
> Only an explicit "voice off / mute / go to sleep" dismisses her.

After dismissal, say **"Nina"** again to wake her.

---

## 2. Zones (which seat / area)

Many commands accept a zone. Nina infers it from how you phrase the request:

| Zone | Trigger phrases |
|---|---|
| **All / everywhere** (default) | "all", "every", or no zone mentioned |
| **Driver** (front-left) | "my", "driver", "left", "my side" |
| **Passenger** (front-right) | "passenger", "right", "his/her side" |
| **Rear-left** | "rear left", "back left" |
| **Rear-right** | "rear right", "back right" |
| **Trunk** | "trunk", "boot" |

Examples: "set **my** temperature to 21" → driver. "open the **passenger** window"
→ front-right. "lock the doors" → all.

---

## 3. Climate commands

| Intent | Say something like | Notes |
|---|---|---|
| Warmer | "make it warmer", "too cold", "heat up", "warmer by two" | 1–3 steps |
| Cooler | "too hot", "cool it down", "cooler by two" | 1–3 steps |
| Set temperature | "set temperature to 22", "23 degrees please", "set my temp to 21" | 16–28 °C, zone-aware |
| Fan up | "more air", "faster fan", "blow harder" | 1–3 steps |
| Fan down | "less air", "quieter fan", "softer" | 1–3 steps |
| Set fan level | "fan level 3", "set fan to 5" | level 1–7 |
| A/C on / off | "turn on the AC", "air conditioning off" | |
| Max A/C | "max AC", "maximum cooling", "blast it" | |
| Vent → face | "air to my face", "blow on me" | |
| Vent → feet | "air to my feet", "warm my feet" | |
| Defrost windshield | "defrost the windshield", "clear the windshield" | |
| Vent split | "face and feet", "both vents" | |
| Max defrost | "max defrost", "full defrost" | |
| Recirculate on / off | "recirculate" / "fresh air" | |
| Auto / manual mode | "auto mode" / "manual mode" | |
| Sync zones | "sync zones", "link both sides", "dual" | |
| Climate power on / off | "turn on the climate", "climate off" | |
| Rear defrost | "rear window defrost" | |

---

## 4. Lights

Say which lamp you mean; Nina maps it to the right light type.

| Light | Say something like |
|---|---|
| Headlights | "headlights on / off", "lights on / off" |
| Fog lights | "fog lights on", "turn off fog" |
| Hazards | "hazard lights", "emergency lights", "flashers on" |
| Cabin / interior | "cabin light on", "interior light off", "dome light", "reading light" |

Example: **"turn on the cabin lights"** → cabin light on. **"headlights off"** →
headlights off.

---

## 5. Doors, windows & seats

| Intent | Say something like | Notes |
|---|---|---|
| Lock | "lock the doors", "secure the car" | zone-aware |
| Unlock | "unlock", "unlock my door" | zone-aware |
| Open window | "open the window", "roll down the driver window" | opens ~halfway by default |
| Close window | "close the window", "roll up", "close all windows" | |
| Seat heat up | "warm my seat", "seat warmer", "more seat heat" | 1–3 steps |
| Seat heat down | "less seat heat", "seat too warm" | 1–3 steps |
| Seat cooling up | "cool my seat", "chill my seat", "more seat cooling" | 1–3 steps |
| Seat cooling down | "less seat cooling", "seat cooling off" | 1–3 steps |

> "Cool the seat" turns seat **cooling on** (it does not mean *less* cooling).

---

## 6. Time, date & status

| Ask | Nina answers |
|---|---|
| "What time is it" / "what is the time now" / "do you have the time" | The current time, e.g. *"It's 3:42 PM."* |
| "What's the date" / "what day is it" | The date, e.g. *"It's Tuesday, June 2nd."* |
| "What's the temperature" / "what temperature is set" | Live readings, e.g. *"Set to 22 degrees on the driver side, currently 21."* |

Status answers use **real values** from the vehicle, not guesses.

---

## 7. Manual questions (RAG)

Nina can answer "how do I…", "where is…", and "what does … mean" questions from
the **vehicle manual**. She retrieves the relevant manual passages and answers
only from them — if the manual doesn't cover it, she says so rather than guessing.

| Type | Say something like |
|---|---|
| How-to | "how do I open the trunk", "how do I charge the car", "how to pair my phone" |
| Locate | "where is the trunk release", "where is the charging port", "where is the hazard button" |
| Explain | "explain the tyre-pressure warning light", "what is that warning light" |

**Example**

> **You:** "Hey Nina, how do I open the trunk?"
> **Nina:** *"On the trunk lid trim panel."* (grounded in the manual)

### The manual document
- The answers come from an ingested PDF vehicle manual (the **Taycan manual** in
  this deployment) — converted to a searchable vector store.
- **Intent:** manual questions are classified as `HOWTO`, `LOCATE`, or `EXPLAIN`,
  which route to the retrieval path instead of a car command.
- To use a **different manual**, ingest its PDF:
  ```
  python -m fsttm.rag.ingest <manual.pdf> \
      --embed models/nomic-embed-text-v1.5.Q4_K_M.gguf \
      --out models/<manual>.npz --source "<name>"
  ```
  then point `system.manual_store` at the new `.npz` and restart.

---

## 8. Chit-chat & out-of-scope

| You say | Nina |
|---|---|
| "Hello Nina", "how are you", "thank you", "good morning" | Replies warmly (conversational) |
| "What's the weather", "tell me a joke", "call my wife", "navigate home" | Politely declines — outside the car's control |

---

## 9. Quick reference

- **Wake:** "Nina" / "Hey Nina"
- **One-shot:** "Hey Nina, <command>"
- **Sleep:** "Hey Nina, voice off" (only an explicit dismissal sleeps her)
- **Climate:** warmer · cooler · set temperature · fan · A/C · vents · defrost
- **Lights:** headlights · fog · hazards · cabin
- **Body:** lock/unlock · windows · seat heat/cool
- **Info:** time · date · temperature status
- **Manual:** how-to · where-is · explain (from the vehicle manual)
- **Zones:** "my/driver" · "passenger" · "rear left/right" · "all"
