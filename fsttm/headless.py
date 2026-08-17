"""
Headless FSM test: stdin text → FSM + LLM → stdout.
No audio hardware, no STT (whisper), no TTS (piper).

Tests the N09-1071 turn-taking state machine with a real LLM in the loop.

Usage:
    python -m fsttm.headless --config config.sample.yaml

Then type messages and press Enter. The FSM manages floor transitions and
the LLM generates responses. Barge-in is simulated by typing while the
model is generating (send 'STOP' to cancel).

FSM state is shown at each prompt: [USER] [FREEu] [SYSTEM] [FREEs] [BOTHs]
"""
import asyncio
import sys
import threading
import time
from fsttm.fsttm import Model as FSM
from fsttm.llama import (
    Initialize, Generate, IntentGenerate, ManualGenerate, StopGenerate, AddSystem,
    Response, ResponseDone, IntentResult, LlamaError,
    make_driver,
)

_MANUAL_INTENTS = {'HOWTO', 'LOCATE', 'EXPLAIN'}

import reactivex as rx
from reactivex.subject import Subject
import reactivex.operators as ops

# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a concise voice assistant. "
    "Answer in one or two short sentences only. "
    "No lists, no markdown, no thinking steps."
)


async def run_headless(model_path: str, system_prompt_override: str = None,
                       intent_mode: bool = False, domains=None, retriever=None,
                       gpt_cfg=None):
    loop = asyncio.get_event_loop()
    turn = FSM()

    global SYSTEM_PROMPT
    if intent_mode and not system_prompt_override:
        # Assemble the intent prompt from the enabled domains (climate/lights/
        # body) so the headless test teaches exactly what the grammar allows.
        from fsttm import intents
        SYSTEM_PROMPT = intents.build_prompt(domains)
        doms = domains or intents.INTENT_DOMAINS
        print(f"[intent] domains: {', '.join(doms)}", flush=True)
    elif system_prompt_override:
        SYSTEM_PROMPT = system_prompt_override

    def _log(who, action, had):
        gained = not had
        print(f"\n  [FSM] {who} {'GAINS' if gained else 'releases'} floor "
              f"(action={action}) → state={turn.state}", flush=True)

    turn.system_cb = lambda a, h: _log('SYSTEM', a, h)
    turn.user_cb   = lambda a, h: _log('USER', a, h)

    # ── LLM driver wired to a Subject ────────────────────────────────────────
    llm_subject: Subject = Subject()
    llm_driver_fn = make_driver(loop)
    from fsttm.llama import Sink as LlamaSink
    llm_source = llm_driver_fn.call(LlamaSink(request=llm_subject))

    current_response = []
    generating = threading.Event()
    _intent_mode = intent_mode

    def on_llm_event(item):
        if type(item) is Response:
            current_response.append(item.text)
            print(item.text, end='', flush=True)
        elif type(item) is ResponseDone:
            generating.clear()
            current_response.clear()
            print()
            try:
                turn.system_action('R')
            except Exception:
                pass
        elif type(item) is IntentResult:
            generating.clear()
            print(f"\n  JSON  → {item.intent_json}")
            print(f"  Voice → {item.tts_text!r}")
            ij = item.intent_json or {}
            is_manual = isinstance(ij, dict) and ij.get('intent') in _MANUAL_INTENTS
            if is_manual and retriever is not None:
                # Manual RAG: retrieve → grounded answer (printed as the response)
                from fsttm.rag import build_answer_prompt
                query = ij.get('topic') or ''
                context, hits = retriever.context(query)
                if context:
                    pages = sorted({h[2].get('page') for h in hits})
                    print(f"  Manual→ {len(hits)} passages pp.{pages}")
                    prompt = build_answer_prompt("Nina", query, context)
                    generating.set()
                    print("  Answer→ ", end='', flush=True)
                    llm_subject.on_next(ManualGenerate(prompt=prompt, context='manual'))
                    return   # ResponseDone (below) prints the streamed answer
                else:
                    print(f"  Manual→ (no passages for {query!r})")
            elif item.intent_json:
                try:
                    from fsttm.grammar import intent_to_protocol_cmd
                    cmds = intent_to_protocol_cmd(item.intent_json)
                    print(f"  Proto → {cmds}")
                except Exception as e:
                    print(f"  Proto → (error: {e})")
            print()
            try:
                turn.system_action('R')
            except Exception:
                pass
        elif type(item) is LlamaError:
            generating.clear()
            print(f"\n[LLM ERROR] {item.error}", flush=True)
            try:
                turn.system_action('R')
            except Exception:
                pass

    llm_source.system.subscribe(on_next=on_llm_event,
                                on_error=lambda e: print(f"[ERROR] {e}"))

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    # n_ctx/n_batch must come from config: the intent base prompt (system +
    # domain prompt + few-shot) exceeds the 2048 default, and the last eval
    # chunk then fails with "llama_decode returned 1". Mirrors server.py.
    g = gpt_cfg or {}
    llm_subject.on_next(Initialize(model_path=model_path,
                                   n_ctx=g.get('n_ctx', 2048),
                                   n_batch=g.get('n_batch', 512),
                                   n_threads=g.get('n_threads', 6),
                                   n_gpu_layers=g.get('n_gpu_layers', 99)))
    llm_subject.on_next(AddSystem(prompt=SYSTEM_PROMPT))
    print("Model ready. Type your message (or STOP to cancel generation).\n")

    # ── Main input loop ───────────────────────────────────────────────────────
    def prompt():
        return f"[{turn.state}] > "

    while True:
        # Wait for any in-progress generation before reading next input
        while generating.is_set():
            await asyncio.sleep(0.05)
        try:
            text = await loop.run_in_executor(None, lambda: input(prompt()))
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        text = text.strip()
        if not text:
            continue

        if text.upper() == 'STOP':
            if generating.is_set():
                print("[Barge-in] Cancelling generation...")
                # Simulate barge-in: user grabs floor during system speech
                try:
                    turn.user_action('G')    # → BOTHs
                    turn.system_action('R')  # system yields → USER
                except Exception:
                    pass
                llm_subject.on_next(StopGenerate())
                generating.clear()
            else:
                print("[No active generation]")
            continue

        if text.upper() == 'STATE':
            cost = turn.system_actions_cost()
            print(f"  state={turn.state}  cost={cost}")
            continue

        # Simulate user taking the floor
        try:
            turn.user_action('G')    # user grabs (self-loop in USER, or FREEs→USER)
        except Exception:
            pass
        try:
            turn.user_action('R')    # user releases after speaking
        except Exception:
            pass

        # Check FSM cost: should system respond?
        if turn.state not in ('FREEu', 'USER'):
            print(f"[FSM] System cannot respond in state {turn.state}")
            continue

        # System grabs floor to generate response
        try:
            turn.system_action('G')
        except Exception as e:
            print(f"[FSM] Cannot grab floor: {e}")
            continue

        generating.set()
        current_response.clear()
        print(f"[assistant] ", end='', flush=True)
        if _intent_mode:
            llm_subject.on_next(IntentGenerate(text=text, context=None,
                                               domains=domains))
        else:
            llm_subject.on_next(Generate(text=text, context=None))


def main():
    import argparse
    import yaml
    from fsttm.aec import EchoCancelSession

    parser = argparse.ArgumentParser("FSTTM headless LLM test")
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', default=None,
                        help='Override model path from config')
    parser.add_argument('--prompt', default=None,
                        help='Path to a text file containing the system prompt')
    parser.add_argument('--intent', action='store_true', default=False,
                        help='Enable two-pass grammar intent mode (eval+rollback)')
    parser.add_argument('--domains', default=None,
                        help='Comma-separated intent domains to enable '
                             '(climate,lights,body,manual). Default: all.')
    parser.add_argument('--manual-store', default=None,
                        help='Ingested .npz vector store (enables manual RAG)')
    parser.add_argument('--manual-embed', default=None,
                        help='Embedding GGUF for manual RAG')
    parser.add_argument('--no-aec', action='store_true', default=False,
                        help='Skip PipeWire echo-cancel setup. Headless LLM/intent '
                             'testing needs no audio devices; use this on servers '
                             '(e.g. Jetson) without a PipeWire session.')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_path = args.model or cfg['gpt']['model']

    custom_prompt = None
    if args.prompt:
        with open(args.prompt) as f:
            custom_prompt = f.read().strip()

    # AEC is only meaningful when real audio I/O is in play. Pure keyboard-driven
    # headless LLM/intent testing touches no audio devices, so allow skipping it
    # on machines without PipeWire (avoids a hard `pactl load-module` failure).
    if args.no_aec:
        from contextlib import nullcontext
        aec_ctx = nullcontext()
    else:
        aec_ctx = EchoCancelSession()

    domains = ([d.strip() for d in args.domains.split(',') if d.strip()]
               if args.domains else None)

    retriever = None
    if args.manual_store and args.manual_embed:
        from fsttm.rag import Retriever
        retriever = Retriever(args.manual_store, args.manual_embed)
        print(f"[manual] RAG ready: {args.manual_store}")

    with aec_ctx:
        asyncio.run(run_headless(model_path,
                                 system_prompt_override=custom_prompt,
                                 intent_mode=args.intent,
                                 domains=domains,
                                 retriever=retriever,
                                 gpt_cfg=cfg.get('gpt') or {}))


if __name__ == '__main__':
    main()
