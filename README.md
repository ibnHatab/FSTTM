
# The Finite-State Turn-Taking Machine


## Install

```shell
CMAKE_ARGS="-DLLAMA_CUBLAS=on -DWHISPER_CUBLAS=on" FORCE_CMAKE=1 LLAMA_CUBLAS=1 WHISPER_CUBLAS=1 pip install -r requirements.txt
```


## Memory management
https://arxiv.org/pdf/2308.15022.pdf


## Turn floor FSM
https://aclanthology.org/N09-1071.pdf


## Rule based
https://www.lri.fr/~mandel//publications/BaudartHirzelMandelShinnarSimeon-REBLS-2018.pdf
https://socraticmodels.github.io/
http://alumni.media.mit.edu/~hugo/publications/papers/VLHCC2004-programmatic-semantics.pdf
https://www.businessrulesgroup.org/brmanifesto.htm


## Streams
The introduction to Reactive Programming you've been missing
https://gist.github.com/staltz/868e7e9bc2a7b8c1f754
https://github.com/yarray/frpy
https://github.com/ggerganov/whisper.cpp.git
https://github.com/mriehl/fysom


## HW section
https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/


## GPT Models

High-performance inference of OpenAI's Whisper automatic speech recognition (ASR) model
https://github.com/ggerganov/whisper.cpp

Run the LLaMA model using 4-bit integer quantization
https://github.com/ggerganov/llama.cpp

Open-source assistant-style large language model based on GPT-J and LLaMa
https://github.com/nomic-ai/gpt4all

Gpt4All Web UI Flask web application
https://github.com/nomic-ai/gpt4all-ui


## Fine tunning
RedPajama-INCITE-3B, an LLM for everyone
https://www.together.xyz/blog/redpajama-3b-updates

Metharme 7B
https://huggingface.co/PygmalionAI/metharme-7b

converter
https://docs.alpindale.dev/pygmalion-7b/#file-hashes

The recommended range for temperature (for chatbots) is between 0.5 to 0.9 and the ideal range for repetition penalty is between 1.1 to 1.2.


## Glue scripts
Pybind11 bindings for whisper.cpp
https://github.com/aarnphm/whispercpp

Python bindings for llama.cpp
https://github.com/abdeladim-s/pyllamacpp

Python Bindings for llama.cpp
https://github.com/abetlen/llama-cpp-python

LangChain
https://pypi.org/project/langchain/

Embedding database.
https://github.com/chroma-core/chroma

LLamaIndex
https://github.com/jerryjliu/llama_index


## Voice activity detection (VAD)
- VAD
  https://github.com/mozilla/DeepSpeech-examples

- adjust_for_ambient_noise
  https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py

- remove speaker input using ducking from linux monitor

  : pactl list short | egrep "alsa_(input|output)" | fgrep -v ".monitor"
  : pactl load-module module-loopback
	sudo sh -c ' echo "load-module module-loopback" >>  /etc/pulse/default.pa '

- cross cancelation in time domain /etc/pulse/default.pa

```
.ifexists module-echo-cancel.so
load-module module-echo-cancel aec_method=webrtc source_name=echocancel sink_name=echocancel1
set-default-source echocancel
set-default-sink echocancel1
.endif
```

Enable echo cancelation

```
#!/usr/bin/env bash
pactl unload-module module-echo-cancel
pactl load-module module-echo-cancel aec_method=webrtc source_name=echocancel sink_name=echocancel1
pacmd set-default-source echocancel
pacmd set-default-sink echocancel1
```

- testing from monitor

In pavcontroll in Recording set sink to Monitor

```
strace -o spork tty
/dev/pts/27
fortune  |tee /dev/pts/27 | RHVoice-client  -s  SLT -r 0 -v -0.1 | aplay
```


### Voice stream
mic_vad.py

    FORMAT = pyaudio.paInt16
    # Network/VAD rate-space
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

```
        vad_audio = VADAudio(loop,
                            aggressiveness=3,
                            device=0,
                            input_rate=16000)

 'device': 0,
 'input_rate': 16000,         # rate
 'sample_rate': 16000,
 'block_size': 320,           # RATE_PROCESS / BLOCKS_PER_SECOND
 'block_size_input': 320,     # frames_per_buffer; RATE_PROCESS / BLOCKS_PER_SECOND

 ```
 len(frame) == 640
 VAD rate ~ 20 f/s

