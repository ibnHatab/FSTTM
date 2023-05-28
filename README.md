# The Finite-State Turn-Taking Machine

## Install

```shell
CMAKE_ARGS="-DLLAMA_CUBLAS=on -DWHISPER_CUBLAS=on" FORCE_CMAKE=1 LLAMA_CUBLAS=1 WHISPER_CUBLAS=1 pip install -r requirements.txt
```


## Turn floor
https://aclanthology.org/N09-1071.pdf

The introduction to Reactive Programming you've been missing
https://gist.github.com/staltz/868e7e9bc2a7b8c1f754


## Rule based
https://www.lri.fr/~mandel//publications/BaudartHirzelMandelShinnarSimeon-REBLS-2018.pdf
https://www.lri.fr/~mandel//publications/BaudartHirzelMandelShinnarSimeon-REBLS-2018.pdf

https://socraticmodels.github.io/

## Streams
https://github.com/yarray/frpy
https://github.com/ggerganov/whisper.cpp.git
https://github.com/mriehl/fysom
https://aclanthology.org/N09-1071.pdf

## HW section
 https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/
https://www.businessrulesgroup.org/brmanifesto.htm
https://gist.github.com/staltz/868e7e9bc2a7b8c1f754

## GPT Models

High-performance inference of OpenAI's Whisper automatic speech recognition (ASR) model
 https://github.com/ggerganov/whisper.cpp

Run the LLaMA model using 4-bit integer quantization
 https://github.com/ggerganov/llama.cpp

Open-source assistant-style large language model based on GPT-J and LLaMa
 https://github.com/nomic-ai/gpt4all

Gpt4All Web UI Flask web application
 https://github.com/nomic-ai/gpt4all-ui

Python library for defining AI personalities for AI-based models
 https://github.com/ParisNeo/PyAIPersonality

## Fine tunning
RedPajama-INCITE-3B, an LLM for everyone
 https://www.together.xyz/blog/redpajama-3b-updates

 ## Glue scripts
Pybind11 bindings for whisper.cpp
 https://github.com/aarnphm/whispercpp

Python bindings for llama.cpp
 https://github.com/abdeladim-s/pyllamacpp

Python Bindings for llama.cpp
 https://github.com/abetlen/llama-cpp-python

LangChain
 https://pypi.org/project/langchain/

embedding database.
 https://github.com/chroma-core/chroma

LlamaIndex
http://alumni.media.mit.edu/~hugo/publications/papers/VLHCC2004-programmatic-semantics.pdf

## Voice activity detection (VAD)
- VAD
  https://github.com/mozilla/DeepSpeech-examples

- adjust_for_ambient_noise
  https://github.com/Uberi/speech_recognition/blob/master/speech_recognition/__init__.py

- remove speaker input using ducking from linux monitor

  : pactl list short | egrep "alsa_(input|output)" | fgrep -v ".monitor"
  : pactl load-module module-loopback
	sudo sh -c ' echo "load-module module-loopback" >>  /etc/pulse/default.pa '

- cross cancelation in time domain
/etc/pulse/default.pa
```
.ifexists module-echo-cancel.so
load-module module-echo-cancel aec_method=webrtc source_name=echocancel sink_name=echocancel1
set-default-source echocancel
set-default-sink echocancel1
.endif

```

```
#!/usr/bin/env bash
pactl unload-module module-echo-cancel
pactl load-module module-echo-cancel aec_method=webrtc source_name=echocancel sink_name=echocancel1
pacmd set-default-source echocancel
pacmd set-default-sink echocancel1
```
