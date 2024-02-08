
import ctypes
import sdl2
import sdl2.sdlmixer as sdlmixer
import array
import math
from rhvoice_wrapper import TTS

def play_pcm_chunk(samples):
    pcm_data = array.array('h', samples).tobytes()
    buflen = len(pcm_data)
    c_buf = (ctypes.c_ubyte * buflen).from_buffer_copy(pcm_data)
    chunk = sdlmixer.Mix_QuickLoad_RAW(
           ctypes.cast(c_buf, ctypes.POINTER(ctypes.c_ubyte)), buflen
       )
    #sdlmixer.Mix_PlayChannel(-1, chunk, 0)
    delay = int(buflen / channels / sample_rate * 600)
    sdlmixer.Mix_PlayChannelTimed(-1, chunk, 0, delay)

    print(f"Delay: {delay}")
    sdl2.SDL_Delay(delay)


# Example PCM data (replace with your own)
sdl2.SDL_Init(sdl2.SDL_INIT_AUDIO)
sdlmixer.Mix_Init(sdlmixer.MIX_INIT_OGG)

# Initialize SDL audio
sample_rate = 24000
channels = 1
if sdlmixer.Mix_OpenAudio(sample_rate, sdlmixer.MIX_DEFAULT_FORMAT, channels, 1024) != 0:
    print("Unable to initialize audio")


tts = TTS(threads=3, force_process=False)
sets = {
            'absolute_rate': 0.2,
            'absolute_pitch': 0.0,
            'absolute_volume': 0.0,
            'punctuation_mode': 1,
            'punctuation_list': '.,:',
            'capitals_mode': 2,
            'cap_pitch_factor':1.3,
            'voice_profile': 'SLT',
            'stream': True,
        }
tts.set_params(**sets)

texts = [
    "Process the text and play the generated audio. Process the text and play the generated audio.",
    "This is a test.",
    "This is a test.",
    "",
    "This is a test.",
    "This is a test.",
]

for text in texts:
    print(text)
    with tts.say(text, format_='pcm', buff=sample_rate*1024) as gen:
        for chunk in gen:
            play_pcm_chunk(chunk)


# Clean up resources
tts.join()
sdlmixer.Mix_CloseAudio()
sdl2.SDL_Quit()

