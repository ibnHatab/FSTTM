
import queue
import threading
import time

import ctypes
import sdl2
import sdl2.sdlmixer as sdlmixer
import array

from rhvoice_wrapper import TTS

from utils import ignore_stderr


import threading
import queue
import pyaudio

class TTSPlayThread(threading.Thread):
    """
    A class that represents a player for text-to-speech conversion.

    Attributes:
        tts (TTS): The text-to-speech engine.
        _queue (Queue): The queue to store the text to be spoken.
        _p_audio (PyAudio): The audio interface.
        _stream (Stream): The audio stream.
        _sets (dict): The settings for the text-to-speech engine.
        _work (bool): Flag indicating if the player is active.
        _clear_queue (Event): Event to clear the queue.
    """

    def __init__(self):
        super().__init__()
        self.tts = TTS(threads=3, force_process=False)
        self._queue = queue.Queue()
        self.sample_rate = 24000
        self.channels = 1

        self._sets = {
            'absolute_rate': 0.2,
            'absolute_pitch': 0.0,
            'absolute_volume': 0.0,
            'punctuation_mode': 1,
            'punctuation_list': '.,:',
            'capitals_mode': 2,
            'cap_pitch_factor':1.3,
            'voice_profile': 'SLT',
        }
        self.configure(self._sets)
        self._work = True
        self._clear_queue = threading.Event()
        self.start()

    def play_pcm_chunk(self, samples):
        pcm_data = array.array('h', samples).tobytes()
        buflen = len(pcm_data)
        c_buf = (ctypes.c_ubyte * buflen).from_buffer_copy(pcm_data)
        chunk = sdlmixer.Mix_QuickLoad_RAW(
            ctypes.cast(c_buf, ctypes.POINTER(ctypes.c_ubyte)), buflen
        )
        delay = int(buflen / self.channels / self.sample_rate * 600)
        sdlmixer.Mix_PlayChannelTimed(-1, chunk, 0, delay)

        #print(f"Delay: {delay}")
        # Wait for the sound to finish playing
        sdl2.SDL_Delay(delay)


    def configure(self, sets):
        """
        Configure the settings for the text-to-speech engine.

        Args:
            sets (dict): The settings to be configured.
        """
        self.tts.set_params(**sets)

    def clear(self):
        """
        Clear the text queue.
        """
        self._clear_queue.set()

    def _clear(self):
        """
        Clear the text queue if the clear event is set.
        """
        if self._clear_queue.is_set():
            while self._queue.qsize():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._clear_queue.clear()

    def stop(self):
        """
        Stop the player and clean up resources.
        """
        if self._work:
            self._work = False
            self._queue.put_nowait(None)
            self.join()
            self.tts.join()
            self._stream.stop_stream()
            self._stream.close()
            self._p_audio.terminate()

    def run(self):
        """
        Start the player and continuously process the text queue.
        """
        # Initialize SDL audio
        sdl2.SDL_Init(sdl2.SDL_INIT_AUDIO)
        sdlmixer.Mix_Init(sdlmixer.MIX_INIT_OGG)
        if sdlmixer.Mix_OpenAudio(self.sample_rate, sdlmixer.MIX_DEFAULT_FORMAT, self.channels, 1024) != 0:
            print("Unable to initialize audio")

        while self._work:
            self._clear()
            data = self._queue.get()
            if not data:
                break
            self._say(data)

        sdlmixer.Mix_CloseAudio()
        sdl2.SDL_Quit()

    def say(self, text: str, print_=False):
        """
        Add text to the queue to be spoken.

        Args:
            text (str): The text to be spoken.
            print_ (bool): Flag indicating if the text should be printed.
        """
        if not text:
            return
        if print_:
            print('TTS >>', text)
        self._queue.put_nowait(text)

    def _say(self, text):
        """
        Process the text and play the generated audio.

        Args:
            text (str): The text to be processed and spoken.
        """
        with self.tts.say(text, format_='pcm', buff=self.sample_rate*1024) as gen:
            for chunk in gen:
                if not self._work or self._clear_queue.is_set():
                    break
                self.play_pcm_chunk(chunk)



if __name__ == '__main__':

    player = TTSPlayThread()
    player.say('''
Q:	Why did the programmer call his mother long distance?
A:	Because that was her name.
    ''')
    time.sleep(5)
    player.say('''
There is a great discovery still to be made in Literature: that of
paying literary men by the quantity they do NOT write.
    ''')
    time.sleep(5)
    player.say('''
There is a great discovery still to be made in Literature: that of
paying literary men by the quantity they do NOT write.
    ''')

    time.sleep(13)
    player.stop()