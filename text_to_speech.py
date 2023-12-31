
import queue
import threading
import time
import pyaudio

from rhvoice_wrapper import TTS

from utils import ignore_stderr


import threading
import queue
import pyaudio

class Player(threading.Thread):
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
        with ignore_stderr():
            self._p_audio = pyaudio.PyAudio()
        self._stream = self._p_audio.open(
            format=self._p_audio.get_format_from_width(2),
            channels=1,
            rate=24000,
            output=True,
            start=False,
        )
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
        self._stream.start_stream()
        while self._work:
            self._clear()
            data = self._queue.get()
            if not data:
                break
            self._say(data)
        self._stream.stop_stream()

    def say(self, text: str, print_=True):
        """
        Add text to the queue to be spoken.

        Args:
            text (str): The text to be spoken.
            print_ (bool): Flag indicating if the text should be printed.
        """
        if not text:
            return
        if print_:
            print(text)
        self._queue.put_nowait(text)

    def _say(self, text):
        """
        Process the text and play the generated audio.

        Args:
            text (str): The text to be processed and spoken.
        """
        with self.tts.say(text, format_='pcm', buff=4*1024) as gen:
            for chunk in gen:
                if not self._work or self._clear_queue.is_set():
                    break
                self._stream.write(chunk)



if __name__ == '__main__':

    player = Player()
    player.say('''
Q:	Why did the programmer call his mother long distance?
A:	Because that was her name.
    ''')

    player.say('''
There is a great discovery still to be made in Literature: that of
paying literary men by the quantity they do NOT write.
    ''')

    time.sleep(3)
    player.stop()