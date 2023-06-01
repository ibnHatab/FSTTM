
import queue
import threading
import pyaudio

from rhvoice_wrapper import TTS

from gpt_fsttm_server.utils import ignoreStderr


class Player(threading.Thread):
    def __init__(self):
        super().__init__()
        self.tts = TTS(threads=3, force_process=False)
        self._queue = queue.Queue()
        with ignoreStderr():
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
        self.tts.set_params(**sets)

    def clear(self):
        self._clear_queue.set()

    def _clear(self):
        if self._clear_queue.is_set():
            while self._queue.qsize():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._clear_queue.clear()

    def stop(self):
        if self._work:
            self._work = False
            self._queue.put_nowait(None)
            self.join()
            self.tts.join()
            self._stream.stop_stream()
            self._stream.close()
            self._p_audio.terminate()

    def run(self):
        self._stream.start_stream()
        while self._work:
            self._clear()
            data = self._queue.get()
            if not data:
                break
            self._say(data)
        self._stream.stop_stream()           

    def say(self, text: str, print_=True):
        if not text:
            return
        if print_:
            print(text)
        self._queue.put_nowait(text)

    def _say(self, text):
        with self.tts.say(text, format_='pcm', buff=4*1024) as gen:
            for chunk in gen:
                if not self._work or self._clear_queue.is_set():
                    break
                self._stream.write(chunk)



if __name__ == '__main__':

    player = Player()        
#     player.say('''
# Q:	Why did the programmer call his mother long distance?
# A:	Because that was her name.
#     ''')
    player.say('''
There is a great discovery still to be made in Literature: that of
paying literary men by the quantity they do NOT write.
    ''')

#     player.say('''
#  The bone-chilling scream split the warm summer night in two, the first
# half being before the scream when it was fairly balmy and calm and
# pleasant, the second half still balmy and quite pleasant for those who
# hadn't heard the scream at all, but not calm or balmy or even very nice
# for those who did hear the scream, discounting the little period of time
# during the actual scream itself when your ears might have been hearing it
# but your brain wasn't reacting yet to let you know.
# 		-- Winning sentence, 1986 Bulwer-Lytton bad fiction contest   
#     ''')