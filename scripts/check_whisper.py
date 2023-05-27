import os
import numpy as np
import whispercpp as w

m = w.Whisper.from_pretrained("small.en")

params = m.params.with_print_realtime(True).build()
print(params)
print(m.context.full(params, w.api.load_wav_file('samples/jfk.wav').mono))


