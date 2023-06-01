import os
import numpy as np
import whispercpp as w
from whispercpp import api


MODEL_NAME = 'base.en'
model = w.Whisper.from_pretrained(MODEL_NAME)

def store_transcript_handler(ctx, n_new, data):
    print('>>', n_new, data)
    segment = ctx.full_n_segments() - n_new
    while segment < ctx.full_n_segments():
        print('>>', ctx.full_get_segment_text(segment))
        segment += 1


params = (model.params
          .with_print_realtime(True)
          .build())

transcript = []
params.on_new_segment(store_transcript_handler, transcript)

pcf32 = w.api.load_wav_file('samples/jfk.wav').mono
model.context.full(params, pcf32)


color = True
segment = 0
while segment < model.context.full_n_segments():
    if not color:
        text = model.context.full_get_segment_text(segment)
    else:
        for j in range(model.context.full_n_tokens(segment)):
#            token_data = model.context.full_get_token_data(segment, j)
            token_data = model.context.full_get_token_prob(segment, j)
            text = model.context.full_get_token_text(segment, j)
            print('>>', token_data, text)
        # model.context.full_get_token_prob

    print()
    segment += 1
