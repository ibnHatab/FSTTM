import os
import numpy as np
import whispercpp as w
from whispercpp import api
from funcy import print_durations
import datetime

start = datetime.datetime.now()
def ms():
    df = (datetime.datetime.now() - start)
    return df.seconds + df.microseconds / 1000000

MODEL_NAME = 'tiny.en'
model = w.Whisper.from_pretrained(MODEL_NAME)

def store_transcript_handler(ctx, n_new, data):
    print(f'{ms()} CB >>                           ', n_new, data)
    segment = ctx.full_n_segments() - n_new
    while segment < ctx.full_n_segments():
        print(f'{ms()} CB SEG >>', ctx.full_get_segment_text(segment))
        segment += 1


params = (model.params
          .with_print_realtime(False)
          .build())
print(params)
transcript = []
params.on_new_segment(store_transcript_handler, transcript)

pcf32 = w.api.load_wav_file('samples/12.16.wav').mono

print(f'{ms()} START {len(pcf32)/640} ')
@print_durations
def run():
    model.context.full(params, pcf32)

run()

color = True

for segment in range(model.context.full_n_segments()):

    if not color:
        print(f'>> {segment}  [{model.context.full_get_segment_start(segment)} --> {model.context.full_get_segment_end(segment)}]  {model.context.full_get_segment_text(segment)}')
    else:
        probs = []
        for token in range(model.context.full_n_tokens(segment)):
            id = model.context.full_get_token_id(segment, token)
            text = model.context.full_get_token_text(segment, token)
            if id >= model.context.eot_token:
                print('>> skip:', id, text)
                continue
            token_data = model.context.full_get_token_data(segment, token)
            probs.append(token_data.p)
            print(f'{ms()} {(segment, token)} TXT>>',  text)

        probs_str = ' '.join([f'{p:.2f}' for p in probs])
        print(f'>> {segment}', probs_str)
