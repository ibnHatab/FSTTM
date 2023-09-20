
import yaml
from typing import List, Optional
from pydantic import BaseModel
import reactivex.operators as ops
import cyclotron_std.argparse as argparse

class VAD(BaseModel):
    vad_aggressiveness: int
    device: Optional[int]
    rate: int

class STT(BaseModel):
    model: str

class TTS(BaseModel):
    model: str
    scorer: Optional[str]
    beam_width: Optional[int]
    lm_alpha: Optional[float]
    lm_beta: Optional[float]


class HttpServer(BaseModel):
    host: str
    port: int
    request_max_size: int


class Server(BaseModel):
    http: HttpServer

class LogLevel(BaseModel):
    logger: str
    level: str

class Log(BaseModel):
    level: List[LogLevel]

class GptParams(BaseModel):
    n_ctx: int
    seed: float
    temp: float
    top_k: int
    top_p: float
    repeat_last_n: int
    n_batch: int
    repeat_penalty: float
    model: str
    n_threads: int
    n_predict: int
    safeword: str
    conversation: str

class Config(BaseModel):
    vad: VAD
    stt: STT
    tts: TTS
    gpt: GptParams
    server: Server
    log: Log


def parse_config(config_data):
    ''' takes a stream with the content of the configuration file as input
    and returns a (hot) stream of arguments .
    '''
    config = config_data.pipe(
        ops.filter(lambda i: i.id == "config"),
        ops.flat_map(lambda i: i.data),
        ops.map(lambda i: yaml.load(
            i,
            Loader=yaml.FullLoader
        )),
        ops.map(lambda i: Config(**i)),
        ops.share(),
    )

    return config

def parse_arguments(argv):
    parser = argparse.ArgumentParser("Finite-State Turn-Taking Machine")
    parser.add_argument('--config', required=True, help="Path of the server configuration file")
    return argv.pipe(
        ops.skip(1),
        argparse.parse(parser),
    )


if __name__ == '__main__':
    import reactivex as rx
    import reactivex.operators as ops

    args = rx.from_(['1', '--config', 'config.sample.yaml'])
    args.pipe(
            parse_arguments,
            ops.to_list()).subscribe(lambda i: print(f"config: {i}"))



