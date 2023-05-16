
import yaml
from typing import List, Optional
from pydantic import BaseModel

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

class Config(BaseModel):
    vad: VAD
    stt: STT
    tts: TTS
    server: Server
    log: Log


def parse_config(config_data):
    ''' takes a stream with the content of the configuration file as input
    and returns a (hot) stream of arguments.
    '''
    data = yaml.load(config_data, Loader=yaml.FullLoader)
    print(data)
    return Config(**data)



if __name__ == '__main__':

    import os
    BASE = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE, '../config.sample.yaml')) as f:
        config_data = f.read()

    config = parse_config(config_data)
    print(config)
