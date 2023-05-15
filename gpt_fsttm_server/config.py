
import yaml
from typing import List, Optional
from pydantic import BaseModel


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
    tts: TTS
    server: Server
    log: Log


def parse_config(config_data):
    ''' takes a stream with the content of the configuration file as input
    and returns a (hot) stream of arguments.
    '''
    data = yaml.load(config_data, Loader=yaml.FullLoader)

    return Config(**data)



if __name__ == '__main__':
    config_data = '''
server:
  http:
    host: "0.0.0.0"
    port: 8080
    request_max_size: 1048576
log:
  level:
    - logger: fsttm_server
      level: DEBUG
tts:
    model: "tts_models/en/ljspeech/tacotron2-DDC"
'''
    config = parse_config(config_data)
    print(config)
