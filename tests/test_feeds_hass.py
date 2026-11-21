"""fsttm.feeds.hass: the entity filter and the words an entity becomes."""
from fsttm.feeds.hass import DEFAULTS, wanted, words


def test_filter():
    cfg = dict(DEFAULTS)
    assert wanted("fan.lab_supply_fan", cfg)
    assert not wanted("sensor.sun_next_dawn", cfg)
    assert not wanted("number.lab_display_brightness", cfg)
    assert not wanted("person.nina", cfg)


def test_words():
    cfg = dict(DEFAULTS)
    fan = {"entity_id": "fan.lab_supply_fan", "state": "on",
           "attributes": {"friendly_name": "lab Supply fan", "percentage": 40, "preset_mode": "night",
                          "preset_modes": ["night", "boost"]}}
    assert words(fan, cfg) == "lab Supply fan: on (percentage 40%, preset mode night)"
    t = {"entity_id": "sensor.lab_outside_temperature", "state": "27.8",
         "attributes": {"friendly_name": "lab Outside temperature", "unit_of_measurement": "°C"}}
    assert words(t, cfg) == "lab Outside temperature: 27.8 °C"
    light = {"entity_id": "light.desk", "state": "on", "attributes": {"brightness": 128}}
    assert words(light, cfg) == "light.desk: on (brightness 50%)"
