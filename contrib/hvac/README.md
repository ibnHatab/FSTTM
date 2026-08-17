# fsttm-hvac

HVAC / vehicle intent domain for the [FSTTM](../../README.md) spoken dialog
engine, plus the `hvac-react/` simulated car cockpit (FastAPI VHAL backend +
React UI) it talks to.

```bash
pip install -e .            # registers the "hvac" domain entry point
bash hvac-react/start.sh    # backend :8000 + UI :5173
```

Enable in the engine config:

```yaml
system:
  intent_mode: true
  domain: "hvac"
domains:
  hvac:
    backend_url: "http://127.0.0.1:8000"
```
