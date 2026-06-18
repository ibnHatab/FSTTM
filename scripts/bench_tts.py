import time, json, numpy as np
import onnxruntime as ort

MODEL = "models/en_US-lessac-medium.onnx"
cfg = json.load(open(MODEL + ".json"))
sr = cfg["audio"]["sample_rate"]
inf = cfg.get("inference", {})
ns, ls, nw = inf.get("noise_scale",0.667), inf.get("length_scale",1.0), inf.get("noise_w",0.8)

PHONEME_LEN = 120
# Inspect input names from a CPU session
probe = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
names = [i.name for i in probe.get_inputs()]
print("model inputs:", names)
ids = np.random.randint(1, 50, size=(1, PHONEME_LEN)).astype(np.int64)
lengths = np.array([PHONEME_LEN], dtype=np.int64)
scales = np.array([ns, ls, nw], dtype=np.float32)
pool = {"input": ids, "input_lengths": lengths, "scales": scales}
feeds = {n: pool[n] for n in names if n in pool}

def bench(provider, n=10):
    try:
        sess = ort.InferenceSession(MODEL, providers=[provider])
    except Exception as e:
        return None, f"init failed: {str(e)[:90]}"
    try:
        for _ in range(3): sess.run(None, feeds)
        t = time.monotonic()
        for _ in range(n): out = sess.run(None, feeds)
        dt = (time.monotonic()-t)/n*1000
    except Exception as e:
        return None, f"run failed: {str(e)[:90]}"
    audio_s = out[0].size / sr
    actual = sess.get_providers()[0]
    return dt, f"[{actual}] {audio_s:.1f}s audio RTF={dt/1000/max(audio_s,0.01):.3f}"

for p in ["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"]:
    dt, info = bench(p)
    print(f"{p:28s} {dt:7.1f} ms   {info}" if dt else f"{p:28s}   SKIP   {info}")
