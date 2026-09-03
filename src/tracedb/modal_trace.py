"""Generate a REAL PyTorch kineto trace on a Modal GPU and save it locally.

  uv run modal run src/tracedb/modal_trace.py   # writes fixtures/real_trace.json

For developing the queries against something the synthetic fixture cannot
reproduce: real kernel names, real correlation ids, and a real launch-latency
distribution. `tracedb.synth` is what the tests use -- this needs a GPU and
costs money, so it is a tool, not a fixture.

It runs on whichever cheap GPU Modal has free: the trace is about *shape*, and
none of the queries care which card produced it.
"""
import modal

app = modal.App("tracedb-gen")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch", "numpy")


@app.function(image=image, gpu=["L4", "T4", "A10G", "L40S"], timeout=1200)
def make_trace() -> bytes:
    import gzip

    import torch
    import torch.nn as nn
    from torch.profiler import ProfilerActivity, profile, schedule

    torch.manual_seed(0)
    dev = "cuda"

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, 8, batch_first=True)
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def forward(self, x):
            a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
            x = x + a
            return x + self.mlp(self.ln2(x))

    d, layers, bs, seq = 512, 4, 8, 256
    model = nn.Sequential(nn.Embedding(32000, d), *[Block(d) for _ in range(layers)],
                          nn.Linear(d, 32000)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 32000, (bs, seq), device=dev)
    y = torch.randint(0, 32000, (bs * seq,), device=dev)
    loss_fn = nn.CrossEntropyLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out.view(-1, 32000), y)
        loss.backward()
        opt.step()
        # Deliberate pathology: a blocking sync + D2H copy every step (the
        # `.item()` bug). It is what makes this trace worth looking at --
        # `tracedb idle` should blame the GPU gap on exactly this line.
        _ = loss.item()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    sched = schedule(wait=1, warmup=2, active=8, repeat=1)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=sched, record_shapes=False, profile_memory=False) as prof:
        for _ in range(11):
            step()
            prof.step()
    torch.cuda.synchronize()
    prof.export_chrome_trace("/tmp/trace.json")
    return gzip.compress(open("/tmp/trace.json", "rb").read())


@app.local_entrypoint()
def main():
    import gzip
    from pathlib import Path
    data = make_trace.remote()
    out = Path("fixtures/real_trace.json")
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(gzip.decompress(data))
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
