"""任务 #23 验证：新 280 维契约 encode/decode 双向往返 + 可渲染性。"""
import numpy as np

from dataset.generator import SceneGenerator, GeneratorConfig
from svg.scene_graph import Scene
from svg.render import render_scene_resvg
from model.targets import encode_scene, decode_scene, SLOT_DIM, NUM_SLOTS

def slots_close(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol)

stats = {"exact": 0, "near": 0, "fail": 0}
max_diff = 0.0
render_ok = 0
for seed in range(60):
    sc = SceneGenerator(GeneratorConfig(), seed=seed).sample()
    s1, bg1 = encode_scene(sc)
    assert s1.shape == (NUM_SLOTS, SLOT_DIM), s1.shape
    sc2 = decode_scene(s1, bg1)
    sc2.validate()  # 结构非法会抛异常
    s2, bg2 = encode_scene(sc2)
    d = float(np.abs(s1 - s2).max())
    max_diff = max(max_diff, d)
    if d < 1e-9:
        stats["exact"] += 1
    elif d < 1e-4:
        stats["near"] += 1
    else:
        stats["fail"] += 1
        if stats["fail"] <= 2:
            print(f"seed={seed} max_diff={d}")
    # SVG 序列化 + resvg 渲染
    from svg.serializer import serialize
    from svg.render import svg_to_rgba
    img = svg_to_rgba(serialize(sc2), 256, 256)
    if img.shape[2] == 4 and not np.isnan(img).any():
        render_ok += 1

print("roundtrip:", stats, "max_diff=%.2e" % max_diff)
print("resvg render OK:", render_ok, "/60")

# 检查全部块有非零激活（防止某块永远编码不上）
s_all = []
for seed in range(60):
    sc = SceneGenerator(GeneratorConfig(), seed=seed).sample()
    s, _ = encode_scene(sc)
    s_all.append(s.reshape(-1, SLOT_DIM))
A = np.concatenate(s_all, axis=0)
nz = [(i, float(np.abs(A[:, i]).max())) for i in range(SLOT_DIM)]
dead = [i for i, m in nz if m < 1e-6]
print("dead dims:", dead)
print("OK_23")
