"""Read only: validate full checkpoint headers without loading model tensors."""

import argparse
import hashlib
import json
from pathlib import Path
import struct


def verify(root, full_hash=False):
    root = Path(root).resolve()
    cfg = json.loads((root / "config.json").read_text())
    for key, value in {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "vision_n_layers": 32,
        "vision_dim": 1024,
        "vision_n_heads": 16,
        "vision_max_n_token": 384,
        "num_nextn_predict_layers": 3,
        "dspark_block_size": 5,
    }.items():
        if cfg.get(key) != value:
            raise ValueError(f"{key}: expected {value!r}, got {cfg.get(key)!r}")
    if cfg.get("max_position_embeddings", 0) < 262144:
        raise ValueError("模型上下文不足 256K")
    idx = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    # The source index is authoritative for this pinned model, but does not load weights.
    ref = (
        Path(__file__).resolve().parents[1]
        / "source/model-metadata/model.safetensors.index.json"
    )
    expected = json.loads(ref.read_text())["weight_map"]
    if idx != expected:
        raise ValueError(
            "权重索引与 R3.4 固定 Vision-Exp 版本不一致；请保留诊断结果核对，勿替换为文本模型"
        )
    shards = sorted(set(idx.values()))
    total = 0
    checked = []
    if len(shards) != 48:
        raise ValueError(f"expected 48 shards, got {len(shards)}")
    for name in shards:
        p = root / name
        # Local ModelScope/HF cache symlinks are allowed when present and readable.
        size = p.stat().st_size
        with p.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError("truncated shard: " + name)
            hlen = struct.unpack("<Q", raw)[0]
            if not 2 <= hlen <= 64 * 1024 * 1024 or hlen + 8 > size:
                raise ValueError("bad header: " + name)
            header = json.loads(f.read(hlen))
        keys = set(header) - {"__metadata__"}
        if keys != {k for k, v in idx.items() if v == name}:
            raise ValueError("header/index mismatch: " + name)
        offsets = sorted(tuple(header[k]["data_offsets"]) for k in keys)
        cursor = 0
        for begin, end in offsets:
            if begin != cursor or end < begin:
                raise ValueError("tensor offset gap/overlap: " + name)
            cursor = end
        if hlen + 8 + cursor != size:
            raise ValueError("truncated/extra shard data: " + name)
        row = {"file": name, "bytes": size, "tensor_headers": len(keys)}
        if full_hash:
            with p.open("rb") as f:
                row["sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        checked.append(row)
        total += size
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        if not (root / name).is_file():
            raise ValueError("missing " + name)
    tk = json.loads((root / "tokenizer.json").read_text())
    tokens = {t["content"]: t["id"] for t in tk["added_tokens"]}
    for i in range(5):
        assert tokens[f"<|place_holder_mm_span_{431 + i:04}|>"] == 129257 + i
    return {
        "status": "PASS",
        "scope": "checkpoint headers and index; SHA256 is an inventory unless compared to a trusted source",
        "weight_bytes": total,
        "shards": checked,
        "vision_tensor_count": sum(k.startswith("vision.") for k in idx),
        "bias_vl_count": sum("bias_vl" in k for k in idx),
        "mtp_tensor_count": sum(k.startswith("mtp.") for k in idx),
        "original_model_files_modified": False,
        "gpu_inference_tested": False,
    }


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("path")
    a.add_argument("--full-hash", action="store_true")
    args = a.parse_args()
    print(json.dumps(verify(args.path, args.full_hash), ensure_ascii=False, indent=2))
