"""
RunPod Serverless Handler — Blender weight painting (bone-heat autoweight).

Runs the SAME blender_autoweight.py used by the DigitalOcean Flask server, but
on a RunPod worker with enough RAM to handle ~1M+ triangle meshes (the 8 GB
droplet OOMs on those).

Large payloads do NOT travel through the RunPod job body (which is capped at a
few MB in/out). Instead:
  - INPUT  : the client uploads the gzipped mesh JSON to Backblaze B2 and passes
             us a public `input_url`. We download + gunzip it.
  - OUTPUT : we gzip the weights JSON and PUT it to a presigned `output_put_url`
             (also on B2). We return only a small summary; the client downloads
             the weights from B2 directly.

This keeps the worker credential-free — it only ever touches public/presigned
URLs that the Next.js server minted.

Job input shape:
  {
    "input_url": "https://.../input.json.gz",      # preferred (large meshes)
    "output_put_url": "https://.../output.json.gz?X-Amz...",  # presigned PUT
    "output_content_type": "application/octet-stream",        # must match presign
    "timeout": 1800,                                # optional, seconds

    # --- OR, for small meshes / local testing, inline: ---
    "vertices": [[x,y,z], ...],
    "triangles": [[i,i,i], ...],
    "bones": [{ "name", "head", "tail", "parent" }, ...]
    # --- OR a self-contained gzipped+base64 blob: ---
    "mesh_gzip_b64": "<base64 of gzip of {vertices,triangles,bones} json>"
  }

Returns (with output_put_url): a small summary { ok, output_uploaded, weight_method,
bone_count, diagnostics, elapsed, ... }. Without it: the full weights inline
(only allowed for small results).
"""

import os
import json
import time
import gzip
import base64
import subprocess
import tempfile
import traceback
import urllib.request

import runpod

BLENDER = os.environ.get("BLENDER_BIN", "/opt/blender/blender")
SCRIPT = os.environ.get("AUTOWEIGHT_SCRIPT", "/app/blender_autoweight.py")
DEFAULT_TIMEOUT = int(os.environ.get("AUTOWEIGHT_TIMEOUT", "1800"))
# Hard ceiling on inline (no-B2) results so we never blow the RunPod result cap.
MAX_INLINE_OUTPUT = int(os.environ.get("MAX_INLINE_OUTPUT", str(6 * 1024 * 1024)))


def _download(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "blender-weights-worker"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _maybe_gunzip(raw):
    # gzip magic bytes 1f 8b
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        return gzip.decompress(raw)
    return raw


def _put(url, data, content_type, timeout=300):
    req = urllib.request.Request(url, data=data, method="PUT")
    req.add_header("Content-Type", content_type)
    req.add_header("Content-Length", str(len(data)))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def _load_input(ji):
    if ji.get("input_url"):
        raw = _maybe_gunzip(_download(ji["input_url"]))
        return json.loads(raw)
    if ji.get("mesh_gzip_b64"):
        return json.loads(gzip.decompress(base64.b64decode(ji["mesh_gzip_b64"])))
    return {
        "vertices": ji["vertices"],
        "triangles": ji["triangles"],
        "bones": ji["bones"],
    }


def handler(job):
    t0 = time.time()
    ji = job.get("input", {}) or {}

    try:
        data = _load_input(ji)
    except Exception as e:
        return {"error": f"input load failed: {e}", "traceback": traceback.format_exc()}

    for k in ("vertices", "triangles", "bones"):
        if k not in data:
            return {"error": f"missing '{k}' in mesh input"}

    V = len(data["vertices"])
    T = len(data["triangles"])
    B = len(data["bones"])
    print(f"[handler] {V} verts, {T} tris, {B} bones", flush=True)

    timeout = int(ji.get("timeout", DEFAULT_TIMEOUT))

    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "input.json")
        out = os.path.join(td, "output.json")
        with open(inp, "w") as f:
            json.dump(data, f)

        try:
            proc = subprocess.run(
                [BLENDER, "--background", "--python", SCRIPT, "--", inp, out],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"blender timed out after {timeout}s", "verts": V, "tris": T}

        if not os.path.exists(out):
            return {
                "error": "blender produced no output file",
                "exit_code": proc.returncode,
                "stdout": (proc.stdout or "")[-4000:],
                "stderr": (proc.stderr or "")[-4000:],
            }

        with open(out, "rb") as f:
            out_bytes = f.read()

    try:
        result_obj = json.loads(out_bytes)
    except Exception:
        result_obj = None

    if isinstance(result_obj, dict) and result_obj.get("error"):
        return {
            "error": f"autoweight failed: {result_obj['error']}",
            "traceback": result_obj.get("traceback"),
        }

    out_put_url = ji.get("output_put_url")
    if out_put_url:
        ct = ji.get("output_content_type", "application/octet-stream")
        gz = gzip.compress(out_bytes, 6)
        try:
            _put(out_put_url, gz, ct)
        except Exception as e:
            return {"error": f"output upload failed: {e}", "traceback": traceback.format_exc()}

        summary = {}
        if isinstance(result_obj, dict):
            summary = {
                "weight_method": result_obj.get("weight_method"),
                "bone_count": result_obj.get("bone_count"),
                "diagnostics": result_obj.get("diagnostics"),
                "elapsed": result_obj.get("elapsed"),
            }
        return {
            "ok": True,
            "output_uploaded": True,
            "output_bytes": len(gz),
            "output_gzip": True,
            "verts": V, "tris": T, "bones": B,
            "handler_elapsed": round(time.time() - t0, 2),
            **summary,
        }

    # No B2 target — return inline (small meshes / testing only).
    if len(out_bytes) > MAX_INLINE_OUTPUT:
        return {
            "error": (
                f"weights too large to return inline ({len(out_bytes)} bytes); "
                "provide output_put_url so the worker can upload to B2"
            ),
            "verts": V, "tris": T,
        }
    inline = result_obj if result_obj is not None else {}
    inline["handler_elapsed"] = round(time.time() - t0, 2)
    return inline


print("[handler] blender-weights worker starting...", flush=True)
runpod.serverless.start({"handler": handler})
