# Blender Weight Painting — RunPod Serverless Worker

Headless Blender bone-heat weight painting on a RunPod serverless endpoint, so
high-poly meshes (~1M+ triangles) that OOM the 8 GB DigitalOcean droplet
(`asitplays.com`) can be rigged. Runs the **exact same** `blender_autoweight.py`
as the DO server.

## Why this exists

The studio rig pipeline used to auto-decimate >200k-vert meshes to a ~200k voxel
proxy before painting, then transfer weights back — which caused stretched/pulled
polygons on thin features. We removed that decimation so meshes are painted at
their real resolution, but full-res 1M-tri jobs need far more RAM than the
droplet has. This endpoint provides that RAM.

## Architecture (large payloads go through Backblaze B2, not the job body)

```
RigUnified (browser)
  │ gzip {vertices,triangles,bones} → presigned PUT → B2 (public input_url)
  ▼
/api/autoweight-runpod/run
  │ presign an OUTPUT PUT url on B2, then POST RunPod /run:
  │   input = { input_url, output_put_url, output_content_type }
  ▼
THIS WORKER (baked headless Blender)
  │ GET input_url → gunzip → blender --background --python blender_autoweight.py
  │ gzip weights → PUT to output_put_url (B2)
  ▼ returns small summary { ok, weight_method, bone_count, diagnostics }
/api/autoweight-runpod/status  ◄── client polls
  ▼ client GETs weights.json.gz from B2 public url → gunzip → use
```

The worker is **credential-free**: it only touches the public `input_url` and the
presigned `output_put_url` that the Next.js server mints. Bone-heat is CPU + RAM
bound, so there is no GPU/CUDA dependency.

## Files

| File | Purpose |
|---|---|
| `handler.py` | RunPod serverless entrypoint (download → blender → upload). |
| `blender_autoweight.py` | Copy of `deploy/blender_autoweight.py` (the bone-heat script). Keep in sync. |
| `Dockerfile` | `python:3.11-slim` + baked Blender 4.2 LTS + `runpod`. |

> **Keep `blender_autoweight.py` in sync** with `deploy/blender_autoweight.py`.
> Re-copy it (and push) whenever the DO version changes:
> `Copy-Item deploy\blender_autoweight.py runpod\blender-weights\blender_autoweight.py -Force`

## Deploy

RunPod builds the image directly from a **public GitHub repo** (same model as the
CorridorKey worker — see `docs/runpod-corridor-key-setup.md`). No local Docker
needed.

### 1. Push these files to a dedicated repo

The build context must be the repo root (Dockerfile + handler.py +
blender_autoweight.py at the top level). Suggested repo:
`https://github.com/romesvonwolf/blender-weights-serverless`

```bash
# from a temp copy of THIS folder's contents at the repo root
git init && git add . && git commit -m "Blender weight-painting serverless worker"
git branch -M main
git remote add origin https://github.com/romesvonwolf/blender-weights-serverless.git
git push -u origin main
```

### 2. Create the RunPod serverless endpoint

1. Go to https://www.runpod.io/console/serverless → **New Endpoint**.
2. Container image: choose **GitHub repo build** → `romesvonwolf/blender-weights-serverless` (branch `main`, Dockerfile at root).
3. Configure:
   - **Name**: `blender-weights`
   - **Worker type / GPU**: cheapest available is fine (this is CPU work). Pick a SKU whose worker has **≥ 32 GB system RAM** (e.g. a 24 GB GPU class, or a CPU endpoint). RAM is the constraint, not VRAM.
   - **Min Workers**: 0
   - **Max Workers**: 1–2
   - **Idle Timeout**: 10 s
   - **Execution Timeout**: 1800 s (30 min) — full-res 1M-tri heat solve is slow.
   - **FlashBoot**: ON (fast warm starts).
4. Create it and copy the **Endpoint ID**.

### 3. Configure the app

Add to `.env.local`:

```
RUNPOD_AUTOWEIGHT_ENDPOINT_ID=<endpoint_id>
# RUNPOD_API_KEY already exists (shared with CorridorKey / HY-Motion)
```

Restart the dev server.

## Test the endpoint directly

```bash
# health
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/health \
  -H "Authorization: Bearer <RUNPOD_API_KEY>"

# tiny inline job (single tri, two bones)
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input":{"vertices":[[0,0,0],[1,0,0],[0,1,0]],"triangles":[[0,1,2]],"bones":[{"name":"root","head":[0,0,0],"tail":[0,0,1],"parent":null},{"name":"pelvis","head":[0,0,1],"tail":[0,0,2],"parent":"root"}]}}'
```

A first call from cold start includes image pull + Blender launch; subsequent warm
calls skip that.
