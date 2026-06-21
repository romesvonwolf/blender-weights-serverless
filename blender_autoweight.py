#!/usr/bin/env python3
"""
Blender automatic weight computation via ARMATURE_AUTO (bone heat diffusion).

Builds a proper connected armature from the provided bone hierarchy, creates
a mesh from raw vertices/triangles, repairs mesh topology for reliable heat
diffusion, parents the mesh to the armature with automatic weights, applies
moderate vertex group smoothing, and returns per-vertex per-bone weight data.

KEY FEATURE: Two-phase post-processing to prevent weight bleeding:
Phase 1 — Island clamping: disconnected mesh parts (via edge connectivity)
are separate "islands". Bones can only influence their own island.
Phase 2 — K-nearest clamping: for each vertex, only the K nearest non-trunk
bones (by Euclidean distance to bone segment) keep their weights. All other
non-trunk bone weights are stripped. This prevents heat from bleeding
through mesh connectivity to reach distant bones (e.g. forearm painting
a hip satchel). Trunk bones (spine/pelvis/neck/head) are exempt.

Input JSON (via argv):
  {
    "vertices": [[x, y, z], ...],
    "triangles": [[i0, i1, i2], ...],
    "bones": [
      { "name": "pelvis", "head": [x,y,z], "tail": [x,y,z], "parent": null },
      { "name": "spine_01", "head": [x,y,z], "tail": [x,y,z], "parent": "pelvis" },
      ...
    ]
  }

Output JSON:
  {
    "weights": { "bone_name": { "vertex_index": weight, ... }, ... },
    "weight_method": "ARMATURE_AUTO" | "ARMATURE_ENVELOPE",
    "bone_count": int,
    "diagnostics": { ... },
    "elapsed": float
  }

Run:
  blender --background --python blender_autoweight.py -- input.json output.json
"""

import bpy
import bmesh
import sys
import json
import traceback
import time
from mathutils import Vector
from mathutils.kdtree import KDTree


def log(msg):
    print(f"[autoweight] {msg}")


def point_to_segment_dist(p, a, b):
    """Distance from point p to line segment a-b."""
    ab = b - a
    ab_sq = ab.dot(ab)
    if ab_sq < 1e-12:
        return (p - a).length
    t = max(0.0, min(1.0, (p - a).dot(ab) / ab_sq))
    closest = a + ab * t
    return (p - closest).length


def repair_mesh(mesh_obj):
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.fill_holes(sides=64)
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh_obj.select_set(False)
    stats = mesh_obj.data
    log(f"Mesh repaired: {len(stats.vertices)} verts, {len(stats.polygons)} faces, {len(stats.edges)} edges")
    return len(stats.vertices)


def build_armature(bones):
    armature_data = bpy.data.armatures.new('WeightArmature')
    armature_obj = bpy.data.objects.new('WeightArmature', armature_data)
    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bone_map = {}
    parent_map = {}
    for b in bones:
        parent_map[b['name']] = b.get('parent')

    ordered = []
    visited = set()
    def visit(name):
        if name in visited:
            return
        parent = parent_map.get(name)
        if parent and parent not in visited:
            visit(parent)
        visited.add(name)
        ordered.append(name)
    for b in bones:
        visit(b['name'])

    bone_data = {b['name']: b for b in bones}

    for name in ordered:
        if name not in bone_data:
            continue
        b = bone_data[name]
        eb = armature_data.edit_bones.new(name)
        head = Vector(b['head'])
        tail = Vector(b['tail'])

        bone_len = (tail - head).length
        if bone_len < 0.001:
            tail = head + Vector((0, 0.02, 0))
            bone_len = 0.02

        eb.head = head
        eb.tail = tail
        eb.use_deform = True

        eb.envelope_distance = max(bone_len * 0.5, 0.02)
        eb.envelope_weight = 1.0

        if b.get('parent') and b['parent'] in bone_map:
            parent_eb = bone_map[b['parent']]
            eb.parent = parent_eb
            if (eb.head - parent_eb.tail).length < 0.01:
                eb.use_connect = True

        bone_map[name] = eb

    bpy.ops.object.mode_set(mode='OBJECT')
    log(f"Armature created: {len(armature_data.bones)} bones")
    return armature_obj


def try_auto_weight(mesh_obj, armature_obj):
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    try:
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        log("ARMATURE_AUTO (bone heat) weighting succeeded")
        return 'ARMATURE_AUTO'
    except RuntimeError as e:
        if 'Bone Heat' not in str(e):
            raise
        log(f"Bone Heat failed: {e}")

    log("Retrying ARMATURE_AUTO after additional mesh cleanup...")
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh_obj.select_set(False)

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    try:
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        log("ARMATURE_AUTO succeeded on retry (after triangulation)")
        return 'ARMATURE_AUTO'
    except RuntimeError as e2:
        log(f"Bone Heat failed again: {e2}, using ARMATURE_ENVELOPE fallback")

    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')
    log("ARMATURE_ENVELOPE weighting applied as fallback")
    return 'ARMATURE_ENVELOPE'


def find_mesh_islands(mesh_obj):
    """Find disconnected mesh islands using edge connectivity (union-find).
    Returns a dict mapping vertex index → island id (int starting from 0).
    """
    mesh = mesh_obj.data
    V = len(mesh.vertices)

    parent = list(range(V))
    rank = [0] * V

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for edge in mesh.edges:
        union(edge.vertices[0], edge.vertices[1])

    root_to_id = {}
    vert_island = {}
    next_id = 0
    for vi in range(V):
        root = find(vi)
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1
        vert_island[vi] = root_to_id[root]

    return vert_island, next_id


def assign_bones_to_islands(mesh_obj, armature_obj, vert_island, num_islands):
    """For each bone, determine which mesh island it belongs to.

    Uses the bone's midpoint and finds the closest vertex in the mesh.
    That vertex's island becomes the bone's island. Returns dict of
    bone_name → island_id.

    Trunk bones (spine, neck, head, pelvis) are assigned to ALL islands
    since they legitimately influence the entire body.
    """
    mesh = mesh_obj.data
    verts = mesh.vertices
    arm_data = armature_obj.data

    kd = KDTree(len(verts))
    for vi, v in enumerate(verts):
        kd.insert(v.co, vi)
    kd.balance()

    trunk_bones = {
        'root', 'pelvis',
        'spine_01', 'spine_02', 'spine_03', 'spine_04', 'spine_05',
        'neck_01', 'neck_02', 'head',
    }

    all_islands = set(range(num_islands))
    bone_islands = {}

    for bone in arm_data.bones:
        if bone.name in trunk_bones:
            bone_islands[bone.name] = all_islands
            continue

        mid = (Vector(bone.head_local) + Vector(bone.tail_local)) * 0.5
        _co, nearest_vi, _dist = kd.find(mid)
        bone_islands[bone.name] = {vert_island[nearest_vi]}

    return bone_islands


def get_limb_group(bone_name):
    """Classify a bone into a limb group for cross-limb distance clamping."""
    n = bone_name.lower()
    if 'thigh' in n or 'calf' in n or 'foot' in n or 'ball' in n or 'toe' in n:
        if '_l' in n:
            return 'leg_l'
        if '_r' in n:
            return 'leg_r'
    if 'upperarm' in n or 'lowerarm' in n or 'hand' in n or 'finger' in n or 'thumb' in n:
        if '_l' in n:
            return 'arm_l'
        if '_r' in n:
            return 'arm_r'
    if 'clavicle' in n:
        if '_l' in n:
            return 'clav_l'
        if '_r' in n:
            return 'clav_r'
    return 'trunk'



def clamp_weights_combined(mesh_obj, armature_obj):
    """Two-phase weight clamping:

    Phase 1 — Island clamping: Strip weights from bones on different
    disconnected mesh islands (definitive gap detection via edge connectivity).

    Phase 2 — K-nearest bone clamping: For each vertex, find the K nearest
    non-trunk bones by Euclidean distance. Strip weights from all other
    non-trunk bones. This prevents distant bones from influencing vertices
    that clearly belong to a closer bone's region. K=4 by default.

    Returns (total_clamped, debug_dict).
    """
    # ---- Phase 1: Island clamping ----
    vert_island, num_islands = find_mesh_islands(mesh_obj)
    log(f"Mesh islands: {num_islands} disconnected parts detected")

    island_sizes = {}
    for iid in vert_island.values():
        island_sizes[iid] = island_sizes.get(iid, 0) + 1
    top_islands = sorted(island_sizes.items(), key=lambda x: -x[1])[:20]

    debug = {
        'num_islands': num_islands,
        'island_sizes': {str(i): s for i, s in top_islands},
        'bone_assignments': {},
        'clamped_per_bone': {},
        'phase1_clamped': 0,
        'phase2_clamped': 0,
    }

    island_clamped = 0
    if num_islands > 1:
        log(f"  Island sizes: {', '.join(f'#{i}:{s}v' for i, s in top_islands)}")

        bone_islands = assign_bones_to_islands(
            mesh_obj, armature_obj, vert_island, num_islands)

        for bname, islands in sorted(bone_islands.items()):
            debug['bone_assignments'][bname] = sorted(islands)
            if len(islands) < num_islands:
                log(f"  Bone '{bname}' → island(s) {islands}")

        mesh = mesh_obj.data
        V = len(mesh.vertices)

        for vg in mesh_obj.vertex_groups:
            bone_name = vg.name
            if bone_name not in bone_islands:
                continue
            allowed_islands = bone_islands[bone_name]
            for vi in range(V):
                try:
                    w = vg.weight(vi)
                except RuntimeError:
                    continue
                if w < 0.0001:
                    continue
                if vert_island[vi] not in allowed_islands:
                    vg.remove([vi])
                    island_clamped += 1

        if island_clamped > 0:
            bpy.context.view_layer.objects.active = mesh_obj
            mesh_obj.select_set(True)
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
            try:
                bpy.ops.object.vertex_group_normalize_all(
                    group_select_mode='ALL', lock_active=False)
            except Exception:
                pass
            bpy.ops.object.mode_set(mode='OBJECT')
            mesh_obj.select_set(False)
            log(f"Phase 1 (island): stripped {island_clamped} cross-island weights")
    else:
        log("Single island — skipping phase 1")

    debug['phase1_clamped'] = island_clamped

    # ---- Phase 2: Cross-limb distance clamping ----
    # For each vertex, find which bone is closest. Strip weights from bones
    # on a DIFFERENT limb that are more than RATIO times farther away.
    # This catches satchel-from-forearm bleed (the forearm bone is far from
    # satchel vertices, while thigh/pelvis bones are close).
    mesh = mesh_obj.data
    verts = mesh.vertices
    V = len(verts)
    arm_data = armature_obj.data

    trunk_bones = {
        'root', 'pelvis',
        'spine_01', 'spine_02', 'spine_03', 'spine_04', 'spine_05',
        'neck_01', 'neck_02', 'head',
    }

    bone_info = {}
    for bone in arm_data.bones:
        bone_info[bone.name] = {
            'head': Vector(bone.head_local),
            'tail': Vector(bone.tail_local),
            'length': bone.length,
            'limb': get_limb_group(bone.name),
        }

    MAX_BONE_INFLUENCES = 4  # keep only the K nearest non-trunk bones per vertex
    phase2_clamped = 0
    phase2_per_bone = {}
    phase2_debug_samples = []

    # Pre-compute Euclidean distance from every vertex to every non-trunk bone
    non_trunk_list = [(bname, bone_info[bname]) for bname in bone_info
                      if bname not in trunk_bones]

    index_to_vg = {vg.index: vg for vg in mesh_obj.vertex_groups
                   if vg.name in bone_info}
    index_to_name = {vg.index: vg.name for vg in mesh_obj.vertex_groups
                     if vg.name in bone_info}
    removals = {}  # group index -> [vertex indices] (batched remove is much faster)

    for vi in range(V):
        vert = verts[vi]
        vert_pos = vert.co

        # Only consider non-trunk bones that actually have weight here.
        weighted_non_trunk = []
        for g in vert.groups:
            bname = index_to_name.get(g.group)
            if not bname or bname in trunk_bones or g.weight < 0.0001:
                continue
            weighted_non_trunk.append((bname, g.group, g.weight))

        if not weighted_non_trunk:
            continue

        bone_dists = [
            (point_to_segment_dist(vert_pos, bi['head'], bi['tail']), bname)
            for bname, bi in non_trunk_list
        ]
        bone_dists.sort()
        allowed_bones = {b for _, b in bone_dists[:MAX_BONE_INFLUENCES]}

        for bname, grp_idx, w in weighted_non_trunk:
            if bname in allowed_bones:
                continue
            removals.setdefault(grp_idx, []).append(vi)
            phase2_clamped += 1
            phase2_per_bone[bname] = phase2_per_bone.get(bname, 0) + 1

            if len(phase2_debug_samples) < 20:
                phase2_debug_samples.append({
                    'vi': vi,
                    'stripped_bone': bname,
                    'stripped_dist': round(point_to_segment_dist(
                        vert_pos, bone_info[bname]['head'], bone_info[bname]['tail']), 2),
                    'nearest_bone': bone_dists[0][1],
                    'nearest_dist': round(bone_dists[0][0], 2),
                    'allowed': [b for _, b in bone_dists[:MAX_BONE_INFLUENCES]],
                    'weight': round(w, 4),
                    'vert': [round(c, 4) for c in vert_pos],
                })

    for grp_idx, vis in removals.items():
        index_to_vg[grp_idx].remove(vis)

    debug['phase2_clamped'] = phase2_clamped
    debug['phase2_per_bone'] = phase2_per_bone
    debug['phase2_debug_samples'] = phase2_debug_samples

    if phase2_clamped > 0:
        bpy.context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)
        bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
        try:
            bpy.ops.object.vertex_group_normalize_all(
                group_select_mode='ALL', lock_active=False)
        except Exception:
            pass
        bpy.ops.object.mode_set(mode='OBJECT')
        mesh_obj.select_set(False)
        top = sorted(phase2_per_bone.items(), key=lambda x: -x[1])[:10]
        log(f"Phase 2 (cross-limb distance): stripped {phase2_clamped} weights")
        log(f"  Top: {', '.join(f'{n}:{c}' for n, c in top)}")
    else:
        log("Phase 2 (cross-limb distance): no cross-limb bleed found")

    if phase2_debug_samples:
        log(f"Phase 2 strip samples ({len(phase2_debug_samples)}):")
        for s in phase2_debug_samples[:10]:
            log(f"  vi={s['vi']} stripped={s['stripped_bone']} "
                f"(dist={s['stripped_dist']}) nearest={s['nearest_bone']} "
                f"(dist={s['nearest_dist']}) allowed={s['allowed']}")

    total = island_clamped + phase2_clamped
    debug['clamped_per_bone'] = phase2_per_bone
    return total, debug


def smooth_vertex_groups(mesh_obj, passes=2, factor=0.3):
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    vg_count = len(mesh_obj.vertex_groups)
    if vg_count == 0:
        return
    bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
    for vg in mesh_obj.vertex_groups:
        mesh_obj.vertex_groups.active = vg
        try:
            bpy.ops.object.vertex_group_smooth(
                group_select_mode='ACTIVE',
                factor=factor,
                repeat=passes,
                expand=0.0,
            )
        except Exception as e:
            log(f"Smoothing failed for {vg.name}: {e}")
    bpy.ops.object.mode_set(mode='OBJECT')
    mesh_obj.select_set(False)
    log(f"Smoothed {vg_count} vertex groups ({passes} passes, factor {factor})")


def extract_weights(mesh_obj, V):
    """Read vertex-group weights via mesh.vertices[].groups (O(V·k)), not
    vertex_group.weight(vi) per bone (O(V·B)) — the latter is catastrophically
    slow on 500k+ vert meshes (minutes of Python↔Blender API calls)."""
    mesh = mesh_obj.data
    index_to_name = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
    log(f"Vertex groups created: {len(index_to_name)}")

    weights = {}
    zero_weight_count = 0
    for vi in range(V):
        vert = mesh.vertices[vi]
        total = 0.0
        for g in vert.groups:
            if g.weight <= 0.0001:
                continue
            name = index_to_name.get(g.group)
            if not name:
                continue
            total += g.weight
            bucket = weights.get(name)
            if bucket is None:
                bucket = {}
                weights[name] = bucket
            bucket[str(vi)] = round(g.weight, 6)
        if total < 0.0001:
            zero_weight_count += 1

    return weights, zero_weight_count


def _prepare_scene(data):
    """Clear the scene, build the mesh from raw verts/tris, and repair topology.

    Shared by both the harmonic and bone-heat paths so they operate on the
    SAME repaired mesh (and therefore the same vertex indexing the client
    expects). Returns (mesh_obj, input_V, input_T, dupe_faces, repaired_V).
    """
    vertices = data['vertices']
    triangles = data['triangles']

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)

    mesh_data = bpy.data.meshes.new('WeightMesh')
    bm = bmesh.new()
    bm_verts = []
    for v in vertices:
        bm_verts.append(bm.verts.new(Vector(v)))
    bm.verts.ensure_lookup_table()
    dupe_faces = 0
    for tri in triangles:
        try:
            bm.faces.new([bm_verts[tri[0]], bm_verts[tri[1]], bm_verts[tri[2]]])
        except ValueError:
            dupe_faces += 1
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()

    mesh_obj = bpy.data.objects.new('WeightMesh', mesh_data)
    bpy.context.collection.objects.link(mesh_obj)
    log(f"Mesh created: {len(mesh_data.vertices)} verts, {len(mesh_data.polygons)} faces (skipped {dupe_faces} dupes)")

    repaired_V = repair_mesh(mesh_obj)
    return mesh_obj, len(vertices), len(triangles), dupe_faces, repaired_V


# ===========================================================================
# Harmonic + bone-visibility solver (the principled core)
# ===========================================================================
# Instead of Blender bone-heat + a tower of anatomy-specific cleanup, we solve
# for smooth, bounded, partition-of-unity weights directly:
#
#   1. Bone visibility: a vertex only seeds from bones it can "see" (the
#      straight line vertex->bone is not blocked by the mesh surface). An
#      inner-thigh vertex cannot see the OTHER leg's bone, so cross-leg bleed
#      is structurally impossible -- no midline/anatomy hacks needed.
#   2. Seeds: among the nearest visible bones, seed targets are 1/d^2 weighted
#      (steep falloff so the own-limb bone dominates) and normalized to 1.
#   3. Harmonic solve: (L + h*I) W = h*P, where L is the cotangent Laplacian.
#      Because per-vertex P sums to 1 and (L+hI)*1 = h*1, the solution is a
#      partition of unity by construction -- smooth, bounded, no speckle.
#
# numpy/scipy are imported lazily so this module still loads (and falls back to
# bone-heat) on an image that hasn't been rebuilt with scipy yet.

# Tunable internals (NOT user-facing; derived/auto-scaled where possible).
_HARM_MAX_CANDIDATES = 8     # nearest bones considered per vertex
_HARM_REACH = 3.0            # only contest bones within REACH x nearest distance
_HARM_PIN_FACTOR = 1.0       # pin stiffness vs Laplacian scale (higher = sharper)
_HARM_MAX_INFLUENCES = 4
_HARM_WMIN = 1e-4
_HARM_CG_MAXITER = 400
# Visibility is tested against points sampled along the bone SHAFT (not the
# closest point, which can be an endpoint buried in shared joint flesh -- e.g.
# the upper-arm head inside the shoulder, reachable from the side torso THROUGH
# solid without crossing the armpit gap). Targeting the shaft forces the ray to
# cross any real air gap. A bone is "visible" iff at least one shaft sample is
# reachable unobstructed.
_HARM_SHAFT_SAMPLES = (0.15, 0.35, 0.5, 0.65, 0.85)


def _read_mesh_numpy(mesh_obj):
    """Pull repaired-mesh geometry into numpy: (co Vx3, normals Vx3, tris Fx3)."""
    import numpy as np
    mesh = mesh_obj.data
    mesh.calc_loop_triangles()
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get('co', co)
    co = co.reshape(-1, 3)
    nrm = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get('normal', nrm)
    nrm = nrm.reshape(-1, 3)
    m = len(mesh.loop_triangles)
    tri = np.empty(m * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get('vertices', tri)
    tri = tri.reshape(-1, 3)
    return co, nrm, tri


def _cotangent_laplacian(co, tri):
    """Sparse cotangent Laplacian L = D - W (V x V), clamped for sliver tris."""
    import numpy as np
    import scipy.sparse as sp
    V = co.shape[0]
    i1, i2, i3 = tri[:, 0], tri[:, 1], tri[:, 2]
    v1, v2, v3 = co[i1], co[i2], co[i3]
    cross = np.cross(v2 - v1, v3 - v1)
    area2 = np.linalg.norm(cross, axis=1)
    area2 = np.maximum(area2, 1e-12)
    # cot of angle at each vertex = dot(edge1, edge2) / (2*area)
    cot1 = np.einsum('ij,ij->i', v2 - v1, v3 - v1) / area2
    cot2 = np.einsum('ij,ij->i', v1 - v2, v3 - v2) / area2
    cot3 = np.einsum('ij,ij->i', v1 - v3, v2 - v3) / area2
    np.clip(cot1, -1e3, 1e3, out=cot1)
    np.clip(cot2, -1e3, 1e3, out=cot2)
    np.clip(cot3, -1e3, 1e3, out=cot3)
    # angle at i1 (cot1) is opposite edge (i2,i3), etc.
    rows = np.concatenate([i2, i3, i3, i1, i1, i2])
    cols = np.concatenate([i3, i2, i1, i3, i2, i1])
    vals = np.concatenate([cot1, cot1, cot2, cot2, cot3, cot3]) * 0.5
    W = sp.coo_matrix((vals, (rows, cols)), shape=(V, V)).tocsr()
    diag = np.asarray(W.sum(axis=1)).ravel()
    L = sp.diags(diag) - W
    return L


def _bone_arrays(bones):
    import numpy as np
    names = [b['name'] for b in bones]
    heads = np.array([b['head'] for b in bones], dtype=np.float64)
    tails = np.array([b['tail'] for b in bones], dtype=np.float64)
    return names, heads, tails


def _all_bone_distances(co, heads, tails):
    """(V,B) point-to-segment distances, vectorized over vertices per bone."""
    import numpy as np
    V = co.shape[0]
    B = heads.shape[0]
    D = np.empty((V, B), dtype=np.float64)
    for b in range(B):
        a = heads[b]
        ab = tails[b] - a
        ab2 = float(ab.dot(ab))
        if ab2 < 1e-12:
            D[:, b] = np.linalg.norm(co - a, axis=1)
        else:
            t = ((co - a) @ ab) / ab2
            np.clip(t, 0.0, 1.0, out=t)
            proj = a[None, :] + t[:, None] * ab[None, :]
            D[:, b] = np.linalg.norm(co - proj, axis=1)
    return D


def compute_weights_harmonic(data):
    t0 = time.time()
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from mathutils import Vector as _V
    from mathutils.bvhtree import BVHTree

    mesh_obj, V_in, T_in, dupe_faces, repaired_V = _prepare_scene(data)
    bones = data['bones']
    B = len(bones)
    names, heads, tails = _bone_arrays(bones)

    co, nrm, tri = _read_mesh_numpy(mesh_obj)
    V = co.shape[0]
    log(f"Harmonic: {V} verts, {tri.shape[0]} tris, {B} bones")

    bbox = co.max(axis=0) - co.min(axis=0)
    diag_len = float(np.linalg.norm(bbox)) or 1.0
    eps = diag_len * 1e-4

    # --- 1. nearest-K bone candidates per vertex ---
    t_d = time.time()
    D = _all_bone_distances(co, heads, tails)
    K = min(_HARM_MAX_CANDIDATES, B)
    cand = np.argpartition(D, K - 1, axis=1)[:, :K]  # (V,K) unsorted top-K
    cand_d = np.take_along_axis(D, cand, axis=1)
    order = np.argsort(cand_d, axis=1)
    cand = np.take_along_axis(cand, order, axis=1)
    cand_d = np.take_along_axis(cand_d, order, axis=1)
    nearest_d = cand_d[:, 0]
    log(f"Harmonic: candidate distances {round(time.time()-t_d,2)}s")

    # --- 2. bone visibility (only for contested verts, only contested cands) ---
    t_v = time.time()
    bvh = BVHTree.FromObject(mesh_obj, bpy.context.evaluated_depsgraph_get())
    seg_pt = np.empty(3, dtype=np.float64)

    # rows/cols/data for the seed matrix P (V x B), sum-to-1 per vertex.
    rows = []
    cols = []
    vals = []

    # contested = 2nd-nearest bone is within REACH x nearest -> ambiguous region.
    contested = cand_d[:, 1] <= (nearest_d * _HARM_REACH) if K > 1 else np.zeros(V, bool)
    n_contested = int(contested.sum())
    log(f"Harmonic: {n_contested}/{V} contested verts get visibility tests")

    ray_counter = [0]

    def _bone_visible(p, a, b):
        """True iff some point along the bone SHAFT is reachable from p without
        the mesh surface in between (i.e. without crossing an air gap). Sampling
        the shaft (not the closest point) is what makes this gap-aware: a tip
        buried in shared joint flesh is never used as the target."""
        ab = b - a
        ab2 = float(ab.dot(ab))
        for tt in _HARM_SHAFT_SAMPLES:
            q = a if ab2 < 1e-12 else (a + ab * tt)
            d = q - p
            dist = float(np.linalg.norm(d))
            if dist < eps:
                return True
            dirv = d / dist
            origin = _V((p[0] + dirv[0] * eps, p[1] + dirv[1] * eps, p[2] + dirv[2] * eps))
            direction = _V((dirv[0], dirv[1], dirv[2]))
            ray_counter[0] += 1
            hit = bvh.ray_cast(origin, direction, dist - 2 * eps)
            if hit[0] is None:
                return True  # a shaft point is in clear line of sight
            if ab2 < 1e-12:
                break
        return False

    for vi in range(V):
        c0 = int(cand[vi, 0])
        if not contested[vi]:
            rows.append(vi); cols.append(c0); vals.append(1.0)
            continue
        p = co[vi]
        # nearest bone is always allowed (auto-visible)
        sel_bones = [c0]
        sel_d = [cand_d[vi, 0]]
        for kk in range(1, K):
            db = cand_d[vi, kk]
            if db > nearest_d[vi] * _HARM_REACH:
                break
            bidx = int(cand[vi, kk])
            if _bone_visible(p, heads[bidx], tails[bidx]):
                sel_bones.append(bidx); sel_d.append(db)
            # else blocked across a gap -> not seeded
        # seed weights ~ 1/d^2 (steep), normalized to sum 1
        inv = np.array([1.0 / (dd * dd + 1e-12) for dd in sel_d])
        inv /= inv.sum()
        for bidx, w in zip(sel_bones, inv):
            rows.append(vi); cols.append(bidx); vals.append(float(w))
    ray_tests = ray_counter[0]
    log(f"Harmonic: visibility {round(time.time()-t_v,2)}s ({ray_tests} ray tests)")

    P = sp.coo_matrix((vals, (rows, cols)), shape=(V, B)).tocsc()

    # --- 3. harmonic solve (L + h I) W = h P, factor once, solve per bone ---
    t_s = time.time()
    L = _cotangent_laplacian(co, tri)
    Ldiag = L.diagonal()
    h = float(np.median(Ldiag[Ldiag > 0])) * _HARM_PIN_FACTOR
    if not np.isfinite(h) or h <= 0:
        h = 1.0
    A = (L + h * sp.identity(V, format='csr')).tocsc()

    solve = None
    try:
        solve = spla.factorized(A)  # SuperLU; fast for many RHS
        log("Harmonic: using SuperLU factorization")
    except Exception as e:
        log(f"Harmonic: splu failed ({e}); using CG")

    Mdiag = None
    if solve is None:
        Mdiag = spla.LinearOperator((V, V), matvec=lambda x: x / A.diagonal())

    W = np.zeros((V, B), dtype=np.float64)
    for b in range(B):
        rhs = h * np.asarray(P[:, b].todense()).ravel()
        if rhs.max() <= 0:
            continue
        if solve is not None:
            W[:, b] = solve(rhs)
        else:
            x, _info = spla.cg(A, rhs, M=Mdiag, maxiter=_HARM_CG_MAXITER)
            W[:, b] = x
    log(f"Harmonic: solve {round(time.time()-t_s,2)}s")

    # --- 4. clamp negatives ---
    np.clip(W, 0.0, None, out=W)

    # --- 4b. VISIBILITY MASK (the anti-gap-bleed step) ---------------------
    # The harmonic solve diffuses weight along mesh connectivity, so a bone's
    # influence leaks across short surface folds and even to far parts (arm ->
    # side torso, arm -> hair). Connectivity diffusion can't see the air gap. So
    # we enforce the physical rule directly: a bone may only keep weight on a
    # vertex whose SHAFT it can see -- a straight segment vertex->shaft-point not
    # blocked by the mesh surface. A vertex's own-limb bone is reached through
    # solid flesh (no surface crossing) -> visible; a bone across a gap is
    # occluded by the surface the ray exits/enters -> masked. Sampling the SHAFT
    # (not the closest point) is essential: the upper-arm head sits inside the
    # shoulder, reachable from the side torso through solid, so closest-point
    # targeting misses that leak. The nearest bone is always kept so no vertex
    # is ever fully zeroed by masking.
    t_m = time.time()
    ray_counter[0] = 0
    mask_pair_tests = 0
    masked_pairs = 0
    nzcount = (W > _HARM_WMIN).sum(axis=1)
    multi = np.where(nzcount > 1)[0]
    for vi in multi.tolist():
        nb = int(cand[vi, 0])  # nearest bone: always allowed
        p = co[vi]
        for b in np.where(W[vi] > _HARM_WMIN)[0].tolist():
            b = int(b)
            if b == nb:
                continue
            mask_pair_tests += 1
            if not _bone_visible(p, heads[b], tails[b]):  # across a gap
                W[vi, b] = 0.0
                masked_pairs += 1
    mask_tests = ray_counter[0]
    log(f"Harmonic: visibility mask {round(time.time()-t_m,2)}s "
        f"({mask_pair_tests} pairs, {mask_tests} rays, {masked_pairs} bleed pairs killed)")

    # --- 4c. max-4 influences, normalize partition of unity ---
    if B > _HARM_MAX_INFLUENCES:
        keep = np.argpartition(W, B - _HARM_MAX_INFLUENCES, axis=1)[:, -_HARM_MAX_INFLUENCES:]
        mask = np.zeros_like(W, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        W[~mask] = 0.0
    rowsum = W.sum(axis=1)
    zero_weight_count = int((rowsum < 1e-8).sum())
    # rescue any all-zero vertex to its nearest bone
    if zero_weight_count:
        zidx = np.where(rowsum < 1e-8)[0]
        W[zidx, cand[zidx, 0]] = 1.0
        rowsum[zidx] = 1.0
    W /= rowsum[:, None]

    # Remap repaired-mesh weights to INPUT vertex order (see voxel path): the
    # client maps weights by the index it sent, but repair_mesh can reorder /
    # merge vertices. Resample at each input vertex via nearest repaired vertex.
    from scipy.spatial import cKDTree as _cKDTree
    V_input = np.asarray(data['vertices'], dtype=np.float64)
    nn = _cKDTree(co).query(V_input, k=1)[1]
    W = W[nn]
    log(f"Harmonic: remapped repaired->input order ({co.shape[0]} -> {V_input.shape[0]} verts)")

    # --- 5. build sparse weights dict {bone: {vi: w}} ---
    t_e = time.time()
    weights = {}
    nz_v, nz_b = np.where(W > _HARM_WMIN)
    for vi, b in zip(nz_v.tolist(), nz_b.tolist()):
        bucket = weights.get(names[b])
        if bucket is None:
            bucket = {}
            weights[names[b]] = bucket
        bucket[str(vi)] = round(float(W[vi, b]), 6)
    log(f"Harmonic: extract {round(time.time()-t_e,2)}s, {len(weights)} bones")

    elapsed = time.time() - t0
    bone_weight_stats = {}
    for bname, bw in weights.items():
        ws = list(bw.values())
        bone_weight_stats[bname] = {
            'verts': len(ws),
            'min_w': round(min(ws), 4) if ws else 0,
            'max_w': round(max(ws), 4) if ws else 0,
            'avg_w': round(sum(ws) / len(ws), 4) if ws else 0,
        }
    debug_vert_positions = [[round(float(co[vi, 0]), 5), round(float(co[vi, 1]), 5),
                             round(float(co[vi, 2]), 5)] for vi in range(min(repaired_V, 5000))]
    log(f"Harmonic done: {len(weights)} bones, {zero_weight_count} zero verts, {elapsed:.1f}s")

    return {
        'weights': weights,
        'bone_count': len(weights),
        'weight_method': 'HARMONIC_VISIBILITY',
        'diagnostics': {
            'input_verts': V_in,
            'input_tris': T_in,
            'repaired_verts': repaired_V,
            'duplicate_faces': dupe_faces,
            'zero_weight_verts': zero_weight_count,
            'bones_with_weights': len(weights),
            'bones_requested': B,
            'solver': {
                'method': 'harmonic_visibility',
                'pin_h': round(h, 6),
                'contested_verts': n_contested,
                'ray_tests': ray_tests,
                'mask_tests': mask_tests,
                'bleed_pairs_killed': masked_pairs,
                'candidates': K,
            },
            'bone_weight_stats': bone_weight_stats,
            'debug_vert_positions': debug_vert_positions,
            'timing': {
                'total_s': round(elapsed, 2),
            },
        },
        'elapsed': round(elapsed, 2),
    }


# ===========================================================================
# Volumetric voxel binder (the robust core)
# ===========================================================================
# Surface methods (bone-heat, harmonic) leak across gaps because the body is a
# single connected manifold -- arm influence always finds a SURFACE path to the
# side torso/hair. The volumetric approach fills the mesh INTERIOR with voxels
# and diffuses each bone's influence THROUGH the solid, so empty space (the
# armpit, the gap between hair and arm) simply has no voxels and cannot carry
# weight. Mirrors Maya's Geodesic Voxel Binding (Dionne & de Lasa, SIGGRAPH
# 2013) and Blender's Voxel Heat Diffuse addon. scipy/numpy are imported lazily
# so the module still loads (and falls back) on an image without scipy.

_VOX_TARGET_AXIS = 150        # cells along the longest bbox axis (quality)
_VOX_MAX_OCCUPIED = 1_000_000  # cap occupied voxels (auto-downscale if exceeded);
                               # per-bone dijkstra cost scales ~M*bones, so this
                               # keeps a ~30-bone humanoid under ~8 min.
_VOX_MIN_AXIS = 40
_VOX_MAX_INFLUENCES = 4
_VOX_WMIN = 1e-4
# Substrings (case-insensitive) of bones that need SURFACE precision (the voxel
# solid fuses adjacent fingers/lips). Body/clothing keeps the volumetric result.
_VOX_FINE_BONE_PATTERNS = (
    'hand', 'finger', 'thumb', 'index', 'middle', 'ring', 'pinky', 'pinkie',
    'toe', 'eye', 'jaw', 'tongue', 'teeth', 'lip',
)


def _ray_hits(bvh, o, d, length, eps):
    """All surface crossing distances (from o, along unit d) up to `length`."""
    hits = []
    base = 0.0
    cur = o.copy()
    while base < length:
        r = bvh.ray_cast(cur, d, length - base)
        if r[0] is None:
            break
        hit_at = base + r[3]
        hits.append(hit_at)
        base = hit_at + eps
        cur = o + d * base
    return hits


def _voxelize_solid(co, tri, res):
    """Solidify the mesh into an occupancy grid via multi-axis ray parity.

    A voxel is SOLID if it is inside the surface along >=2 of the three axis
    sweeps (majority vote -> robust to a hole on any single axis) OR it directly
    contains the surface (boundary). Returns (solid bool[nx,ny,nz], origin xyz,
    cell size). Handles non-watertight / multi-component / self-intersecting
    meshes -- exactly the messy output of image-to-3D models.
    """
    import numpy as np
    from mathutils import Vector as _V
    from mathutils.bvhtree import BVHTree

    verts = [_V((float(p[0]), float(p[1]), float(p[2]))) for p in co]
    polys = [(int(t[0]), int(t[1]), int(t[2])) for t in tri]
    bvh = BVHTree.FromPolygons(verts, polys, all_triangles=True, epsilon=0.0)

    mn = co.min(axis=0)
    mx = co.max(axis=0)
    ext = mx - mn
    longest = float(ext.max()) or 1.0
    cell = longest / float(res)
    pad = cell * 2.0
    origin = mn - pad
    dims = np.floor((mx + pad - origin) / cell).astype(int) + 2
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    eps = cell * 1e-4

    inside_count = np.zeros((nx, ny, nz), dtype=np.uint8)
    boundary = np.zeros((nx, ny, nz), dtype=bool)

    axes = [
        # (axis, n_axis, dir vector, the two perpendicular extents/counts)
        (2, nz, _V((0.0, 0.0, 1.0)), nx, ny),
        (1, ny, _V((0.0, 1.0, 0.0)), nx, nz),
        (0, nx, _V((1.0, 0.0, 0.0)), ny, nz),
    ]
    for axis, n_axis, dvec, na, nb in axes:
        centers = (np.arange(n_axis) + 0.5) * cell  # distance from grid corner
        length = n_axis * cell
        for ia in range(na):
            for ib in range(nb):
                if axis == 2:
                    ox = origin[0] + (ia + 0.5) * cell
                    oy = origin[1] + (ib + 0.5) * cell
                    o = _V((ox, oy, float(origin[2])))
                elif axis == 1:
                    ox = origin[0] + (ia + 0.5) * cell
                    oz = origin[2] + (ib + 0.5) * cell
                    o = _V((ox, float(origin[1]), oz))
                else:
                    oy = origin[1] + (ia + 0.5) * cell
                    oz = origin[2] + (ib + 0.5) * cell
                    o = _V((float(origin[0]), oy, oz))
                hits = _ray_hits(bvh, o, dvec, length, eps)
                if not hits:
                    continue
                ha = np.asarray(hits)
                kb = (ha / cell).astype(int)
                kb = kb[(kb >= 0) & (kb < n_axis)]
                inside = (np.searchsorted(ha, centers, side='right') & 1).astype(bool)
                if axis == 2:
                    boundary[ia, ib, kb] = True
                    inside_count[ia, ib, inside] += 1
                elif axis == 1:
                    boundary[ia, kb, ib] = True
                    inside_count[ia, inside, ib] += 1
                else:
                    boundary[kb, ia, ib] = True
                    inside_count[inside, ia, ib] += 1

    solid = (inside_count >= 2) | boundary
    return solid, np.asarray(origin, dtype=np.float64), float(cell)


def _build_voxel_graph(solid, cell):
    """Build the occupied-voxel graph (26-connectivity). Returns:
      vid   : int grid mapping (i,j,k)->voxel id (-1 = empty)
      M     : occupied voxel count
      G     : weighted graph (Euclidean edge lengths) for geodesic distance
      A     : unit adjacency (for fast Laplacian smoothing)
      occ   : (M,3) voxel index coords
    Edges exist ONLY between solid neighbours, so distance can never travel
    across empty space (the armpit / hair gap) -- the whole point."""
    import numpy as np
    import scipy.sparse as sp
    nx, ny, nz = solid.shape
    vid = -np.ones(solid.shape, dtype=np.int64)
    occ = np.argwhere(solid)
    M = occ.shape[0]
    if M == 0:
        return vid, 0, None, None, occ
    vid[occ[:, 0], occ[:, 1], occ[:, 2]] = np.arange(M)
    # 26-connectivity, half the offsets (the graph is made symmetric below)
    offs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (0, 1):
                if dz == 0 and (dx < 0 or (dx == 0 and dy <= 0)):
                    continue
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offs.append((dx, dy, dz))
    rs, cs, ws = [], [], []
    for dx, dy, dz in offs:
        xa0, xa1 = max(0, -dx), nx - max(0, dx)
        ya0, ya1 = max(0, -dy), ny - max(0, dy)
        za0, za1 = max(0, -dz), nz - max(0, dz)
        a = vid[xa0:xa1, ya0:ya1, za0:za1]
        b = vid[xa0 + dx:xa1 + dx, ya0 + dy:ya1 + dy, za0 + dz:za1 + dz]
        m = (a >= 0) & (b >= 0)
        cnt = int(m.sum())
        if cnt == 0:
            continue
        w = cell * float((dx * dx + dy * dy + dz * dz) ** 0.5)
        rs.append(a[m]); cs.append(b[m]); ws.append(np.full(cnt, w))
    r = np.concatenate(rs); c = np.concatenate(cs); wv = np.concatenate(ws)
    rr = np.concatenate([r, c]); cc = np.concatenate([c, r]); ww = np.concatenate([wv, wv])
    G = sp.csr_matrix((ww, (rr, cc)), shape=(M, M))
    A = sp.csr_matrix((np.ones(rr.shape[0], dtype=np.float32), (rr, cc)), shape=(M, M))
    return vid, M, G, A, occ


def _voxel_bone_sources(heads, tails, origin, cell, vid, solid, centers):
    """Bone segments -> per-bone lists of source voxel ids. A bone whose segment
    lies outside the solid snaps to the nearest occupied voxel."""
    import numpy as np
    from scipy.spatial import cKDTree
    nx, ny, nz = solid.shape
    B = heads.shape[0]
    sources = [set() for _ in range(B)]
    tree = None
    for b in range(B):
        a = heads[b]
        seg = tails[b] - a
        seglen = float(np.linalg.norm(seg))
        n = max(2, int(seglen / (cell * 0.5)) + 1)
        ts = np.linspace(0.0, 1.0, n)
        pts = a[None, :] + ts[:, None] * seg[None, :]
        gi = np.floor((pts - origin[None, :]) / cell).astype(int)
        for k in range(n):
            i, j, kk = int(gi[k, 0]), int(gi[k, 1]), int(gi[k, 2])
            if 0 <= i < nx and 0 <= j < ny and 0 <= kk < nz and vid[i, j, kk] >= 0:
                sources[b].add(int(vid[i, j, kk]))
        if not sources[b]:
            if tree is None:
                tree = cKDTree(centers)
            mid = a + 0.5 * seg
            _, vidx = tree.query(mid)
            sources[b].add(int(vidx))
    return [np.array(sorted(s), dtype=np.int64) for s in sources]


# Geodesic falloff tunables (parameter-light, model-agnostic).
_VOX_WIN_LO = 1.5      # ratio d_b/d_min below which a bone contributes fully
_VOX_WIN_HI = 2.5      # ratio above which it's fully suppressed (gap kill)
_VOX_SMOOTH_ITERS = 10  # Laplacian smoothing passes on the voxel graph
_VOX_DENSE_GATE = 2.0  # densify: seed voxel to nearest bone iff geodesic <= gate*euclid+slack


def _euclid_nearest_seg(pts, heads, tails):
    """For each point, return (nearest_bone_index, distance) by point-to-segment
    Euclidean distance to every bone."""
    import numpy as np
    M = pts.shape[0]
    B = heads.shape[0]
    ab = tails - heads
    ab2 = np.maximum((ab * ab).sum(1), 1e-12)
    best_b = np.zeros(M, np.int64)
    best_d = np.full(M, np.inf)
    for b in range(B):
        ap = pts - heads[b]
        t = np.clip((ap @ ab[b]) / ab2[b], 0.0, 1.0)
        proj = heads[b] + t[:, None] * ab[b]
        dd = np.linalg.norm(pts - proj, axis=1)
        upd = dd < best_d
        best_d[upd] = dd[upd]
        best_b[upd] = b
    return best_b, best_d


def _densify_sources(G, bone_sources, centers, heads, tails, M, B, cell,
                     gate=_VOX_DENSE_GATE):
    """Geodesic-gated dense Voronoi seeding — the core robustness fix.

    The bone segments come from getBoneSegments, which gives LEAF bones (hand,
    head, breast, toe) only a tiny 0.15x stub. Seeding sources only where that
    stub crosses voxels starves the leaf's region, so the long parent bone
    (lowerarm/spine/neck) out-reaches it and "wins" the hand/head/breast. Pure
    Euclidean Voronoi seeding would fix that but reintroduces gap-jumps (an arm
    grabbing a nearby-in-space-but-far-through-the-body ponytail).

    This combines both: run an initial geodesic from the sparse stubs, then add
    each voxel to its Euclidean-nearest bone ONLY IF the geodesic distance to
    that bone is not a long detour (<= gate*euclid + slack). Result: every bone
    densely owns its true volumetric region, while voxels separated by an air
    gap (geodesic >> euclid) are NOT seeded across the gap."""
    import numpy as np
    import scipy.sparse.csgraph as csg
    D0 = np.full((M, B), np.inf, dtype=np.float64)
    for b in range(B):
        src = bone_sources[b]
        if src.size:
            D0[:, b] = csg.dijkstra(G, directed=False, indices=src, min_only=True)
    nb, ed = _euclid_nearest_seg(centers, heads, tails)
    slack = 3.0 * cell
    d0_at = D0[np.arange(M), nb]
    keep = np.isfinite(d0_at) & (d0_at <= gate * ed + slack)
    out = [set(s.tolist()) for s in bone_sources]
    sel = np.where(keep)[0]
    for v in sel.tolist():
        out[int(nb[v])].add(v)
    return [np.array(sorted(s), dtype=np.int64) for s in out]


def _solve_voxel_geodesic(G, A, M, bone_sources, B, cell):
    """Per-bone influence THROUGH the volume by multi-source geodesic distance.

    scipy.sparse.csgraph.dijkstra is C-level and scales to millions of voxels
    (unlike a dense splu factorization, which stalls past ~200k 3D nodes). Each
    bone's distance field is converted to weight via a 1/d^2 falloff, but ONLY
    over bones whose geodesic reach is comparable to the voxel's nearest bone (a
    smoothstep window on d_b/d_min). A far-but-reachable bone -- e.g. the arm
    bone reached from deep in the torso only by going up-over-down -- gets a
    large d_b/d_min ratio and is suppressed. Unreachable bones (across an air
    gap) have d=inf and contribute nothing. A few Laplacian smoothing passes
    remove voxel blockiness. Returns W (M×B) float32."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    D = np.full((M, B), np.inf, dtype=np.float64)
    for b in range(B):
        src = bone_sources[b]
        if src.size == 0:
            continue
        D[:, b] = csg.dijkstra(G, directed=False, indices=src, min_only=True)
    eps = cell * 2.0
    dadj = D + eps
    dmin = dadj.min(axis=1)
    ratio = dadj / dmin[:, None]
    win = np.clip((_VOX_WIN_HI - ratio) / (_VOX_WIN_HI - _VOX_WIN_LO), 0.0, 1.0)
    win = win * win * (3.0 - 2.0 * win)  # smoothstep
    win = np.nan_to_num(win, nan=0.0)  # isolated voxels (all bones inf) -> 0
    inv = np.zeros((M, B), dtype=np.float64)
    finite = np.isfinite(D)
    inv[finite] = 1.0 / dadj[finite] ** 2
    W = (inv * win).astype(np.float32)
    rs = W.sum(axis=1)
    rs[rs < 1e-12] = 1.0
    W /= rs[:, None]
    # Laplacian smoothing (sparse matvec, fast) to de-blockify the volume field.
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    S = sp.diags((1.0 / deg).astype(np.float32)) @ A
    for _ in range(_VOX_SMOOTH_ITERS):
        W = 0.5 * W + 0.5 * (S @ W)
    rs = W.sum(axis=1)
    rs[rs < 1e-12] = 1.0
    W /= rs[:, None]
    np.clip(W, 0.0, 1.0, out=W)
    return W


def _transfer_voxels_to_verts(co, origin, cell, vid, W_vox, centers):
    """Trilinear interpolation of per-bone voxel weights onto surface verts."""
    import numpy as np
    from scipy.spatial import cKDTree
    nx, ny, nz = vid.shape
    V = co.shape[0]
    B = W_vox.shape[1]
    gf = (co - origin[None, :]) / cell - 0.5
    i0 = np.floor(gf).astype(int)
    f = gf - i0
    accW = np.zeros((V, B), dtype=np.float64)
    accC = np.zeros(V, dtype=np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ii = i0[:, 0] + dx
                jj = i0[:, 1] + dy
                kk = i0[:, 2] + dz
                inb = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny) & (kk >= 0) & (kk < nz)
                cx = f[:, 0] if dx else (1.0 - f[:, 0])
                cy = f[:, 1] if dy else (1.0 - f[:, 1])
                cz = f[:, 2] if dz else (1.0 - f[:, 2])
                coeff = cx * cy * cz
                iic = np.clip(ii, 0, nx - 1)
                jjc = np.clip(jj, 0, ny - 1)
                kkc = np.clip(kk, 0, nz - 1)
                vids = vid[iic, jjc, kkc]
                valid = inb & (vids >= 0) & (coeff > 0)
                if not np.any(valid):
                    continue
                vv = vids[valid]
                cc = coeff[valid]
                accW[valid] += W_vox[vv] * cc[:, None]
                accC[valid] += cc
    # verts with no solid coverage -> nearest occupied voxel
    miss = np.where(accC <= 1e-9)[0]
    if miss.size:
        tree = cKDTree(centers)
        _, nv = tree.query(co[miss])
        accW[miss] = W_vox[nv]
        accC[miss] = 1.0
    accW /= accC[:, None]
    return accW


def _surface_harmonic_W(co, tri, heads, tails):
    """Lean surface harmonic (nearest-bone seed, no visibility) used ONLY for
    fine regions. Surface connectivity keeps fingers/lips separate (the one
    thing the volumetric solid fuses)."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    V = co.shape[0]
    B = heads.shape[0]
    D = _all_bone_distances(co, heads, tails)
    nearest = np.argmin(D, axis=1)
    P = sp.coo_matrix((np.ones(V), (np.arange(V), nearest)), shape=(V, B)).tocsc()
    L = _cotangent_laplacian(co, tri)
    Ldiag = L.diagonal()
    h = float(np.median(Ldiag[Ldiag > 0]))
    if not np.isfinite(h) or h <= 0:
        h = 1.0
    A = (L + h * sp.identity(V, format='csr')).tocsc()
    try:
        solve = spla.factorized(A)
    except Exception:
        solve = None
    W = np.zeros((V, B), dtype=np.float64)
    for b in range(B):
        rhs = h * np.asarray(P[:, b].todense()).ravel()
        if rhs.max() <= 0:
            continue
        if solve is not None:
            W[:, b] = solve(rhs)
        else:
            x, _i = spla.cg(A, rhs, maxiter=400)
            W[:, b] = x
    np.clip(W, 0.0, None, out=W)
    rs = W.sum(axis=1)
    rs[rs < 1e-8] = 1.0
    W /= rs[:, None]
    return W


def compute_weights_voxel(data):
    t0 = time.time()
    import numpy as np

    mesh_obj, V_in, T_in, dupe_faces, repaired_V = _prepare_scene(data)
    bones = data['bones']
    B = len(bones)
    names, heads, tails = _bone_arrays(bones)
    co, nrm, tri = _read_mesh_numpy(mesh_obj)
    V = co.shape[0]
    log(f"Voxel: {V} verts, {tri.shape[0]} tris, {B} bones")

    # --- 1. voxelize to a solid grid (auto-resolution under the occupied cap) -
    t_v = time.time()
    res = _VOX_TARGET_AXIS
    solid, origin, cell = _voxelize_solid(co, tri, res)
    M0 = int(solid.sum())
    if M0 > _VOX_MAX_OCCUPIED:
        res2 = max(_VOX_MIN_AXIS, int(res * (_VOX_MAX_OCCUPIED / max(M0, 1)) ** (1.0 / 3.0)))
        log(f"Voxel: {M0} occupied > cap {_VOX_MAX_OCCUPIED}; res {res}->{res2}")
        solid, origin, cell = _voxelize_solid(co, tri, res2)
        res = res2
        M0 = int(solid.sum())
    frac = M0 / float(solid.size)
    log(f"Voxel: res={res} cell={cell:.4f} solid={M0} ({frac*100:.1f}% of grid) in {round(time.time()-t_v,2)}s")
    if M0 < 8:
        raise RuntimeError(f"voxelization degenerate ({M0} solid voxels)")
    if frac > 0.985:
        raise RuntimeError(f"voxelization filled {frac*100:.1f}% of grid (bad normals / inverted mesh)")

    # --- 2. voxel graph + bone source voxels ---------------------------------
    t_g = time.time()
    vid, M, G, A, occ = _build_voxel_graph(solid, cell)
    centers = origin[None, :] + (occ + 0.5) * cell
    bone_sources = _voxel_bone_sources(heads, tails, origin, cell, vid, solid, centers)
    n_src_sparse = int(sum(s.size for s in bone_sources))
    # Geodesic-gated dense Voronoi seeding: give every bone (esp. leaf bones
    # like hand/head/breast that get a tiny stub from getBoneSegments) its full
    # volumetric region, while rejecting gap-jumps. This is the structural fix
    # for lowerarm->hand, spine->breast, neck->head, and thigh<->thigh bleed.
    bone_sources = _densify_sources(G, bone_sources, centers, heads, tails, M, B, cell)
    n_src = int(sum(s.size for s in bone_sources))
    log(f"Voxel: graph {M} nodes, {G.nnz//2} edges, {n_src_sparse}->{n_src} source voxels (gated dense) in {round(time.time()-t_g,2)}s")

    # --- 3. per-bone geodesic influence through the volume -------------------
    t_s = time.time()
    W_vox = _solve_voxel_geodesic(G, A, M, bone_sources, B, cell)
    log(f"Voxel: volume geodesic solve {round(time.time()-t_s,2)}s")

    # --- 4. transfer to surface vertices -------------------------------------
    t_t = time.time()
    W = _transfer_voxels_to_verts(co, origin, cell, vid, W_vox, centers)
    log(f"Voxel: transfer {round(time.time()-t_t,2)}s")

    # --- 5. fine-region surface pass (fingers / face) ------------------------
    fine_bones = np.array([
        any(p in nm.lower() for p in _VOX_FINE_BONE_PATTERNS) for nm in names
    ], dtype=bool)
    fine_verts_count = 0
    if fine_bones.any():
        t_f = time.time()
        W_surf = _surface_harmonic_W(co, tri, heads, tails)
        dom_vox = np.argmax(W, axis=1)
        dom_surf = np.argmax(W_surf, axis=1)
        fine_vert = fine_bones[dom_vox] | fine_bones[dom_surf]
        fine_verts_count = int(fine_vert.sum())
        W[fine_vert] = W_surf[fine_vert]
        log(f"Voxel: fine surface pass on {fine_verts_count} verts "
            f"({int(fine_bones.sum())} fine bones) in {round(time.time()-t_f,2)}s")

    # --- 6. max-4 influences, normalize, rescue ------------------------------
    if B > _VOX_MAX_INFLUENCES:
        keep = np.argpartition(W, B - _VOX_MAX_INFLUENCES, axis=1)[:, -_VOX_MAX_INFLUENCES:]
        mask = np.zeros_like(W, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        W[~mask] = 0.0
    rowsum = W.sum(axis=1)
    zero_weight_count = int((rowsum < 1e-8).sum())
    if zero_weight_count:
        # rescue all-zero verts to nearest bone by segment distance
        D = _all_bone_distances(co, heads, tails)
        zidx = np.where(rowsum < 1e-8)[0]
        W[zidx, np.argmin(D[zidx], axis=1)] = 1.0
        rowsum[zidx] = 1.0
    W /= rowsum[:, None]

    # --- 6.5 remap repaired-mesh weights to the INPUT vertex order -----------
    # CRITICAL: Blender's repair_mesh (remove_doubles / delete_loose) can
    # reorder and merge vertices, so W is indexed by the REPAIRED mesh, not the
    # array the client sent. The client maps weights back by INPUT index
    # (weights[si][vi]), so returning repaired-order indices silently scrambles
    # every weight (the long-standing "speckle"/cross-gap bleed). Resample W at
    # each input vertex via its nearest repaired vertex so the indices the
    # client expects are correct.
    from scipy.spatial import cKDTree as _cKDTree
    V_input = np.asarray(data['vertices'], dtype=np.float64)
    nn = _cKDTree(co).query(V_input, k=1)[1]
    W = W[nn]
    log(f"Voxel: remapped repaired->input order ({co.shape[0]} repaired -> {V_input.shape[0]} input verts)")

    # --- 7. extract sparse weights dict (INPUT vertex order) -----------------
    t_e = time.time()
    weights = {}
    nz_v, nz_b = np.where(W > _VOX_WMIN)
    for vi, b in zip(nz_v.tolist(), nz_b.tolist()):
        bucket = weights.get(names[b])
        if bucket is None:
            bucket = {}
            weights[names[b]] = bucket
        bucket[str(vi)] = round(float(W[vi, b]), 6)
    log(f"Voxel: extract {round(time.time()-t_e,2)}s, {len(weights)} bones")

    elapsed = time.time() - t0
    bone_weight_stats = {}
    for bname, bw in weights.items():
        ws = list(bw.values())
        bone_weight_stats[bname] = {
            'verts': len(ws),
            'min_w': round(min(ws), 4) if ws else 0,
            'max_w': round(max(ws), 4) if ws else 0,
            'avg_w': round(sum(ws) / len(ws), 4) if ws else 0,
        }
    log(f"Voxel done: {len(weights)} bones, {zero_weight_count} zero verts, {elapsed:.1f}s")

    # Lab debug: return the repaired-mesh vertex positions so an external
    # harness can remap server weight indices to its own mesh by position
    # (Blender's repair_mesh can reorder vertices vs the input array).
    debug_positions_b64 = None
    if data.get('debug_positions'):
        import base64
        debug_positions_b64 = base64.b64encode(
            co.astype('float32').tobytes()).decode('ascii')

    return {
        'weights': weights,
        'bone_count': len(weights),
        'weight_method': 'VOXEL_VOLUMETRIC',
        'debug_positions_b64': debug_positions_b64,
        'diagnostics': {
            'input_verts': V_in,
            'input_tris': T_in,
            'repaired_verts': repaired_V,
            'duplicate_faces': dupe_faces,
            'zero_weight_verts': zero_weight_count,
            'bones_with_weights': len(weights),
            'bones_requested': B,
            'solver': {
                'method': 'voxel_volumetric',
                'voxel_res': res,
                'voxel_cell': round(cell, 5),
                'solid_voxels': M0,
                'graph_edges': int(G.nnz // 2),
                'grid_fill_frac': round(frac, 4),
                'source_voxels': n_src,
                'fine_bones': int(fine_bones.sum()),
                'fine_verts': fine_verts_count,
            },
            'bone_weight_stats': bone_weight_stats,
            'timing': {'total_s': round(elapsed, 2)},
        },
        'elapsed': round(elapsed, 2),
    }


def compute_weights(data):
    """Dispatch: voxel (volumetric, robust) by default; harmonic then bone-heat
    as fallbacks. Each stage records why it fell back in the returned JSON."""
    method = (data.get('method') or 'voxel').lower()

    def _harmonic_then_boneheat(prev_err=None, prev_tb=None):
        try:
            res = compute_weights_harmonic(data)
        except Exception as e2:
            tb2 = traceback.format_exc()
            log(f"Harmonic failed ({type(e2).__name__}: {e2}); bone-heat")
            traceback.print_exc()
            res = compute_weights_bone_heat(data)
            if isinstance(res, dict):
                d = res.setdefault('diagnostics', {})
                d['harmonic_fallback_error'] = f"{type(e2).__name__}: {e2}"
                d['harmonic_fallback_traceback'] = tb2[-2000:]
        if prev_err and isinstance(res, dict):
            d = res.setdefault('diagnostics', {})
            d['voxel_fallback_error'] = prev_err
            d['voxel_fallback_traceback'] = (prev_tb or '')[-2000:]
        return res

    if method in ('voxel', 'volumetric', 'voxel_volumetric'):
        try:
            return compute_weights_voxel(data)
        except Exception as e:
            tb = traceback.format_exc()
            log(f"Voxel solver failed ({type(e).__name__}: {e}); falling back")
            traceback.print_exc()
            return _harmonic_then_boneheat(f"{type(e).__name__}: {e}", tb)

    if method in ('harmonic', 'visibility', 'harmonic_visibility'):
        try:
            return compute_weights_harmonic(data)
        except Exception as e:
            tb = traceback.format_exc()
            log(f"Harmonic solver failed ({type(e).__name__}: {e}); falling back to bone-heat")
            traceback.print_exc()
            result = compute_weights_bone_heat(data)
            if isinstance(result, dict):
                diag = result.setdefault('diagnostics', {})
                diag['harmonic_fallback_error'] = f"{type(e).__name__}: {e}"
                diag['harmonic_fallback_traceback'] = tb[-2000:]
            return result

    return compute_weights_bone_heat(data)


def compute_weights_bone_heat(data):
    t0 = time.time()
    vertices = data['vertices']
    triangles = data['triangles']
    bones = data['bones']
    V = len(vertices)
    T = len(triangles)
    B = len(bones)
    log(f"Input: {V} verts, {T} tris, {B} bones")

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.armatures:
        if block.users == 0:
            bpy.data.armatures.remove(block)

    mesh_data = bpy.data.meshes.new('WeightMesh')
    bm = bmesh.new()
    bm_verts = []
    for v in vertices:
        bm_verts.append(bm.verts.new(Vector(v)))
    bm.verts.ensure_lookup_table()
    dupe_faces = 0
    for tri in triangles:
        try:
            bm.faces.new([bm_verts[tri[0]], bm_verts[tri[1]], bm_verts[tri[2]]])
        except ValueError:
            dupe_faces += 1
    bm.to_mesh(mesh_data)
    bm.free()
    mesh_data.update()

    mesh_obj = bpy.data.objects.new('WeightMesh', mesh_data)
    bpy.context.collection.objects.link(mesh_obj)
    log(f"Mesh created: {len(mesh_data.vertices)} verts, {len(mesh_data.polygons)} faces (skipped {dupe_faces} dupes)")

    repaired_V = repair_mesh(mesh_obj)
    armature_obj = build_armature(bones)

    t_heat = time.time()
    weight_method = try_auto_weight(mesh_obj, armature_obj)
    heat_elapsed = round(time.time() - t_heat, 2)
    log(f"Bone heat solve: {heat_elapsed}s ({weight_method})")

    # Smooth FIRST, then clamp. Smoothing can re-introduce bleed across gaps,
    # so clamping must be the final distance-enforcement step.
    t_smooth = time.time()
    smooth_vertex_groups(mesh_obj, passes=1, factor=0.15)

    if weight_method == 'ARMATURE_ENVELOPE':
        smooth_vertex_groups(mesh_obj, passes=2, factor=0.3)
    smooth_elapsed = round(time.time() - t_smooth, 2)
    log(f"Smoothing: {smooth_elapsed}s")

    # Two-phase weight clamping (AFTER smoothing):
    # Phase 1 — Island: strip cross-island bleed (edge connectivity)
    # Phase 2 — Cross-limb distance: strip same-mesh bleed where a bone
    #           on limb A influences vertices much closer to limb B
    t_clamp = time.time()
    clamped_count, island_debug = clamp_weights_combined(mesh_obj, armature_obj)
    clamp_elapsed = round(time.time() - t_clamp, 2)
    log(f"Weight clamping: {clamp_elapsed}s")

    t_extract = time.time()
    weights, zero_weight_count = extract_weights(mesh_obj, repaired_V)
    extract_elapsed = round(time.time() - t_extract, 2)
    log(f"Weight extraction: {extract_elapsed}s")

    # Build per-bone weight stats for debugging
    bone_weight_stats = {}
    for bname, bw in weights.items():
        ws = list(bw.values())
        bone_weight_stats[bname] = {
            'verts': len(ws),
            'min_w': round(min(ws), 4) if ws else 0,
            'max_w': round(max(ws), 4) if ws else 0,
            'avg_w': round(sum(ws) / len(ws), 4) if ws else 0,
        }

    # Vertex positions for debug visualization (first 5000 verts to keep size down)
    mesh_data = mesh_obj.data
    debug_vert_positions = []
    for vi in range(min(repaired_V, 5000)):
        co = mesh_data.vertices[vi].co
        debug_vert_positions.append([round(co.x, 5), round(co.y, 5), round(co.z, 5)])

    elapsed = time.time() - t0
    log(f"Done: {len(weights)} bones with weights, {zero_weight_count} zero-weight verts, {elapsed:.1f}s total")

    return {
        'weights': weights,
        'bone_count': len(weights),
        'weight_method': weight_method,
        'diagnostics': {
            'input_verts': V,
            'input_tris': T,
            'repaired_verts': repaired_V,
            'duplicate_faces': dupe_faces,
            'zero_weight_verts': zero_weight_count,
            'bones_with_weights': len(weights),
            'bones_requested': B,
            'island_clamped_entries': clamped_count,
            'islands': island_debug,
            'bone_weight_stats': bone_weight_stats,
            'debug_vert_positions': debug_vert_positions,
            'timing': {
                'bone_heat_s': heat_elapsed,
                'smooth_s': smooth_elapsed,
                'clamp_s': clamp_elapsed,
                'extract_s': extract_elapsed,
                'total_s': round(elapsed, 2),
            },
        },
        'elapsed': round(elapsed, 2),
    }


def main():
    argv = sys.argv
    try:
        sep = argv.index('--')
    except ValueError:
        log("ERROR: Missing '--' separator")
        return

    input_path = argv[sep + 1]
    output_path = argv[sep + 2]

    with open(input_path) as f:
        data = json.load(f)

    try:
        result = compute_weights(data)
    except Exception as e:
        log(f"FAILED: {e}")
        traceback.print_exc()
        result = {'error': str(e), 'traceback': traceback.format_exc()}

    with open(output_path, 'w') as f:
        json.dump(result, f)
    log(f"Output written to {output_path}")


if __name__ == '__main__':
    main()
else:
    main()
