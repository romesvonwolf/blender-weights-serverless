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
        best_dist = float('inf')
        best_island = 0

        for vi, v in enumerate(verts):
            d = (v.co - mid).length_squared
            if d < best_dist:
                best_dist = d
                best_island = vert_island[vi]

        bone_islands[bone.name] = {best_island}

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

    all_vg_map = {vg.name: vg for vg in mesh_obj.vertex_groups
                  if vg.name in bone_info}

    for vi in range(V):
        vert_pos = verts[vi].co

        # Compute distance to every non-trunk bone
        bone_dists = []
        for bname, bi in non_trunk_list:
            d = point_to_segment_dist(vert_pos, bi['head'], bi['tail'])
            bone_dists.append((d, bname))

        # Sort by distance — the K nearest are "allowed"
        bone_dists.sort()
        allowed_bones = set(b[1] for b in bone_dists[:MAX_BONE_INFLUENCES])

        # Strip weights from non-trunk bones NOT in the allowed set
        for bname, bi in non_trunk_list:
            if bname in allowed_bones:
                continue
            vg = all_vg_map.get(bname)
            if not vg:
                continue
            try:
                w = vg.weight(vi)
            except RuntimeError:
                continue
            if w < 0.0001:
                continue

            vg.remove([vi])
            phase2_clamped += 1
            phase2_per_bone[bname] = phase2_per_bone.get(bname, 0) + 1

            if len(phase2_debug_samples) < 20:
                nearest_name = bone_dists[0][1]
                phase2_debug_samples.append({
                    'vi': vi,
                    'stripped_bone': bname,
                    'stripped_dist': round(point_to_segment_dist(
                        vert_pos, bone_info[bname]['head'], bone_info[bname]['tail']), 2),
                    'nearest_bone': nearest_name,
                    'nearest_dist': round(bone_dists[0][0], 2),
                    'allowed': [b[1] for b in bone_dists[:MAX_BONE_INFLUENCES]],
                    'weight': round(w, 4),
                    'vert': [round(c, 4) for c in vert_pos],
                })

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
    weights = {}
    zero_weight_count = 0
    vg_names = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
    log(f"Vertex groups created: {len(vg_names)}")

    for vg_idx, vg_name in vg_names.items():
        bone_weights = {}
        for vi in range(V):
            try:
                w = mesh_obj.vertex_groups[vg_idx].weight(vi)
                if w > 0.0001:
                    bone_weights[str(vi)] = round(w, 6)
            except RuntimeError:
                pass
        if bone_weights:
            weights[vg_name] = bone_weights

    for vi in range(V):
        total = 0.0
        for vg_name, bw in weights.items():
            total += bw.get(str(vi), 0.0)
        if total < 0.0001:
            zero_weight_count += 1

    return weights, zero_weight_count


def compute_weights(data):
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
    weight_method = try_auto_weight(mesh_obj, armature_obj)

    # Smooth FIRST, then clamp. Smoothing can re-introduce bleed across gaps,
    # so clamping must be the final distance-enforcement step.
    smooth_vertex_groups(mesh_obj, passes=1, factor=0.15)

    if weight_method == 'ARMATURE_ENVELOPE':
        smooth_vertex_groups(mesh_obj, passes=2, factor=0.3)

    # Two-phase weight clamping (AFTER smoothing):
    # Phase 1 — Island: strip cross-island bleed (edge connectivity)
    # Phase 2 — Cross-limb distance: strip same-mesh bleed where a bone
    #           on limb A influences vertices much closer to limb B
    clamped_count, island_debug = clamp_weights_combined(mesh_obj, armature_obj)

    weights, zero_weight_count = extract_weights(mesh_obj, repaired_V)

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
