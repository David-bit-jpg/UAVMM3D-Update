# -*- coding: utf-8 -*-
"""UE editor Python (headless): dump the 7 drone blueprints' component hierarchy.

For each BP_Drone01_<model> blueprint under
/Game/Carla/Blueprints/Vehicles/Drone_Pack/Drone_Bp/test_drones/ we
  1) load the asset, spawn one actor of the generated class at the origin with identity rotation
     (so every component world transform == transform relative to the actor),
  2) walk every SceneComponent: name, class, parent, relative loc/rot/scale, world loc/rot/scale,
     component tags, static/skeletal mesh path + its local bounds, BoxComponent unscaled/scaled extent,
  3) dump raw LOD0 vertices of every static mesh (mesh-local, cm) as float32 binaries for offline analysis,
  4) also list the SimpleConstructionScript nodes via SubobjectDataSubsystem (secondary view),
  5) destroy the actor (nothing is saved).

Run:
  "C:/Program Files/Epic Games/UE_5.8/Engine/Binaries/Win64/UnrealEditor-Cmd.exe" E:/UavIndoorSim/UavIndoorSim.uproject
      -run=pythonscript -script=E:/Open3DUAVDet/tools/annot_audit/sim_assets/dump_drone_bp.py
      -nullrhi -unattended -nosplash -stdout -FullStdOutLogOutput
Output: E:/Open3DUAVDet/output/camnorm/annot_audit/sim_assets/drone_bp_dump.json (+ *.f32 vertex files)
"""
import json
import os
import struct
import traceback
from array import array

import unreal

OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/sim_assets'
VERT_DIR = os.path.join(OUT, 'verts')
ROOT = '/Game/Carla/Blueprints/Vehicles/Drone_Pack/Drone_Bp/test_drones/'
MODELS = ['DJI-avata2', 'DJI-mavic-mini', 'DJI-phantom4', 'Matrice-600-Pro', 'drone-unk3', 'm210-rtk', 'matrix-300-RTK']
MAX_VERTS_PER_MESH = 400000


def log(s):
    unreal.log('[dump_drone_bp] %s' % s)


def v3(v):
    try:
        return [float(v.x), float(v.y), float(v.z)]
    except Exception:
        return None


def rot(r):
    try:
        return {'roll': float(r.roll), 'pitch': float(r.pitch), 'yaw': float(r.yaw)}
    except Exception:
        return None


def quat(q):
    try:
        return [float(q.x), float(q.y), float(q.z), float(q.w)]
    except Exception:
        return None


def xf(t):
    try:
        return {'translation': v3(t.translation), 'rotation_quat_xyzw': quat(t.rotation),
                'rotator': rot(t.rotator()), 'scale3d': v3(t.scale3d)}
    except Exception:
        return None


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        return 'ERR:%s' % e


def bounds_info(b):
    """BoxSphereBounds -> dict"""
    try:
        return {'origin': v3(b.origin), 'box_extent': v3(b.box_extent), 'sphere_radius': float(b.sphere_radius)}
    except Exception as e:
        return 'ERR:%s' % e


def box_info(b):
    try:
        return {'min': v3(b.min), 'max': v3(b.max)}
    except Exception as e:
        return 'ERR:%s' % e


def dump_static_mesh_verts(sm, tag):
    """Dump LOD0 vertices (mesh local space, cm) for all sections -> float32 file. Returns info dict."""
    info = {'file': None, 'n_verts': 0, 'n_sections': 0, 'err': None}
    try:
        n_sec = sm.get_num_sections(0)
        info['n_sections'] = int(n_sec)
        buf = array('f')
        total = 0
        for s in range(n_sec):
            res = unreal.ProceduralMeshLibrary.get_section_from_static_mesh(sm, 0, s)
            verts = res[0]
            for v in verts:
                buf.append(float(v.x)); buf.append(float(v.y)); buf.append(float(v.z))
            total += len(verts)
            if total > MAX_VERTS_PER_MESH:
                info['truncated'] = True
                break
        info['n_verts'] = total
        os.makedirs(VERT_DIR, exist_ok=True)
        fn = os.path.join(VERT_DIR, tag + '.f32')
        with open(fn, 'wb') as f:
            buf.tofile(f)
        info['file'] = fn
    except Exception as e:
        info['err'] = '%s' % e
    return info


def comp_record(c, model):
    d = {}
    d['name'] = c.get_name()
    d['class'] = c.get_class().get_name()
    try:
        p = c.get_attach_parent()
        d['parent'] = p.get_name() if p else None
    except Exception as e:
        d['parent'] = 'ERR:%s' % e
    d['relative_location'] = v3(safe(c.get_editor_property, 'relative_location'))
    d['relative_rotation'] = rot(safe(c.get_editor_property, 'relative_rotation'))
    d['relative_scale3d'] = v3(safe(c.get_editor_property, 'relative_scale3d'))
    d['absolute_rotation'] = safe(c.get_editor_property, 'absolute_rotation')
    d['absolute_scale'] = safe(c.get_editor_property, 'absolute_scale')
    d['world_location'] = v3(safe(c.get_world_location))
    d['world_rotation'] = rot(safe(c.get_world_rotation))
    d['world_scale'] = v3(safe(c.get_world_scale))
    d['world_transform'] = xf(safe(c.get_world_transform))
    d['relative_transform'] = xf(safe(c.get_relative_transform))
    try:
        d['tags'] = [str(t) for t in c.get_editor_property('component_tags')]
    except Exception as e:
        d['tags'] = 'ERR:%s' % e
    d['visible'] = safe(c.is_visible)
    d['hidden_in_game'] = safe(c.get_editor_property, 'hidden_in_game')
    # primitive bounds (world space, actor at identity => actor space)
    try:
        d['world_bounds'] = bounds_info(c.get_editor_property('bounds'))
    except Exception:
        try:
            d['world_bounds'] = bounds_info(c.bounds)
        except Exception as e:
            d['world_bounds'] = 'ERR:%s' % e
    if isinstance(c, unreal.PrimitiveComponent):
        try:
            lb = c.get_local_bounds()
            d['local_bounds_minmax'] = [v3(lb[0]), v3(lb[1])]
        except Exception as e:
            d['local_bounds_minmax'] = 'ERR:%s' % e
    if isinstance(c, unreal.StaticMeshComponent):
        sm = safe(c.get_editor_property, 'static_mesh')
        if isinstance(sm, unreal.StaticMesh):
            d['static_mesh'] = sm.get_path_name()
            d['mesh_bounds'] = bounds_info(safe(sm.get_bounds))
            d['mesh_bounding_box'] = box_info(safe(sm.get_bounding_box))
            d['mesh_num_lods'] = safe(sm.get_num_lods)
            d['mesh_num_verts_lod0'] = safe(sm.get_num_vertices, 0)
            tag = '%s__%s' % (model, c.get_name())
            d['verts'] = dump_static_mesh_verts(sm, tag)
        else:
            d['static_mesh'] = None if sm is None else str(sm)
    if isinstance(c, unreal.SkinnedMeshComponent):
        sk = None
        for getter in ('get_skinned_asset',):
            try:
                sk = getattr(c, getter)()
                break
            except Exception:
                pass
        if sk is None:
            for prop in ('skeletal_mesh_asset', 'skeletal_mesh', 'skinned_asset'):
                try:
                    sk = c.get_editor_property(prop)
                    if sk is not None:
                        break
                except Exception:
                    pass
        if sk is not None:
            d['skeletal_mesh'] = sk.get_path_name()
            d['mesh_bounds'] = bounds_info(safe(sk.get_bounds))
            try:
                d['mesh_imported_bounds'] = bounds_info(sk.get_imported_bounds())
            except Exception as e:
                d['mesh_imported_bounds'] = 'ERR:%s' % e
    if isinstance(c, unreal.BoxComponent):
        d['box_unscaled_extent'] = v3(safe(c.get_unscaled_box_extent))
        d['box_scaled_extent'] = v3(safe(c.get_scaled_box_extent))
        d['box_extent_prop'] = v3(safe(c.get_editor_property, 'box_extent'))
    if isinstance(c, unreal.ChildActorComponent):
        try:
            ca = c.get_child_actor()
            d['child_actor_class'] = ca.get_class().get_name() if ca else None
        except Exception as e:
            d['child_actor_class'] = 'ERR:%s' % e
    return d


def scs_records(bp):
    """Secondary view: SimpleConstructionScript nodes via SubobjectDataSubsystem."""
    out = []
    try:
        sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
        handles = sds.k2_gather_subobject_data_for_blueprint(bp)
        lib = unreal.SubobjectDataBlueprintFunctionLibrary
        for h in handles:
            rec = {}
            try:
                data = lib.get_data(h)
                rec['variable_name'] = str(safe(lib.get_variable_name, data))
                rec['display_name'] = str(safe(lib.get_display_name, data))
                obj = safe(lib.get_object, data)
                rec['object'] = obj.get_path_name() if isinstance(obj, unreal.Object) else str(obj)
                rec['object_class'] = obj.get_class().get_name() if isinstance(obj, unreal.Object) else None
                if isinstance(obj, unreal.SceneComponent):
                    rec['relative_location'] = v3(safe(obj.get_editor_property, 'relative_location'))
                    rec['relative_rotation'] = rot(safe(obj.get_editor_property, 'relative_rotation'))
                    rec['relative_scale3d'] = v3(safe(obj.get_editor_property, 'relative_scale3d'))
                if isinstance(obj, unreal.StaticMeshComponent):
                    sm = safe(obj.get_editor_property, 'static_mesh')
                    rec['static_mesh'] = sm.get_path_name() if isinstance(sm, unreal.StaticMesh) else str(sm)
                if isinstance(obj, unreal.BoxComponent):
                    rec['box_unscaled_extent'] = v3(safe(obj.get_unscaled_box_extent))
                try:
                    ph = lib.get_data(data.get_editor_property('parent_object_handle'))
                    rec['parent_variable_name'] = str(safe(lib.get_variable_name, ph))
                except Exception:
                    pass
            except Exception as e:
                rec['err'] = '%s' % e
            out.append(rec)
    except Exception as e:
        out.append({'err': 'scs failed: %s\n%s' % (e, traceback.format_exc())})
    return out


def spawn(cls):
    try:
        eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        a = eas.spawn_actor_from_class(cls, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        if a:
            return a, 'EditorActorSubsystem'
    except Exception as e:
        log('EditorActorSubsystem spawn failed: %s' % e)
    a = unreal.EditorLevelLibrary.spawn_actor_from_class(cls, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    return a, 'EditorLevelLibrary'


def main():
    os.makedirs(OUT, exist_ok=True)
    result = {'models': {}, 'errors': []}
    try:
        w = unreal.EditorLevelLibrary.get_editor_world()
        log('editor world: %s' % (w.get_path_name() if w else None))
        if w is None:
            unreal.EditorLevelLibrary.load_level('/Engine/Maps/Entry')
            log('loaded /Engine/Maps/Entry')
    except Exception as e:
        log('world check failed: %s' % e)
    for m in MODELS:
        path = ROOT + 'BP_Drone01_' + m
        rec = {'asset': path}
        log('===== %s' % m)
        try:
            bp = unreal.EditorAssetLibrary.load_asset(path)
            if bp is None:
                raise RuntimeError('load_asset returned None')
            rec['bp_class'] = bp.get_class().get_name()
            cls = bp.generated_class()
            rec['generated_class'] = cls.get_name() if cls else None
            try:
                rec['parent_class'] = bp.get_editor_property('parent_class').get_name()
            except Exception as e:
                rec['parent_class'] = 'ERR:%s' % e
            rec['scs'] = scs_records(bp)
            actor, how = spawn(cls)
            if actor is None:
                raise RuntimeError('spawn failed')
            rec['spawned_via'] = how
            rec['actor_class'] = actor.get_class().get_name()
            rec['actor_transform'] = xf(actor.get_actor_transform())
            rec['actor_tags'] = [str(t) for t in actor.tags]
            try:
                o, e = actor.get_actor_bounds(False, True)
                rec['actor_bounds_all'] = {'origin': v3(o), 'extent': v3(e)}
                o, e = actor.get_actor_bounds(True, True)
                rec['actor_bounds_colliding'] = {'origin': v3(o), 'extent': v3(e)}
            except Exception as ex:
                rec['actor_bounds_all'] = 'ERR:%s' % ex
            try:
                rc = actor.root_component
                rec['root_component'] = rc.get_name() if rc else None
            except Exception as ex:
                rec['root_component'] = 'ERR:%s' % ex
            comps = actor.get_components_by_class(unreal.SceneComponent)
            rec['components'] = []
            for c in comps:
                try:
                    cr = comp_record(c, m)
                except Exception as ex:
                    cr = {'name': safe(c.get_name), 'err': '%s\n%s' % (ex, traceback.format_exc())}
                rec['components'].append(cr)
                log('  comp %-40s %-24s parent=%-28s relrot=%s relscale=%s worldrot=%s worldscale=%s' % (
                    cr.get('name'), cr.get('class'), cr.get('parent'), cr.get('relative_rotation'),
                    cr.get('relative_scale3d'), cr.get('world_rotation'), cr.get('world_scale')))
                if 'static_mesh' in cr:
                    log('       mesh=%s bounds=%s bbox=%s verts=%s' % (cr.get('static_mesh'), cr.get('mesh_bounds'), cr.get('mesh_bounding_box'), cr.get('verts')))
                if 'skeletal_mesh' in cr:
                    log('       skel=%s bounds=%s' % (cr.get('skeletal_mesh'), cr.get('mesh_bounds')))
                if 'box_unscaled_extent' in cr:
                    log('       BOX unscaled=%s scaled=%s worldxf=%s' % (cr.get('box_unscaled_extent'), cr.get('box_scaled_extent'), cr.get('world_transform')))
            log('  actor bounds(all)=%s colliding=%s' % (rec.get('actor_bounds_all'), rec.get('actor_bounds_colliding')))
            try:
                actor.destroy_actor()
            except Exception as ex:
                log('destroy failed: %s' % ex)
        except Exception as e:
            rec['err'] = '%s\n%s' % (e, traceback.format_exc())
            log('ERROR %s: %s' % (m, rec['err']))
        result['models'][m] = rec
    fn = os.path.join(OUT, 'drone_bp_dump.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    log('wrote %s' % fn)
    log('DONE')


main()
