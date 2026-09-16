# -*- coding: utf-8 -*-
"""UE editor Python（无头）：逐个机型导出无人机资产，供离线渲染与机头/尺度核对。

每个 BP_Drone01_<model>：在原点、零旋转处生成一个实例（组件世界变换 == 相对 actor 的变换），对每个可见 StaticMeshComponent：
  - 组件世界变换、网格资产路径、网格包围盒
  - 网格导入设置（asset_import_data 的全部可读属性：源文件、导入旋转/平移/缩放、force_front_x_axis 等）
  - LOD0 每个分段：材质槽名、材质路径、父材质、向量/贴图参数、用到的贴图；
    顶点（已变换到 actor 系，cm）、UV0、三角形索引 -> 二进制文件
  - 分段用到的贴图导出成 TGA（离线按 UV 取色）
BoundingCheck 盒：未缩放 extent、世界变换。什么都不保存回工程（实例生成后销毁）。

    "C:/Program Files/Epic Games/UE_5.8/Engine/Binaries/Win64/UnrealEditor-Cmd.exe" E:/UavIndoorSim/UavIndoorSim.uproject
        -run=pythonscript -script=E:/Open3DUAVDet/tools/annot_audit/sim_assets/dump_drone_assets_v2.py
        -nullrhi -unattended -nosplash -stdout -FullStdOutLogOutput
输出：E:/Open3DUAVDet/output/camnorm/annot_audit/assets_v2/
"""
import json
import os
import traceback
from array import array

import unreal

OUT = 'E:/Open3DUAVDet/output/camnorm/annot_audit/assets_v2'
ROOT = '/Game/Carla/Blueprints/Vehicles/Drone_Pack/Drone_Bp/test_drones/'
MODELS = ['DJI-avata2', 'DJI-mavic-mini', 'DJI-phantom4', 'Matrice-600-Pro', 'drone-unk3', 'm210-rtk', 'matrix-300-RTK']
IMPORT_PROPS = ['import_translation', 'import_rotation', 'import_uniform_scale', 'convert_scene', 'force_front_x_axis',
                'convert_scene_unit', 'transform_vertex_to_absolute', 'bake_pivot_in_vertex', 'combine_meshes',
                'vertex_color_import_option', 'normal_import_method', 'reorder_material_to_fbx_order']
TEX_PARAM_HINTS = ('base', 'albedo', 'diffuse', 'color', 'colour', 'tex', 'map')


def log(s):
    unreal.log('[dump_assets_v2] %s' % s)


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


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        return 'ERR:%s' % e


def jsonable(x):
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, unreal.Vector):
        return v3(x)
    if isinstance(x, unreal.Rotator):
        return rot(x)
    if isinstance(x, unreal.Object):
        return x.get_path_name()
    try:
        return str(x)
    except Exception:
        return None


def import_data(sm):
    d = {}
    aid = safe(sm.get_editor_property, 'asset_import_data')
    if not isinstance(aid, unreal.Object):
        return {'asset_import_data': str(aid)}
    d['class'] = aid.get_class().get_name()
    d['source_files'] = jsonable(safe(aid.extract_filenames))
    for p in IMPORT_PROPS:
        val = safe(aid.get_editor_property, p)
        if not (isinstance(val, str) and val.startswith('ERR:')):
            d[p] = jsonable(val)
    return d


def export_texture(tex, done):
    path = tex.get_path_name()
    if path in done:
        return done[path]
    fn = os.path.join(OUT, 'tex', path.strip('/').replace('/', '__').replace('.', '_') + '.tga')
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    try:
        task = unreal.AssetExportTask()
        task.object = tex
        task.filename = fn
        task.automated = True
        task.prompt = False
        task.replace_identical = True
        ok = unreal.Exporter.run_asset_export_task(task)
        done[path] = fn if ok and os.path.exists(fn) else 'ERR:export failed'
    except Exception as e:
        done[path] = 'ERR:%s' % e
    return done[path]


def material_info(mi, tex_done):
    d = {'path': mi.get_path_name() if mi else None}
    if mi is None:
        return d
    d['class'] = mi.get_class().get_name()
    mel = unreal.MaterialEditingLibrary
    try:
        base = mi
        chain = []
        while isinstance(base, unreal.MaterialInstance):
            chain.append(base.get_path_name())
            base = base.get_editor_property('parent')
        d['parent_chain'] = chain + ([base.get_path_name()] if base else [])
    except Exception as e:
        d['parent_chain'] = 'ERR:%s' % e
    vec, texp = {}, {}
    try:
        if isinstance(mi, unreal.MaterialInstance):
            for n in safe(mel.get_vector_parameter_names, mi) or []:
                c = safe(mel.get_material_instance_vector_parameter_value, mi, n)
                vec[str(n)] = [float(c.r), float(c.g), float(c.b), float(c.a)] if hasattr(c, 'r') else str(c)
            for n in safe(mel.get_texture_parameter_names, mi) or []:
                t = safe(mel.get_material_instance_texture_parameter_value, mi, n)
                texp[str(n)] = t.get_path_name() if isinstance(t, unreal.Texture) else str(t)
        else:
            for n in safe(mel.get_vector_parameter_names, mi) or []:
                c = safe(mel.get_material_default_vector_parameter_value, mi, n)
                vec[str(n)] = [float(c.r), float(c.g), float(c.b), float(c.a)] if hasattr(c, 'r') else str(c)
            for n in safe(mel.get_texture_parameter_names, mi) or []:
                t = safe(mel.get_material_default_texture_parameter_value, mi, n)
                texp[str(n)] = t.get_path_name() if isinstance(t, unreal.Texture) else str(t)
    except Exception as e:
        d['param_err'] = '%s' % e
    d['vector_params'] = vec
    d['texture_params'] = texp
    used = []
    try:
        base_mat = mi.get_base_material() if hasattr(mi, 'get_base_material') else mi
        for t in safe(mel.get_used_textures, base_mat) or []:
            if isinstance(t, unreal.Texture):
                used.append(t.get_path_name())
    except Exception as e:
        d['used_err'] = '%s' % e
    d['used_textures'] = used
    # 选一张“底色”贴图导出：优先贴图参数里名字像 base/albedo/diffuse 的，其次用到的第一张
    pick = None
    for n, p in texp.items():
        if any(h in n.lower() for h in TEX_PARAM_HINTS) and p.startswith('/'):
            pick = p
            break
    if pick is None:
        cands = [p for p in list(texp.values()) + used if isinstance(p, str) and p.startswith('/')]
        nonnormal = [p for p in cands if not any(k in p.lower() for k in ('normal', '_n.', '_n_', 'rough', 'metal', 'orm', 'ao', 'mask'))]
        pick = (nonnormal or cands or [None])[0]
    d['basecolor_texture'] = pick
    if pick:
        tex = unreal.EditorAssetLibrary.load_asset(pick.split('.')[0])
        if isinstance(tex, unreal.Texture):
            d['basecolor_texture_file'] = export_texture(tex, tex_done)
    return d


def dump_mesh_component(c, model, tex_done):
    d = {'name': c.get_name(), 'class': c.get_class().get_name()}
    wt = c.get_world_transform()
    d['world_location'] = v3(wt.translation)
    d['world_rotation'] = rot(safe(lambda: wt.rotation.rotator()))
    d['world_scale'] = v3(wt.scale3d)
    d['relative_rotation'] = rot(safe(c.get_editor_property, 'relative_rotation'))
    d['relative_location'] = v3(safe(c.get_editor_property, 'relative_location'))
    d['relative_scale3d'] = v3(safe(c.get_editor_property, 'relative_scale3d'))
    d['visible'] = safe(c.is_visible)
    d['hidden_in_game'] = safe(c.get_editor_property, 'hidden_in_game')
    sm = safe(c.get_editor_property, 'static_mesh')
    if not isinstance(sm, unreal.StaticMesh):
        d['static_mesh'] = str(sm)
        return d
    d['static_mesh'] = sm.get_path_name()
    bb = safe(sm.get_bounding_box)
    d['mesh_bbox_local_cm'] = {'min': v3(bb.min), 'max': v3(bb.max)} if hasattr(bb, 'min') else str(bb)
    d['import'] = import_data(sm)
    mats = c.get_materials()
    d['sections'] = []
    n_sec = int(sm.get_num_sections(0))
    for s in range(n_sec):
        rec = {'index': s}
        try:
            mi_idx = safe(sm.get_material_index, 0, s) if hasattr(sm, 'get_material_index') else s
            rec['material_index'] = mi_idx if isinstance(mi_idx, int) else s
        except Exception:
            rec['material_index'] = s
        try:
            slots = sm.get_editor_property('static_materials')
            rec['slot_name'] = str(slots[rec['material_index']].material_slot_name) if rec['material_index'] < len(slots) else None
        except Exception as e:
            rec['slot_name'] = 'ERR:%s' % e
        mi = mats[rec['material_index']] if rec['material_index'] < len(mats) else None
        rec['material'] = material_info(mi, tex_done)
        try:
            verts, tris, normals, uvs, tangents = unreal.ProceduralMeshLibrary.get_section_from_static_mesh(sm, 0, s)
            vb, ub, tb = array('f'), array('f'), array('i')
            for v in verts:
                w = unreal.MathLibrary.transform_location(wt, v)
                vb.extend((float(w.x), float(w.y), float(w.z)))
            for uv in uvs:
                ub.extend((float(uv.x), float(uv.y)))
            tb.extend(int(t) for t in tris)
            base = os.path.join(OUT, 'mesh', '%s__%s__s%02d' % (model, c.get_name().replace(' ', '_'), s))
            os.makedirs(os.path.dirname(base), exist_ok=True)
            with open(base + '.v.f32', 'wb') as f:
                vb.tofile(f)
            with open(base + '.uv.f32', 'wb') as f:
                ub.tofile(f)
            with open(base + '.tri.i32', 'wb') as f:
                tb.tofile(f)
            rec['n_verts'], rec['n_tris'] = len(verts), len(tris) // 3
            rec['files'] = base
            xs = vb[0::3]; ys = vb[1::3]; zs = vb[2::3]
            if len(xs):
                rec['centroid_actor_cm'] = [sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)]
                rec['bbox_actor_cm'] = [[min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]]
        except Exception as e:
            rec['geom_err'] = '%s' % e
        d['sections'].append(rec)
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        if unreal.EditorLevelLibrary.get_editor_world() is None:
            unreal.EditorLevelLibrary.load_level('/Engine/Maps/Entry')
    except Exception as e:
        log('world check failed: %s' % e)
    result, tex_done = {'models': {}}, {}
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for m in MODELS:
        rec = {'asset': ROOT + 'BP_Drone01_' + m}
        log('===== %s' % m)
        try:
            bp = unreal.EditorAssetLibrary.load_asset(rec['asset'])
            actor = eas.spawn_actor_from_class(bp.generated_class(), unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
            rec['mesh_components'], rec['boxes'] = [], []
            for c in actor.get_components_by_class(unreal.SceneComponent):
                if isinstance(c, unreal.StaticMeshComponent):
                    try:
                        mc = dump_mesh_component(c, m, tex_done)
                    except Exception as e:
                        mc = {'name': safe(c.get_name), 'err': '%s\n%s' % (e, traceback.format_exc())}
                    rec['mesh_components'].append(mc)
                    log('  mesh comp %s visible=%s mesh=%s sections=%d import=%s' % (
                        mc.get('name'), mc.get('visible'), mc.get('static_mesh'), len(mc.get('sections', [])), mc.get('import')))
                elif isinstance(c, unreal.BoxComponent):
                    wt = c.get_world_transform()
                    rec['boxes'].append({'name': c.get_name(), 'unscaled_extent_cm': v3(c.get_unscaled_box_extent()),
                                         'world_location': v3(wt.translation), 'world_rotation': rot(safe(lambda: wt.rotation.rotator())),
                                         'world_scale': v3(wt.scale3d)})
            actor.destroy_actor()
        except Exception as e:
            rec['err'] = '%s\n%s' % (e, traceback.format_exc())
            log('ERROR %s: %s' % (m, rec['err']))
        result['models'][m] = rec
        with open(os.path.join(OUT, 'assets_v2.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
    result['textures'] = tex_done
    with open(os.path.join(OUT, 'assets_v2.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    log('DONE')


main()
