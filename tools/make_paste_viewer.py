# -*- coding: utf-8 -*-
"""把 mm_paste_aug.py 输出目录里的 .npz 样本做成一个本地 HTML 查看器（左右箭头翻页，检查标注是否准确）。

    D:/Miniconda3/envs/city/python.exe tools/make_paste_viewer.py --dir output/paste_100
    -> output/paste_100/viewer/index.html   （用浏览器直接打开，图片在同目录 img/ 下，不依赖服务器）

每个样本 6 个面板（RGB 翻译背景 / RGB 仿真背景 / IR / DVS / LiDAR 深度+tag / 雷达热图），都是 512x288 的训练分辨率；
3D 框的 8 个角点由标签 box9d + 该帧 K 现算，页面里用 canvas 画（B 键开关），所以看到的框就是标签本身。
"""
import argparse
import glob
import json
import os
import pickle
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

PROTO = np.array([[-.5, -.5, -.5], [.5, -.5, -.5], [.5, .5, -.5], [-.5, .5, -.5],
                  [-.5, -.5, .5], [.5, -.5, .5], [.5, .5, .5], [-.5, .5, .5]])


def corners_uv(b, K_raw, raw_wh, W, H):
    K = np.array(K_raw, np.float64).copy()
    K[0] *= W / float(raw_wh[0])
    K[1] *= H / float(raw_wh[1])
    pts = (PROTO * b[3:6]) @ Rot.from_euler('xyz', b[6:9]).as_matrix().T + b[:3]
    uv = (K @ pts.T).T
    return (uv[:, :2] / uv[:, 2:3]).round(2).tolist()


def depth_vis(depth, tag):
    d = depth.astype(np.float32) / 100.0
    img = np.zeros(depth.shape + (3,), np.uint8)
    m = depth > 0
    if m.any():
        n = np.clip(d / 40.0, 0, 1)
        col = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_JET)
        img[m] = col[m]
    img = cv2.dilate(img, np.ones((2, 2), np.uint8))
    tm = cv2.dilate(tag, np.ones((3, 3), np.uint8)) > 0
    img[tm] = (255, 0, 255)
    return img


def hm_vis(hm):
    n = hm / max(float(hm.max()), 1e-6)
    return cv2.applyColorMap((255 * n).astype(np.uint8), cv2.COLORMAP_INFERNO)


HTML = r'''<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>Paste-aug viewer</title>
<style>
 body{margin:0;background:#1b1b1b;color:#ddd;font:13px/1.4 system-ui,Segoe UI,Arial}
 #top{display:flex;align-items:center;gap:10px;padding:8px 12px;background:#262626;position:sticky;top:0;z-index:5}
 button{background:#3a3a3a;color:#eee;border:1px solid #555;border-radius:4px;padding:6px 12px;font-size:14px;cursor:pointer}
 button:hover{background:#4a4a4a}
 #idx{font-size:15px;min-width:90px;text-align:center}
 #info{padding:6px 12px;color:#bbb;white-space:pre-wrap}
 #main{display:flex;gap:10px;padding:0 12px 12px;flex-wrap:wrap}
 .panel{position:relative;background:#000;max-width:100%}
 .panel canvas{display:block;max-width:100%;height:auto}
 #top{flex-wrap:wrap}
 .thumb canvas{max-width:100%;height:auto}
 .cap{position:absolute;left:4px;top:2px;color:#fff;font-size:12px;text-shadow:0 0 3px #000}
 #grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;padding:0 12px 16px}
 .thumb{position:relative;cursor:pointer;border:2px solid transparent}
 .thumb.sel{border-color:#6f6}
 #checks{padding:0 12px 16px;color:#bbb}
 table{border-collapse:collapse}td{padding:2px 10px 2px 0}
 kbd{background:#333;border:1px solid #666;border-radius:3px;padding:0 4px}
</style></head><body>
<div id="top">
 <button id="prev">&#9664; 上一个</button><span id="idx"></span><button id="next">下一个 &#9654;</button>
 <button id="tbox">框：开 (B)</button><button id="tlab">标签文字：开 (L)</button>
 <span>放大：</span><input id="zoom" type="range" min="2" max="8" value="4" style="width:90px"><span id="zv">4x</span>
 <span style="margin-left:auto;color:#999">键盘 <kbd>←</kbd> <kbd>→</kbd> 翻页，<kbd>B</kbd> 框，<kbd>1-9</kbd>/<kbd>0</kbd> 切主图，<kbd>Home</kbd>/<kbd>End</kbd></span>
</div>
<div id="info"></div>
<div id="main">
 <div class="panel"><canvas id="big" width="1024" height="576"></canvas><div class="cap" id="bigcap"></div></div>
 <div class="panel"><canvas id="zoomc" width="512" height="576"></canvas><div class="cap">放大（框中心附近）</div></div>
</div>
<div id="grid"></div>
<div id="checks"></div>
<script>
const DATA = __DATA__;
const MODS = [['rgb','RGB 翻译背景（学生输入）','uv'],['rgb_sim','RGB 仿真背景（教师输入）','uv'],['ir','IR','uv'],['dvs','DVS','uv'],
              ['lidar','LiDAR 深度 + tag(品红)，从 B 的新点云投影','uv'],['radar','雷达速度热图','uv'],
              ['rgb_src','A 帧原始 RGB（源，原框）','uv_src'],['lidar_src','A 帧原始 LiDAR（源，原框：原始噪声长这样）','uv_src'],
              ['rgb_bg','B 帧增广前原始 RGB（黄框 = 被抹掉的原无人机）','uv_bg'],['lidar_bg','B 帧增广前原始 LiDAR（稀疏程度对照）','uv_bg']];
const uvFor = (d, k) => (d[MODS[k][2]] || d.uv);
const EDGES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
let cur = 0, mod = 0, showBox = true, showLab = true, zoomF = 4;
const imgs = {};
function loadImg(src){ if(!imgs[src]){ const im=new Image(); im.src=src; imgs[src]=im; } return imgs[src]; }
function drawBox(ctx, uv, sx, sy, ox, oy, lw, label){
  if(!showBox || !uv || uv.length===0) return;
  if(Array.isArray(uv[0][0])){   // 多个框（B 帧原有的无人机）：黄色
    for(const u of uv){ ctx.lineWidth = lw; ctx.strokeStyle = '#e8c832'; ctx.beginPath();
      for(const [i,j] of EDGES){ ctx.moveTo((u[i][0]-ox)*sx,(u[i][1]-oy)*sy); ctx.lineTo((u[j][0]-ox)*sx,(u[j][1]-oy)*sy); }
      ctx.stroke(); }
    return;
  }
  ctx.lineWidth = lw; ctx.strokeStyle = '#3fe03f'; ctx.beginPath();
  for(const [i,j] of EDGES){ ctx.moveTo((uv[i][0]-ox)*sx,(uv[i][1]-oy)*sy); ctx.lineTo((uv[j][0]-ox)*sx,(uv[j][1]-oy)*sy); }
  ctx.stroke();
  // 前面（角 0-3，靠近相机的一面用红色标出，便于看朝向）
  ctx.strokeStyle = '#ff5050'; ctx.beginPath();
  for(const [i,j] of [[0,1],[1,2],[2,3],[3,0]]){ ctx.moveTo((uv[i][0]-ox)*sx,(uv[i][1]-oy)*sy); ctx.lineTo((uv[j][0]-ox)*sx,(uv[j][1]-oy)*sy); }
  ctx.stroke();
  if(showLab && label){ ctx.fillStyle='#3fe03f'; ctx.font = (12*Math.max(1,lw))+'px system-ui';
    const xs = uv.map(p=>p[0]), ys = uv.map(p=>p[1]);
    ctx.fillText(label, (Math.min(...xs)-ox)*sx, (Math.min(...ys)-oy)*sy - 4); }
}
function render(){
  const d = DATA[cur];
  document.getElementById('idx').textContent = (cur+1)+' / '+DATA.length;
  document.getElementById('info').textContent =
    d.name+'\n机型 '+d.cls+'  框 '+d.dim.map(v=>v.toFixed(2)).join(' x ')+' m   距离 '+d.range_old.toFixed(1)+' -> '+d.range_new.toFixed(1)+' m   缩放 '+d.s.toFixed(2)+'   表观 '+d.px.toFixed(0)+' px   matte '+d.matte.toFixed(2)+
    '\n源 A: '+d.A+'\n背景 B: '+d.B+(d.bg_px ? '\nLiDAR 非零像素：B 增广前 '+d.bg_px[0]+' -> 增广后 '+d.bg_px[1]+'（差别只来自抹掉的原无人机和贴入的无人机）' : '');
  const big = document.getElementById('big'), ctx = big.getContext('2d');
  const key = MODS[mod][0];
  const im = loadImg('img/'+d.name+'_'+key+'.jpg');
  const draw = () => {
    ctx.fillStyle='#000'; ctx.fillRect(0,0,big.width,big.height);
    ctx.drawImage(im,0,0,big.width,big.height);
    const uvm = uvFor(d, mod);
    drawBox(ctx, uvm, 2, 2, 0, 0, 2, d.cls+' '+(MODS[mod][2]==='uv_src' ? d.range_old : d.range_new).toFixed(1)+'m');
    document.getElementById('bigcap').textContent = MODS[mod][1];
    // 放大窗：以框中心为中心，取 (512/zoomF) x (576/zoomF) 的窗口
    const zc = document.getElementById('zoomc'), zx = zc.getContext('2d');
    const uvz = (uvm && Array.isArray(uvm[0][0])) ? d.uv : uvm;   // 多框面板的放大窗仍以新框为中心
    const xs = uvz.map(p=>p[0]), ys = uvz.map(p=>p[1]);
    const cx = (Math.min(...xs)+Math.max(...xs))/2, cy = (Math.min(...ys)+Math.max(...ys))/2;
    const ww = zc.width/zoomF/2, wh = zc.height/zoomF/2;   // 512x288 坐标下的窗口尺寸
    let ox = Math.max(0, Math.min(512-ww, cx-ww/2)), oy = Math.max(0, Math.min(288-wh, cy-wh/2));
    zx.imageSmoothingEnabled = false;
    zx.fillStyle='#000'; zx.fillRect(0,0,zc.width,zc.height);
    zx.drawImage(im, ox, oy, ww, wh, 0, 0, zc.width, zc.height);
    drawBox(zx, uvm, zoomF*2, zoomF*2, ox, oy, 2, '');
  };
  im.addEventListener('load', draw, {once:true}); if(im.complete && im.naturalWidth) draw();   // 两条都挂，避免 complete/onload 竞争漏画
  // 缩略图
  const g = document.getElementById('grid'); g.innerHTML='';
  MODS.forEach((m,k)=>{
    const div = document.createElement('div'); div.className='thumb'+(k===mod?' sel':''); div.onclick=()=>{mod=k;render();};
    const c = document.createElement('canvas'); c.width=512; c.height=288; div.appendChild(c);
    const cap = document.createElement('div'); cap.className='cap'; cap.textContent=(k+1)+'. '+m[1]; div.appendChild(cap);
    g.appendChild(div);
    const t = loadImg('img/'+d.name+'_'+m[0]+'.jpg');
    const dr = ()=>{ const x=c.getContext('2d'); x.drawImage(t,0,0,512,288); drawBox(x,uvFor(d,k),1,1,0,0,1,''); };
    t.addEventListener('load', dr, {once:true}); if(t.complete && t.naturalWidth) dr();
  });
  const ck = d.checks || {};
  const f = (k, n=2) => (typeof ck[k]==='number' ? ck[k].toFixed(n) : '—');
  document.getElementById('checks').innerHTML = '<table>'+
    '<tr><td>LiDAR 无人机点在框内比例：A 原始 -> 沿射线去噪 -> 按 B 的 σ 重新加噪</td><td>'+f('lidar_in_box_old')+' -> '+f('lidar_in_box_denoised')+' -> '+f('lidar_in_box_new')+'（'+f('lidar_n',0)+' 点；σ_A '+f('sigma_A',1)+' m，σ_B '+f('sigma_B',1)+' m；射线穿框 '+f('lidar_ray_hit_frac')+'）</td></tr>'+
    '<tr><td>雷达回波在框内比例（前 -> 后）</td><td>'+f('radar_in_box_old')+' -> '+f('radar_in_box_new')+'（'+f('radar_n',0)+' 点）</td></tr>'+
    '<tr><td>贴入像素落在新框凸包内的比例</td><td>'+f('alpha_in_hull')+'</td></tr>'+
    '<tr><td>tag 像素落在新框凸包内的比例</td><td>'+f('tag_px_in_hull')+'</td></tr>'+
    '</table>';
  // 预加载下一个
  if(cur+1<DATA.length) MODS.forEach(m=>loadImg('img/'+DATA[cur+1].name+'_'+m[0]+'.jpg'));
}
document.getElementById('prev').onclick=()=>{cur=(cur-1+DATA.length)%DATA.length;render();};
document.getElementById('next').onclick=()=>{cur=(cur+1)%DATA.length;render();};
document.getElementById('tbox').onclick=()=>{showBox=!showBox;document.getElementById('tbox').textContent='框：'+(showBox?'开':'关')+' (B)';render();};
document.getElementById('tlab').onclick=()=>{showLab=!showLab;document.getElementById('tlab').textContent='标签文字：'+(showLab?'开':'关')+' (L)';render();};
document.getElementById('zoom').oninput=(e)=>{zoomF=+e.target.value;document.getElementById('zv').textContent=zoomF+'x';render();};
document.addEventListener('keydown',(e)=>{
  if(e.key==='ArrowRight'){cur=(cur+1)%DATA.length;render();}
  else if(e.key==='ArrowLeft'){cur=(cur-1+DATA.length)%DATA.length;render();}
  else if(e.key==='b'||e.key==='B'){document.getElementById('tbox').onclick();}
  else if(e.key==='l'||e.key==='L'){document.getElementById('tlab').onclick();}
  else if(e.key==='Home'){cur=0;render();} else if(e.key==='End'){cur=DATA.length-1;render();}
  else if(e.key>='1'&&e.key<='9'){mod=Math.min(+e.key-1, MODS.length-1);render();}
  else if(e.key==='0'){mod=Math.min(9, MODS.length-1);render();}
});
render();
</script></body></html>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='mm_paste_aug.py 的输出目录（含 .npz）')
    ap.add_argument('--out', default='', help='默认 <dir>/viewer')
    ap.add_argument('--cache', default='', help='背景帧所在缓存（给出则多两格：B 帧增广前的原始 RGB / LiDAR，用来对照稀疏程度）')
    ap.add_argument('--split', default='train')
    args = ap.parse_args()
    out = args.out or os.path.join(args.dir, 'viewer')
    os.makedirs(os.path.join(out, 'img'), exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.dir, '*.npz')))
    cache = None
    if args.cache:
        cs = os.path.join(args.cache, args.split)
        cidx = pickle.load(open(os.path.join(cs, 'index.pkl'), 'rb'))
        cache = {'key': {m['seq'] + '/' + m['frame']: i for i, m in enumerate(cidx['metas']) if m},
                 'rgb': np.load(os.path.join(cs, 'rgb.npy'), mmap_mode='r'),
                 'depth': np.load(os.path.join(cs, 'depth.npy'), mmap_mode='r'),
                 'tag': np.load(os.path.join(cs, 'tag.npy'), mmap_mode='r'), 'metas': cidx['metas']}
    data = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        name = os.path.splitext(os.path.basename(f))[0]
        rgb, rgb_sim, ir, dvs = z['rgb'], z['rgb_sim'], z['ir'], z['dvs']
        H, W = rgb.shape[:2]
        panels = {'rgb': rgb, 'rgb_sim': rgb_sim, 'ir': cv2.cvtColor(ir, cv2.COLOR_GRAY2BGR), 'dvs': dvs,
                  'lidar': depth_vis(z['depth'], z['tag']), 'radar': hm_vis(z['radar_hm'])}
        has_src = 'depth_src' in z.files
        if has_src:
            panels['rgb_src'] = z['rgb_src']
            panels['lidar_src'] = depth_vis(z['depth_src'], z['tag_src'])
        bg_px = None
        uv_bg = None
        if cache is not None and str(z['B']) in cache['key']:
            ib = cache['key'][str(z['B'])]
            panels['rgb_bg'] = np.ascontiguousarray(cache['rgb'][ib])
            panels['lidar_bg'] = depth_vis(np.asarray(cache['depth'][ib]), np.asarray(cache['tag'][ib]))
            bg_px = [int((np.asarray(cache['depth'][ib]) > 0).sum()), int((z['depth'] > 0).sum())]
            mb = cache['metas'][ib]
            uv_bg = [corners_uv(np.asarray(bb, np.float64), mb['K_raw'], mb['raw_wh'], W, H) for bb in mb['boxes9d']]
        for k, im in panels.items():
            cv2.imwrite(os.path.join(out, 'img', '%s_%s.jpg' % (name, k)), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        b = z['box9d'].astype(np.float64)
        uv = corners_uv(b, z['K_raw'], z['raw_wh'], W, H)
        uv_src = z['uv_src'].round(2).tolist() if has_src else None
        ck = z['checks'].item() if 'checks' in z else {}
        tr_s = float(z['s'])
        data.append({'name': name, 'cls': str(z['name']), 'dim': [float(v) for v in b[3:6]], 'uv': uv, 'uv_src': uv_src,
                     'uv_bg': uv_bg, 'bg_px': bg_px,
                     'range_old': float(ck.get('range_old', np.linalg.norm(b[:3]) * tr_s)), 'range_new': float(np.linalg.norm(b[:3])),
                     's': tr_s, 'px': float(ck.get('px_new', max(max(p[0] for p in uv) - min(p[0] for p in uv), max(p[1] for p in uv) - min(p[1] for p in uv)))),
                     'matte': float(ck.get('matte_frac', -1)), 'A': str(z['A']), 'B': str(z['B']),
                     'checks': {k: (float(v) if not isinstance(v, str) else v) for k, v in ck.items()}})
    html = HTML.replace('__DATA__', json.dumps(data, ensure_ascii=False))
    with open(os.path.join(out, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)
    print('写出 %s（%d 个样本，%d 张图）' % (os.path.join(out, 'index.html'), len(data), 6 * len(data)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
