# -*- coding: utf-8 -*-
"""[review1113 lidraw] = geo1004/lidar/irlid/strip_aux.py (used to make release/kzl_s*.pth) + pts_head.* (raw-point head, arm b).
Removes training-only auxiliary heads, saves {'model_state', 'epoch', 'src'} (release format).  python strip_aux.py <in.pth> <out.pth>"""
import os, sys
import torch
AUX = ('aux_head.', 'lid_head.', 'pts_head.')
a = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
sd = {k: v for k, v in a['model_state'].items() if not k.startswith(AUX)}
os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
torch.save({'model_state': sd, 'epoch': a.get('epoch'), 'src': sys.argv[1] + ' (aux heads removed: %d tensors)' % (len(a['model_state']) - len(sd))}, sys.argv[2])
print('%s: %d tensors (%d removed), epoch %s' % (os.path.basename(sys.argv[2]), len(sd), len(a['model_state']) - len(sd), a.get('epoch')))
