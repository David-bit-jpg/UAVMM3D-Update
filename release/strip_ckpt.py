# 把训练存档（带优化器）瘦身成发布版权重 {'model_state', 'epoch', 'src'}，与原来 output/release_* 里的文件同格式
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location='cpu', weights_only=False)
torch.save({'model_state': ck['model_state'], 'epoch': ck.get('epoch'), 'src': src}, dst)
print('写出', dst)
