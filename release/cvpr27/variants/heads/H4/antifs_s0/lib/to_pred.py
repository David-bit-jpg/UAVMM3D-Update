# [review1113 copy of srcx/lib/to_pred.py -> E:/Open3DUAVDet/output/camnorm/geo1004/gproto/review1113/heads/H4/antifs_s0/lib] sys.path -> redirected sx; code unchanged
# [outdoor] detector records -> prediction pickle for score.py: list of dict(seq, frame, boxes9d (1x9 camera frame), scores)
import pickle, sys, numpy as np
d = pickle.load(open(sys.argv[1], 'rb')); sp = sys.argv[2]; out = sys.argv[3]
P = [dict(seq=r['seq'], frame=r['frame'], boxes9d=np.asarray(r['box'], np.float64).reshape(1, 9), scores=np.array([r['conf']]))
     for r in d['recs'][sp] if r['box'] is not None]
pickle.dump(P, open(out, 'wb')); print(len(P), 'preds ->', out)
