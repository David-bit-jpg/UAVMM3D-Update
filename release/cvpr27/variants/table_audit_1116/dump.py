import json,sys
for fn in sys.argv[1:]:
    R=json.load(open(fn))
    for f,v in sorted(R.items()):
        if 'err' in v: continue
        s=f.replace('E:/Open3DUAVDet/output/camnorm/geo1004/gproto/','G/').replace('C:/Users/jiang/OneDrive/文档/xwechat_files/wxid_pu1uzzbuzwsx22_b4ed/msg/file/2026-11/results_to_send_20261116_1459/runs','W')
        flag='' if abs(v['AP']-v['AP_default'])<1e-6 else ' TIE-SENS(def AP %.2f UDS %.2f)'%(v['AP_default'],v['UDS_default'])
        cov='' if v['hit']==v['N'] else ' cov %d/%d'%(v['hit'],v['N'])
        print('%-4s %s | AP %.2f ATE %.3f ASE %.2f AOE %.2f mDR %.1f UDS %.2f ties %d%s%s'%('MAV' if v['split']=='real_test' else 'RD',s,v['AP'],v['ATE'],v['ASE'],v['AOE'],v['mDR'],v['UDS'],v['ties'],cov,flag))
