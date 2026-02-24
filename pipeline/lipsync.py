import argparse
import os
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument("--vid",required=True)
p.add_argument("--aud",required=True)
p.add_argument("--out",required=True)
p.add_argument("--w2l",default="third_party/Wav2Lip")
p.add_argument("--ckpt",default="assets/models/wav2lip/wav2lip_gan.pth")
p.add_argument("--pads",default="0 10 0 0")
p.add_argument("--rf",type=int,default=1)
p.add_argument("--bs",type=int,default=1)
p.add_argument("--fbs",type=int,default=1)
p.add_argument("--nosmooth",action="store_true")
a=p.parse_args()

if not os.path.exists(a.vid):
    print("Missing video:",a.vid)
    sys.exit(1)
if not os.path.exists(a.aud):
    print("Missing audio:",a.aud)
    sys.exit(1)
if not os.path.exists(a.w2l):
    print("Missing Wav2Lip folder:",a.w2l)
    print("Fix: git clone the Wav2Lip repo into that path")
    sys.exit(1)

inf=os.path.join(a.w2l,"inference.py")
if not os.path.exists(inf):
    print("Missing:",inf)
    sys.exit(1)
if not os.path.exists(a.ckpt):
    print("Missing checkpoint:",a.ckpt)
    sys.exit(1)

od=os.path.dirname(a.out)
if od and (not os.path.exists(od)):
    os.makedirs(od,exist_ok=True)

ps=a.pads.strip().split()
if len(ps)!=4:
    print("Bad --pads. Need 4 ints like: '0 10 0 0'")
    sys.exit(1)

cmd=[sys.executable,"inference.py","--checkpoint_path",a.ckpt,"--face",a.vid,"--audio",a.aud,"--outfile",a.out,"--pads",ps[0],ps[1],ps[2],ps[3],"--resize_factor",str(a.rf),"--wav2lip_batch_size",str(a.bs),"--face_det_batch_size",str(a.fbs)]
if a.nosmooth:
    cmd.append("--nosmooth")

print("Running Wav2Lip...")
r=subprocess.run(cmd,cwd=a.w2l)
if r.returncode!=0:
    print("Wav2Lip failed")
    sys.exit(r.returncode)
print("Wrote:",a.out)
