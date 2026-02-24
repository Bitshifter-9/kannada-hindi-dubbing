import argparse
import json
import os
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument("--inp",required=True)
p.add_argument("--out",required=True)
p.add_argument("--ref",required=True)
p.add_argument("--model",default="tts_models/multilingual/multi-dataset/xtts_v2")
p.add_argument("--lang",default="hi")
p.add_argument("--sr",type=int,default=16000)
p.add_argument("--redo",action="store_true")
p.add_argument("--gpu",action="store_true")
a=p.parse_args()

try:
    from TTS.api import TTS
except Exception:
    print("Missing dependency: TTS")
    print("Fix:")
    print("  pip install TTS soundfile")
    sys.exit(1)

for c in ["ffmpeg","ffprobe"]:
    r=subprocess.run([c,"-version"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if r.returncode!=0:
        print("Missing command:",c)
        sys.exit(1)

if not os.path.exists(a.inp):
    print("Missing:",a.inp)
    sys.exit(1)
if not os.path.exists(a.ref):
    print("Missing:",a.ref)
    sys.exit(1)

f=open(a.inp,"r",encoding="utf-8")
it=json.load(f)
f.close()

wd="data/interim/tts/wav"
td="data/interim/tts/tmp"
if not os.path.exists(wd):
    os.makedirs(wd,exist_ok=True)
if not os.path.exists(td):
    os.makedirs(td,exist_ok=True)

tts=TTS(model_name=a.model,progress_bar=False,gpu=bool(a.gpu))

out=[]
for s in it:
    i=int(s.get("id",0))
    st=float(s.get("start",0.0))
    en=float(s.get("end",0.0))
    tg=float(s.get("dur",round(en-st,3)))
    tx=(s.get("hi") or "").strip()

    fn="seg_"+str(i).zfill(4)+".wav"
    fp=os.path.join(wd,fn)

    raw=os.path.join(td,"raw_"+str(i).zfill(4)+".wav")
    rs=os.path.join(td,"rs_"+str(i).zfill(4)+".wav")
    tr=os.path.join(td,"tr_"+str(i).zfill(4)+".wav")
    sp=os.path.join(td,"sp_"+str(i).zfill(4)+".wav")

    if (not a.redo) and os.path.exists(fp):
        pass
    else:
        if not tx:
            r=subprocess.run(["ffmpeg","-y","-f","lavfi","-i","anullsrc=r="+str(a.sr)+":cl=mono","-t",str(tg),"-ac","1","-ar",str(a.sr),fp],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if r.returncode!=0:
                print("ffmpeg failed for silence:",i)
                print(r.stderr.decode("utf-8",errors="ignore"))
                sys.exit(1)
        else:
            tts.tts_to_file(text=tx,speaker_wav=a.ref,language=a.lang,file_path=raw)
            r=subprocess.run(["ffmpeg","-y","-i",raw,"-ac","1","-ar",str(a.sr),rs],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if r.returncode!=0:
                print("ffmpeg resample failed:",i)
                print(r.stderr.decode("utf-8",errors="ignore"))
                sys.exit(1)
            r=subprocess.run(["ffmpeg","-y","-i",rs,"-af","silenceremove=start_periods=1:start_silence=0.10:start_threshold=-35dB:stop_periods=1:stop_silence=0.10:stop_threshold=-35dB","-ac","1","-ar",str(a.sr),tr],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if r.returncode!=0:
                print("ffmpeg trim failed:",i)
                print(r.stderr.decode("utf-8",errors="ignore"))
                sys.exit(1)

            r=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",tr],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            ds=r.stdout.decode("utf-8",errors="ignore").strip()
            gd=0.0
            if ds:
                try:
                    gd=float(ds)
                except Exception:
                    gd=0.0

            rt=0.0
            if tg>0.01 and gd>0.01:
                rt=gd/tg

            if tg>0.01 and gd>0.01 and abs(gd-tg)>0.05 and rt>=0.80 and rt<=1.25:
                r=subprocess.run(["ffmpeg","-y","-i",tr,"-af","atempo="+str(round(rt,5)),"-ac","1","-ar",str(a.sr),sp],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                if r.returncode!=0:
                    print("ffmpeg atempo failed:",i)
                    print(r.stderr.decode("utf-8",errors="ignore"))
                    sys.exit(1)
                r=subprocess.run(["ffmpeg","-y","-i",sp,"-af","apad","-t",str(tg),"-ac","1","-ar",str(a.sr),fp],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                if r.returncode!=0:
                    print("ffmpeg pad/cut failed:",i)
                    print(r.stderr.decode("utf-8",errors="ignore"))
                    sys.exit(1)
            else:
                r=subprocess.run(["ffmpeg","-y","-i",tr,"-af","apad","-t",str(tg),"-ac","1","-ar",str(a.sr),fp],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                if r.returncode!=0:
                    print("ffmpeg pad/cut failed:",i)
                    print(r.stderr.decode("utf-8",errors="ignore"))
                    sys.exit(1)

    r=subprocess.run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",fp],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    ds=r.stdout.decode("utf-8",errors="ignore").strip()
    fd=0.0
    if ds:
        try:
            fd=float(ds)
        except Exception:
            fd=0.0

    err=0.0
    if tg>0.01 and fd>0.01:
        err=round(fd-tg,3)

    out.append({"id":i,"start":st,"end":en,"dur_t":tg,"dur_a":fd,"err":err,"wav":fp})
    print("seg",str(i).zfill(4),"t",tg,"a",round(fd,3),"err",err)

od=os.path.dirname(a.out)
if od and (not os.path.exists(od)):
    os.makedirs(od,exist_ok=True)
f=open(a.out,"w",encoding="utf-8")
json.dump(out,f,indent=2,ensure_ascii=False)
f.close()
print("Wrote:",a.out,"items:",len(out))
