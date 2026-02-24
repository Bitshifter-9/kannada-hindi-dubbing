import argparse,json,os
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
p=argparse.ArgumentParser()
p.add_argument("--inp",required=True)
p.add_argument("--out",required=True)
p.add_argument("--model",required=True)
p.add_argument("--wps",type=float,default=2.6)
p.add_argument("--beams",type=int,default=4)
p.add_argument("--max",type=int,default=256)
a=p.parse_args()
f=open(a.inp,"r",encoding="utf-8");it=json.load(f);f.close()
tok=AutoTokenizer.from_pretrained(a.model)
md=AutoModelForSeq2SeqLM.from_pretrained(a.model)
md.eval()
out=[]
for s in it:
    i=int(s.get("id",0));sc=int(s.get("scene",0));st=float(s.get("start",0.0));en=float(s.get("end",0.0));dr=round(en-st,3)
    tx=(s.get("text") or "").strip()
    hi="";mw=int(dr*a.wps)
    if mw<1:mw=1
    if tx:
        cp="data/interim/tr/txt/seg_"+str(i).zfill(4)+".txt"
        if os.path.exists(cp):
            f=open(cp,"r",encoding="utf-8",errors="ignore");hi=f.read().strip();f.close()
        else:
            enc=tok(tx,return_tensors="pt",truncation=True)
            gen=md.generate(**enc,num_beams=a.beams,max_length=a.max)
            hi=tok.batch_decode(gen,skip_special_tokens=True)[0].strip()
            d=os.path.dirname(cp)
            if d and not os.path.exists(d):os.makedirs(d,exist_ok=True)
            f=open(cp,"w",encoding="utf-8");f.write(hi);f.close()
        w=hi.split()
        if len(w)>mw:hi=" ".join(w[:mw])
    out.append({"id":i,"scene":sc,"start":st,"end":en,"dur":dr,"en":tx,"hi":hi,"mw":mw})
d=os.path.dirname(a.out)
if d and not os.path.exists(d):os.makedirs(d,exist_ok=True)
f=open(a.out,"w",encoding="utf-8");json.dump(out,f,indent=2,ensure_ascii=False);f.close()
print("Wrote:",a.out,"items:",len(out))
