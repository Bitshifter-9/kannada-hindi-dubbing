import argparse
import json
import os
import sys

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

p=argparse.ArgumentParser()
p.add_argument("--inp",required=True)
p.add_argument("--out",required=True)
p.add_argument("--model",required=True)
p.add_argument("--beams",type=int,default=4)
p.add_argument("--max_len",type=int,default=256)
p.add_argument("--redo",action="store_true")
a=p.parse_args()

try:
    import sentencepiece  # noqa: F401
except Exception:
    print("Missing dependency: sentencepiece")
    print("Fix:")
    print("  pip install sentencepiece sacremoses")
    sys.exit(1)

f=open(a.inp,"r",encoding="utf-8")
it=json.load(f)
f.close()

tok=None
try:
    tok=AutoTokenizer.from_pretrained(a.model)
except Exception:
    try:
        from transformers import MarianTokenizer
        tok=MarianTokenizer.from_pretrained(a.model)
    except Exception as e:
        print("Tokenizer load failed for model:",a.model)
        print("Try:")
        print("  pip install sentencepiece sacremoses")
        raise e
md=AutoModelForSeq2SeqLM.from_pretrained(a.model)
md.eval()

out=[]
td="data/interim/tr/txt"
if not os.path.exists(td):
    os.makedirs(td,exist_ok=True)

for s in it:
    i=int(s.get("id",0))
    sc=int(s.get("scene",0))
    st=float(s.get("start",0.0))
    en=float(s.get("end",0.0))
    dr=round(en-st,3)
    tx=(s.get("text") or "").strip()

    hi=""
    tp=os.path.join(td,"seg_"+str(i).zfill(4)+".txt")
    if (not a.redo) and os.path.exists(tp):
        f=open(tp,"r",encoding="utf-8",errors="ignore")
        hi=f.read().strip()
        f.close()
    elif tx:
        enc=tok(tx,return_tensors="pt",truncation=True)
        gen=md.generate(**enc,num_beams=a.beams,max_length=a.max_len)
        hi=tok.batch_decode(gen,skip_special_tokens=True)[0].strip()
        hi=hi.replace("आरक्षण","बुकिंग")
        hi=hi.replace("booking","बुकिंग")
        f=open(tp,"w",encoding="utf-8")
        f.write(hi)
        f.close()

    out.append({"id":i,"scene":sc,"start":st,"end":en,"dur":dr,"en":tx,"hi":hi})

od=os.path.dirname(a.out)
if od and not os.path.exists(od):
    os.makedirs(od,exist_ok=True)

f=open(a.out,"w",encoding="utf-8")
json.dump(out,f,indent=2,ensure_ascii=False)
f.close()

print("Wrote:",a.out,"items:",len(out))
