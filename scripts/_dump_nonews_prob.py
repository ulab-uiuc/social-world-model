"""Dump per-record no-news softmax prob (the null-routing signal) for a test file."""
import argparse, json, torch
from swm.data import Record
from swm.attributer import BasicPriorAttributer
from swm.dataset import build_attributer_news_prompt, build_attributer_no_news_prompt

ap=argparse.ArgumentParser()
ap.add_argument('--test-data-path',required=True)
ap.add_argument('--attributer-path',required=True)
ap.add_argument('--output-path',required=True)
ap.add_argument('--model-name',default='Qwen/Qwen3-8B')
ap.add_argument('--max-seq-length',type=int,default=1024)
a=ap.parse_args()

at=BasicPriorAttributer(model_name=a.model_name,max_seq_length=a.max_seq_length)
at.load(a.attributer_path); at.model.eval()
recs=[Record.from_dict(json.loads(l)) for l in open(a.test_data_path) if l.strip()]
out=[]
with torch.no_grad():
  for r in recs:
    news=r.news or []
    if not news:
        out.append({'market_id':r.market_id,'t':r.target.get('t'),'no_news_prob':1.0,'top_news_prob':0.0}); continue
    prompts=[build_attributer_news_prompt(r,r.target,n) for n in news[:at.max_news]]
    prompts.append(build_attributer_no_news_prompt(r,r.target))
    ids=[];ms=[]
    for p in prompts:
        e=at.tokenizer(p,padding='max_length',truncation=True,max_length=a.max_seq_length,return_tensors='pt')
        ids.append(e['input_ids']);ms.append(e['attention_mask'])
    ii=torch.cat(ids).to(at.model.llm.device);mm=torch.cat(ms).to(at.model.llm.device)
    logits=[]
    for i in range(0,ii.size(0),8):
        lo=at.model(input_ids=ii[i:i+8],attention_mask=mm[i:i+8])
        if lo.dim()==2 and lo.size(-1)==1: lo=lo.squeeze(-1)
        logits.append(lo.cpu())
    logits=torch.cat(logits)
    if getattr(at,'per_news_bce',False):
        # per-news Bernoulli: each news independent sigmoid; no-news emergent.
        p=torch.sigmoid(logits)[:len(news[:at.max_news])]
        no_news=float(1.0-p.clamp(min=0.0,max=1.0).max().item()) if p.numel() else 1.0
        top=float(p.max()) if p.numel() else 0.0
        out.append({'market_id':r.market_id,'t':r.target.get('t'),
                    'no_news_prob':no_news,'top_news_prob':top})
    else:
        q=torch.softmax(logits/at.target_temperature,dim=0)
        out.append({'market_id':r.market_id,'t':r.target.get('t'),
                    'no_news_prob':float(q[-1]),'top_news_prob':float(q[:-1].max())})
json.dump(out,open(a.output_path,'w'))
print(f"dumped {len(out)} -> {a.output_path}")
