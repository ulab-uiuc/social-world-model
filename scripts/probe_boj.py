"""Probe: fix the BoJ event (history+question), feed CUSTOM news, get 8B μ_i per news."""
import json, torch
from swm.forecaster import MultiEventForecaster
from swm.dataset import MultiEventForecasterDataset, _pack_prompts
from swm.utils.utils import load_records
MID="KXCBDECISIONJAPAN-25DEC18-H2140"
# === 自定义新闻：(标题, 正文) — 改这里 ===
CUSTOM=[
 ("Japan core inflation accelerates to 3.8% in November, fastest since 1991",
  "Japan's core consumer price index rose 3.8% year-on-year in November, the fastest pace in over three decades, intensifying debate over how quickly the central bank must act to contain price pressures. Economists noted services inflation broadened well beyond food and energy."),
 ("Japan's largest union federation secures record 6.2% wage increase for 2026",
  "Rengo, Japan's biggest labor group, announced its member unions won an average 6.2% pay raise in early negotiations, the highest in over 30 years. Policymakers have long said durable wage growth is the precondition for normalizing monetary policy."),
 ("Yen slumps past 162 to the dollar, importers warn of cost crisis",
  "The yen weakened beyond 162 per dollar, a fresh multi-decade low, prompting warnings from importers and retailers about surging costs. Ministry officials repeated that excessive, one-sided currency moves are undesirable and that authorities are watching markets closely."),
 ("Japan's economy unexpectedly contracts 0.9% in Q3 as consumption slumps",
  "Japan's GDP shrank an annualized 0.9% in the third quarter, far worse than forecast, as households cut spending. Analysts said the data complicates any plan to tighten policy and raises the risk of the central bank staying on hold."),
 ("Global oil prices crash 30% on demand fears, easing inflation worldwide",
  "Brent crude tumbled roughly 30% over the past month amid weakening global demand, a development that typically cools headline inflation and reduces the urgency for central banks to raise interest rates."),
 ("Toyota and Sony post record profits as weak yen flatters exporters",
  "Japan's largest exporters reported record quarterly earnings, helped by a historically weak yen that inflates overseas revenue when converted back home. Some board members welcomed the currency tailwind."),
]
recs=load_records("data/social-world-model-v6-qwen3.5-397B-clean-semdedup/test_kalshi_final.jsonl")
rec=None
for r in recs:
    if r.market_id==MID and abs((r.target['p']-r.history[-1]['p'])-0.116)<0.01: rec=r;break
bp=rec.history[-1]['p']
print(f"Event: {rec.question}")
print(f"before={bp:.3f}  (真实 target={rec.target['p']:.3f}, Δ={rec.target['p']-bp:+.3f})\n")
fc=MultiEventForecaster(model_name="Qwen/Qwen3-8B",max_seq_length=1024,max_news=30,predict_delta=True)
fc.load("saves_local/fc8b_v9odds_semdedup/final-model");fc.model.eval()
ds=MultiEventForecasterDataset(records=[],tokenizer=fc.tokenizer,max_news=30,max_seq_length=1024,predict_delta=True)
news=[{"title":t,"description":d} for t,d in CUSTOM]
prompts=[ds._build_prompt(rec,rec.target,nw) for nw in news]
pk=_pack_prompts(fc.tokenizer,prompts,1024)
with torch.no_grad():
    mu=fc.model(input_ids=pk["input_ids"].to(fc.model.llm.device),attention_mask=pk["attention_mask"].to(fc.model.llm.device)).view(-1).cpu().tolist()
print(f"{'μ_i':>9}  新闻 (forecaster 单独读这条 → Δ 预测)")
for m,(t,_) in zip(mu,CUSTOM):
    print(f"{m:>+9.4f}  {t[:72]}")
