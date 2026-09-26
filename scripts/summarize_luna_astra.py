"""Add a readable, measured summary after the complete frozen campaign."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'docs/evaluations/luna_astra_results.json').read_text())
if not r['complete']:raise SystemExit('Refuse to summarize an incomplete campaign as complete')
groups=r['groups'];runs=r['runs'];tower=[x for x in runs if x['endpoint']=='tower_completion'];stop=[x for x in runs if x['endpoint']=='correct_abstention']
passed=sum(x['endpoint_pass'] for x in runs);calls=sum(x['model_calls'] for x in runs);errors=sum(x['model_errors'] for x in runs)
def label(g):
    return ('Luna '+g['fast_effort']+' / Astra '+g['slow_effort']) if g['mode']=='luna_hybrid' else ('Luna only '+g['fast_effort'] if g['mode']=='luna_only' else 'Astra only '+g['fast_effort'])
block=['## Measured pilot results', '',f'All **{len(runs)} scheduled attempts** are retained. **{sum(x["endpoint_pass"] for x in tower)}/{len(tower)}** tower-completion checks and **{sum(x["endpoint_pass"] for x in stop)}/{len(stop)}** correct-stop checks passed. There were **{calls} model calls**, **{errors} failed provider calls**, and **{sum(x["guard_interventions"] for x in runs)} guard interventions**. Those intervention counts matter even when the final endpoint passed.', '', '| Configuration | Tower | Correct stop | Total model wait | Input tokens | Slow repairs | Rule agreement |','|---|---:|---:|---:|---:|---:|---:|']
for g in groups:
    block.append(f'| {label(g)} | {g["tower_passed"]}/{g["tower_n"]} | {g["abstention_passed"]}/{g["abstention_n"]} | {g["total_model_wait_seconds"]:.2f} s | {g["input_tokens"]:,} | {g["repair_calls"]} | {g["rule_agreement"]}/{g["decisions"]} |')
block+=['','Each row covers the same three cases. Raw proposal errors can be hidden by successful guards; endpoint success is not proof that every model decision was correct. All call traces remain inspectable on the public page.','', 'The measured call totals do not establish dollar-cost efficiency. They include startup and subscription-client overhead, and the cases are too few for a stable effort ranking. There is no demonstrated online recovery during manipulation in this experiment.','']
def insert(path,title,block):
    text=path.read_text();start='<!-- measured-results:start -->';end='<!-- measured-results:end -->'
    if start in text:text=text[:text.index(start)]+text[text.index(end)+len(end):]
    index=text.index(title)
    path.write_text(text[:index]+start+'\n'+'\n'.join(block)+'\n'+end+'\n\n'+text[index:])
insert(ROOT/'docs/LUNA_ASTRA_README.md','## What is compared?',block)
zg=[g for g in groups if g['mode']=='luna_hybrid' and g['slow_effort']=='high']
zh=['## 此次实测结果','',f'57 次正式尝试全部保留：搭塔检查 **{sum(x["endpoint_pass"] for x in tower)}/{len(tower)}**，正确停止检查 **{sum(x["endpoint_pass"] for x in stop)}/{len(stop)}**。共 **{calls} 次模型调用**，其中 **{errors} 次服务调用失败**；执行检查发生 **{sum(x["guard_interventions"] for x in runs)} 次干预**。','', '你指定的 Astra high 慢模型这一列：','', '| 快模型档位 | 搭塔 | 正确停止 | 三次尝试累计模型等待 | 输入 token |','|---|---:|---:|---:|---:|']
for g in zg:zh.append(f'| Luna {g["fast_effort"]} | {g["tower_passed"]}/{g["tower_n"]} | {g["abstention_passed"]}/{g["abstention_n"]} | {g["total_model_wait_seconds"]:.2f} 秒 | {g["input_tokens"]:,} |')
zh+=['','完整的 19 个配置对照表见英文说明和网页。最终通过不代表模型每一步都正确：检查器可能拦截错误建议，或转交 Astra 后停止。请结合原始建议、实际动作和慢模型修订次数一起看。','', '这些结果只能说明这三个案例中的表现；不能宣称某一推理档位普遍更好、更快或更省钱，也还没有证明操作过程中的在线自纠正。','']
insert(ROOT/'docs/LUNA_ASTRA_README.zh-CN.md','## 一共有多少组',zh)
print('Updated both readable guides from all 57 measured attempts')
