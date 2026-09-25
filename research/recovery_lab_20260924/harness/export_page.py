"""Whitelist public-to-local-gallery artifacts; never expose raw private traces."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];LAB=Path(__file__).resolve().parents[1];WEB=ROOT/'web/recovery';RUNS=ROOT/'outputs/recovery_lab_20260924'
for sub in ['media','evidence','docs']:(WEB/sub).mkdir(exist_ok=True)
rows=[]
for p in sorted(RUNS.iterdir()):
 if not p.is_dir() or not (p/'result.json').exists():continue
 result=json.loads((p/'result.json').read_text());calls=json.loads((p/'api_calls.json').read_text());fault=json.loads((p/'fault_events.private.json').read_text());r=dict(result,id=p.name)
 r['pilot']=('occlusion' in p.name and '_v2_' not in p.name) or p.name in {'persistent_occlusion_4001'}
 r['policy_label']='主线冻结策略' if r['candidate_sha256'].startswith('9d4484') else '当前助手修复 v2' if p.name.startswith('repair_v2') else '当前助手修复 v1'
 r['motion_calls']=sum(c['method'] in {'move_tool','move_joints','gripper'} for c in calls)
 r['start_sim_time']=calls[0]['time'];r['fault_events']=fault['events'];r['adapter_sha256']=fault['adapter_sha256'];r['runner_sha256']=fault['runner_sha256'];r['notes']=[c for c in calls if c['method']=='note']
 r['note']='开发/演示场景；未做独立预留验证。侧面面板保留实际策略输入，腕部画面用于回放。'
 if r['pilot']:r['note']='早期宽遮挡 pilot（包含背景），保留审计，不与精确目标遮挡合并统计。'
 elif p.name.startswith('reobserve_occlusion_v2') or p.name.startswith('persistent_occlusion_v2'):r['note']='反例：精确遮挡下，v1 色彩计数误认蓝色背景；候选停止，证据保留。'
 events=[{'time':r['start_sim_time'],'label':'开始观测'}]
 if fault['events']:
  event=fault['events'][0];labels={'sensor_rectangle':'注入图像遮挡','jaw_close_suppressed':'首次闭合失效','jaw_open_during_lift':'抬升中开爪'}
  events.append({'time':event['time'],'label':labels.get(event['event'],event['event'])})
 for phrase,label in [('Initial task colors re-observed','策略判定初始观测完整'),('Grasp not confirmed','视觉未确认抓起'),('blue retry','重新抓取'),('budget exhausted','观测预算耗尽，停止')]:
  match=next((c for c in r['notes'] if phrase in c['params']['text']),None)
  if match:
   if phrase=='Initial task colors re-observed' and not r['success']:label='误判：初始观测完整'
   events.append({'time':match['time'],'label':label})
 events.append({'time':r['start_sim_time']+r['policy_sim_seconds'],'label':'策略完成，交给独立判分' if r['success'] else '策略结束，任务未完成'})
 # Canonical parameters describe configuration, never inferred chronological learning.
 guard='none' if r['candidate_sha256'].startswith('9d4484') else 'rgbd-height' if r['candidate_sha256'].startswith('1fd870') else 'rgb'
 kind=fault['fault']; occ=kind in {'occlusion','persistent_occlusion'}
 params={'task':'tower3','observation':'rgbd-proprio','guard':guard,'initial_reobserve':0 if guard=='none' else 2,'wait_ms':0 if guard=='none' else 200,'min_pixels':None if guard=='none' else 8,'guard_height_mm':[4,150] if guard=='rgbd-height' else None,'fault': 'occlusion' if occ else kind,'roi':('broad-rgb' if r['pilot'] else 'tabletop') if occ else 'na','masked_side_calls':(999 if kind=='persistent_occlusion' else 2) if occ else 0,'seed':r['seed']}
 if kind=='miss':params.update(close_index=1,applied_angle_rad=.8)
 if kind=='slip':params.update(lift_trigger_mm=35,open_ms=450,applied_angle_rad=.8)
 r['parameters']=params
 r['experiment_id']='tower3__guard='+guard+'__reobs='+str(params['initial_reobserve'])+'__wait_ms='+str(params['wait_ms'])+'__fault='+params['fault']+'__roi='+params['roi']+'__frames='+str(params['masked_side_calls'])+'__seed='+str(r['seed'])
 r['configuration_sha256']=hashlib.sha256(json.dumps(params,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 r['policy_parent_sha256']=None if guard=='none' else ('d4939d8ac2bac5775cc1844f0398f91927faa7a39db07e1b266800f8a8e48429' if guard=='rgbd-height' else '9d4484224ddcfb8c00e8c930ca68471a3e10b3ebd067a7c9fcc587a62e765e4b')
 r['policy_label']={'none':'无初始可见性门控','rgb':'RGB 色彩门控 · 复观 2 次','rgbd-height':'RGB-D 高度门控 · 复观 2 次'}[guard]
 r['fault_label']=('宽区域 / pilot' if r['pilot'] else '目标区域')+(' · 前 999 次观测（持续）' if kind=='persistent_occlusion' else ' · 前 2 次观测') if occ else {'none':'无扰动','miss':'首次闭合 → 0.8 rad','slip':'抬升 35 mm → 开爪 450 ms'}[kind]
 r['events']=events;r['evidence']='evidence/'+p.name+'.json';r['video']='media/'+p.name+'.mp4';r['poster']='media/'+p.name+'.png';r['input']='media/'+p.name+'_input.png'
 if not (p/'demo.mp4').exists():raise RuntimeError('Missing video: '+p.name)
 shutil.copy2(p/'demo.mp4',WEB/r['video']);poster=sorted(p.glob('video_frame_*.png'))[-1];shutil.copy2(poster,WEB/r['poster']);shutil.copy2(p/'camera_000.png',WEB/r['input'])
 r['video_metadata']=json.loads((p/'video.json').read_text());r['video_sha256']=hashlib.sha256((p/'demo.mp4').read_bytes()).hexdigest()
 (WEB/r['evidence']).write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n');rows.append(r)
EVENT_LABELS={'开始观测': 'Initial observation', '注入图像遮挡': 'Image occlusion injected', '首次闭合失效': 'First gripper close suppressed', '抬升中开爪': 'Gripper opened during lift', '策略判定初始观测完整': 'Policy judged the observation complete', '误判：初始观测完整': 'False positive: observation judged complete', '视觉未确认抓起': 'Grasp not visually confirmed', '重新抓取': 'Retry grasp', '观测预算耗尽，停止': 'Observation budget exhausted; stop', '策略完成，交给独立判分': 'Policy ended; private evaluation', '策略结束，任务未完成': 'Policy stopped; task incomplete'}
for row in rows:
 for event in row["events"]:event["label"]=EVENT_LABELS.get(event["label"],event["label"])
manifest={'date':'2026-09-24','base_commit':'3a2a48a','source':'actual local MuJoCo trials','contract_tests':14,'naming_schema':'tower-recovery-parameters/1','runs':rows}
(WEB/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
# English web documentation is synchronized by scripts/export_dual_results.py.
print('Exported',len(rows),'trials')
