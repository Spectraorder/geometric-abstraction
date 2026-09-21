"""Export only public demo assets; raw model sessions and local paths stay private."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

TRIALS = [('第 1 轮学习','round_001/trial_81'),('第 2 轮学习','round_002/trial_81'),
          ('第 2 轮确认','round_002/trial_82'),('第 2 轮确认','round_002/trial_83'),
          ('冻结后验证','heldout/trial_91'),('冻结后验证','heldout/trial_92'),('视觉关闭对照','audit/blackout')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('web/vision'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    for name in ['media','evidence']:(args.output/name).mkdir()
    manifest={'experiment':'vision_from_scratch_01','sensors':'RGB-D + calibration + proprioception',
              'prior_task_skills':False,'prior_task_policy':False,'supplied_robot_sdk':True,
              'policy_object_ground_truth':False,'online_stability_oracle':False,'trials':[]}
    cards=[]
    for label,relative in TRIALS:
        source=args.campaign/relative
        result=json.loads((source/'result.json').read_text())
        video=json.loads((source/'video.json').read_text())
        name=relative.replace('/','_');clip=f'media/{name}.mp4';poster=f'media/{name}.png'
        shutil.copyfile(source/'demo.mp4',args.output/clip)
        shutil.copyfile(sorted(source.glob('video_frame_*.png'))[-1],args.output/poster)
        entry={'label':label,**result,'video':video,'clip':clip,'poster':poster,
               'video_sha256':hashlib.sha256((args.output/clip).read_bytes()).hexdigest()}
        manifest['trials'].append(entry)
        evidence=f'evidence/{name}.json'
        (args.output/evidence).write_text(json.dumps(entry,ensure_ascii=False,indent=2)+'\n')
        status='PASS' if result['success'] else 'FAIL'
        cards.append(f'<article><div class="row"><h2>{label} · seed {result["seed"]}</h2><b class="{status}">{status}</b></div><p>策略执行 {result["policy_sim_seconds"]:.2f}s · RGB-D {result["camera_calls"]} 次 · 视频 {video["duration_seconds"]:.2f}s / {video["speed"]}×</p><video controls playsinline preload="none" poster="{poster}" src="{clip}"></video><p><a href="{evidence}">评估证据</a> · <a href="{clip}" download>下载完整录像</a></p></article>')
    for number in [1,2]:
        source=args.campaign/f'round_{number:03d}'
        for name in ['candidate.py','prompt.txt','response.json']:
            target=args.output/'evidence'/f'round_{number:03d}_{name.replace(".txt",".md")}'
            shutil.copyfile(source/name,target)
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (args.output/'index.html').write_text('''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SO-100 · 视觉学习三层塔</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{max-width:1100px;margin:48px auto;padding:0 24px;background:#101722;color:#edf3fa;font:16px/1.7 system-ui}header{max-width:870px}h1{font-size:clamp(26px,5vw,42px);line-height:1.25}h2{font-size:20px;margin:0}.sub{color:#a9bacb}.stats{display:flex;flex-wrap:wrap;gap:16px;margin:28px 0}.stats div{background:#203344;padding:14px 22px;border-radius:8px}.stats strong{font-size:27px;display:block}article{padding:20px;background:#1b2838;border:1px solid #304153;border-radius:12px;margin:26px 0}.row{display:flex;align-items:center;justify-content:space-between;gap:16px}.PASS{color:#7fe6b9}.FAIL{color:#ffae8d}video{width:100%;aspect-ratio:8/5;background:#000;border-radius:6px}a{color:#8de3cc}footer{margin:42px 0;color:#a9bacb}code{overflow-wrap:anywhere}@media(max-width:500px){body{padding:0 12px}article{padding:12px}}
</style></head><body><header><p class="sub">ASPIRE / SO-100 · MAC SIMULATION · 2026-09-21</p>
<h1>从空任务技能库，学会三层塔</h1><p>gpt-6-astra 根据 RGB-D 观测生成与修复控制策略。第一轮失败，第二轮通过三个种子；冻结代码后，再通过两个预留种子。失败与成功的完整物理轨迹全部保留。</p>
<p class="sub">输入：RGB-D + 相机标定 + 本体感知。策略不接收物体真值或在线稳定性判定。提供模型预训练知识和通用 IK SDK；“从零”指没有旧任务策略或技能库。</p></header>
<div class="stats"><div><strong>2</strong>学习轮次</div><div><strong>3 / 3</strong>第二轮确认</div><div><strong>2 / 2</strong>冻结后验证</div><div><strong>7</strong>完整执行录像</div></div>
<p><a href="manifest.json">完整结果清单</a> · <a href="evidence/round_001_candidate.py">首轮策略</a> · <a href="evidence/round_002_candidate.py">最终策略</a> · <a href="evidence/round_002_prompt.md">修复输入</a></p>
'''+''.join(cards)+'''<footer><p>录像均为实际 MuJoCo 物理状态回放，2 倍速；不是在线控制接口。腕部画面仅用于展示，控制器使用侧面 RGB-D。视觉关闭对照没有执行抓取。</p><p>限定场景：理想 RGB-D、已知标定与机器人模型、物体 XY ±4 mm 扰动、初始 yaw=0。前三组历史实验均使用 privileged state，本页仅展示新视觉实验。</p><p>公开页面只包含选定录像、候选代码、提示词和结果；完整本地轨迹保留在实验目录。</p></footer></body></html>''')
    print(args.output.resolve())


if __name__=='__main__':main()
