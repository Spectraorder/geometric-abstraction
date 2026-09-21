# 可视化：固定公网镜像、Mac 本地 IP 与后台隧道

已部署的 [固定公网 Demo](https://aspire-so100-vision.vercel.app/) 是 Vercel 静态镜像，不依赖 Mac 在线。Mac 上同时运行本地 IP 服务及 SSH/443 公网隧道。三个入口展示同一份 `web/vision/`：七段实际物理录像、两轮候选/提示词、结果与哈希。

验证记录见 [deployment-status.json](deployment-status.json)。固定镜像和 Mac 隧道都通过首页 200、视频分段请求 206、完整 MP4 哈希核对；固定镜像已在浏览器实际播放，观察到 18.24 秒的播放进度，总时长 37.6 秒。

## 本地 IP

只展示页面需要 Python 3.10+，不用安装 MuJoCo：

```bash
python3 scripts/serve_vision_demo.py --bind 0.0.0.0 --port 8091
```

本机地址 `http://127.0.0.1:8091/`，局域网地址 `http://<Mac 局域网 IP>:8091/`。Wi-Fi IP 可用 `ipconfig getifaddr en0` 查询。局域网 IP 本身不能被互联网直接访问，需要下节隧道。

服务只提供选定静态文件，不开放仿真控制接口。HTTP/1.1 + Range 支持视频拖动播放；拒绝越界路径、指向目录外的软链接、隐藏目录/文件和目录列表。完整原始实验目录、模型会话与凭据不会发布。

## Mac 后台服务与公网隧道

本机已安装并运行两个当前用户的 LaunchAgent：`com.aspire.vision.http`、`com.aspire.vision.tunnel`。关闭终端后继续运行，进程退出后重启，用户下次登录时启动。

```bash
python3 scripts/vision_demo_service.py start --provider pinggy --port 8091
python3 scripts/vision_demo_service.py status
python3 scripts/vision_demo_service.py stop
```

当前 Mac 隧道：[HTTPS 入口](https://fgsjk-185-220-238-74.run.pinggy-free.link/)。**Pinggy 免费连接每次有效 60 分钟**，到期后台服务会重连并获得新网址；重启也会换网址。查询最新 `last_public_url`，不要把本段网址视为固定入口。分享长期链接请使用页首 Vercel 镜像。

Mac 需要开机、联网且保持唤醒；合盖睡眠、断网或退出用户会话会影响 Mac 入口，但不影响静态镜像。SSH 仅转发展示端口，不提供远程 shell 或 SSH agent 转发。脚本生成专用本地密钥并使用独立 known_hosts；这些文件保留在本机，未提交 Git。

日志与服务状态：`~/Library/Application Support/AspireVision/`。登录启动配置：`~/Library/LaunchAgents/`。`stop` 停止进程并删除本服务的启动配置，不删除录像。工作区路径写入本机 LaunchAgent，运行期间不要删除或移动它；移动时先 stop，再从新路径 start。

Cloudflare 也可选，但本机网络实测多次发生连接超时，因此本次没有将它作为交付入口：

```bash
brew install cloudflared
python3 scripts/vision_demo_service.py start --provider cloudflare --port 8091
```

Cloudflare Quick Tunnel 同样为随机临时地址，没有 SLA。两种服务都没有购买域名或付费隧道套餐。服务 `running` 和日志中的 URL 只是运行状态，实际可访问性需请求 URL 验证。

官方说明：[Pinggy 免费连接时限](https://pinggy.io/help/) · [Cloudflare Quick Tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)。

## 固定公网镜像的更新

项目：`lucas-projects-a9e8b0a3/aspire-so100-vision`。只部署静态展示文件，没有在线机械臂控制服务。

```bash
cd web/vision
vercel link --yes --project aspire-so100-vision --scope lucas-projects-a9e8b0a3
vercel deploy --prod --yes --scope lucas-projects-a9e8b0a3
```

需要相应 Vercel 登录权限。`.vercel/` 和 `.env*` 已忽略，不要提交凭据。本次通过 CLI 发布；Vercel 与上游 GitHub 仓库的自动关联未成功，因此推送/合并 PR 不会自动重新发布镜像，需要运行上述部署命令。

## 重新导出展示包

原始实验目录需包含本次两轮、种子 81–83、预留 91–92 和视觉关闭对照。导出器只复制白名单资产，输出目录必须不存在：

```bash
python3 scripts/export_vision_demo.py \
  --campaign outputs/vision_from_scratch_01 \
  --output outputs/vision_public_export
python3 scripts/serve_vision_demo.py --root outputs/vision_public_export --port 8092
```

每段视频有策略 SHA-256、录像 SHA-256、相机调用次数和实际策略时间，画面明确标注 2 倍速。冻结策略的重跑和从空技能库学习见 [视觉实验指南](VISION_FROM_SCRATCH.zh-CN.md)。
