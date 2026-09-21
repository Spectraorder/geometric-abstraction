# 本机 IP、公开 HTTPS 和 macOS 后台运行

本次 [公网 Demo](https://jelsoft-boulder-hierarchy-weights.trycloudflare.com/) 已验证首页 200、视频 Range 206、完整 MP4 传输与源文件 SHA-256 一致。此临时网址依赖实验 Mac 在线，重启隧道可能改变。见 [部署验证记录](deployment-status.json)。

可视化随分支保存在 `web/vision/`，包括七段实际物理录像、结果清单、两轮候选策略与提示词。克隆本分支后，不需要重新训练即可展示。公开服务仅提供这些选定静态文件，不启动机械臂控制接口，不提供目录列表，也不发布原始模型会话、认证信息或整个实验目录。

## 立即本地运行

只展示页面需要 Python 3.10+，不需要安装 MuJoCo。执行仿真和学习另需运行指南中的 Python 3.14、uv 依赖和 Codex 登录。

```bash
python3 scripts/serve_vision_demo.py --bind 0.0.0.0 --port 8091
```

本机打开 `http://127.0.0.1:8091/`。同一局域网通过 `http://<这台 Mac 的局域网 IP>:8091/` 访问。Mac 可用 `ipconfig getifaddr en0` 查询 Wi-Fi IP；不同网络接口请使用对应名称。局域网地址不能直接被互联网上的设备访问。

服务支持 HTTP Range / 206，便于浏览器按需加载和拖动 MP4；文件系统访问限制在 `web/vision/`，拒绝指向目录外的路径和软链接。它是单机 Demo 的静态服务，不是多租户托管平台。

## 后台运行并创建公网链接

```bash
brew install cloudflared
python3 scripts/vision_demo_service.py start --port 8091
python3 scripts/vision_demo_service.py status
```

`start` 创建两个当前用户的 LaunchAgent：`com.aspire.vision.http` 和 `com.aspire.vision.tunnel`。服务退出后会由 launchd 重启，用户下次登录时启动；不需要保持终端窗口打开。Cloudflare 隧道将公网 HTTPS 请求转发到本机服务，不需要路由器端口映射或公网 IPv4。

`status` 输出 `last_public_url`，这是最新日志中的地址；请同时检查两个服务的 `running`。公网是否真实可用还应请求该地址验证，不能只凭进程存在判断。

```bash
curl -I https://<status 输出的子域名>.trycloudflare.com/
python3 scripts/vision_demo_service.py stop
```

`stop` 停止两个进程并删除其登录启动配置，不删除录像。再次 `start` 会创建新隧道。启动脚本使用绝对路径指向当前 checkout；不要在后台运行时删除或移动该工作区。移动后先 stop，再从新目录 start。

日志保存在 `~/Library/Application Support/AspireVision/`；LaunchAgent 配置保存在 `~/Library/LaunchAgents/`。这些文件只在本地生成，不提交 Git。如果建立隧道失败，检查 `tunnel.stderr.log` 和网络出口；脚本不会把“分配了域名”误报为隧道已连接。

**Mac 必须保持开机、联网和唤醒。** 退出用户会话、合盖睡眠或断网会影响公网访问。Quick Tunnel 为临时随机网址，隧道重启后可能改变，不提供固定域名或可用性 SLA。需要长期固定入口时，应配置自有域名的 named tunnel；本次没有购买域名或修改 DNS。参见 [Cloudflare 官方 Quick Tunnel 说明](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)。

## 重建公开展示包

新的原始实验必须具有本次目录结构（两轮、确认种子 81–83、留出种子 91–92、视觉关闭对照）。导出器只复制明确选定文件，输出目录必须不存在：

```bash
python3 scripts/export_vision_demo.py \
  --campaign outputs/vision_from_scratch_01 \
  --output outputs/vision_public_export
python3 scripts/serve_vision_demo.py --root outputs/vision_public_export --port 8092
```

每个公开结果记录策略 SHA-256、录像 SHA-256、传感器调用次数和实际策略时间。视频均为 2 倍速，在界面和视频中明确标注。已成功策略可用 `policies/stack_three_blocks_vision.py` 复跑，见 [视觉实验指南](VISION_FROM_SCRATCH.zh-CN.md)。

首次临时隧道在传输检查中出现超时，重连后完整约 2 MB 视频在 4.92 秒内传输成功。Codex 内置浏览器首次呈现画廊，但点击播放时页崩溃，因此该次 UI 播放检查未记为通过；页面保留单独下载入口。
