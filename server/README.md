# go-poker 接入指南

把训练好的 Deep CFR 模型（`models/standard/phase2/checkpoint_iter_1000.pt`）接入
[go-poker](https://github.com/evanofslack/go-poker) 的机器人座位。

## 架构

```
go-poker (Go)                        本仓库 (Python)
┌─────────────────┐   POST /v1/act   ┌──────────────────────────┐
│ server/bot.go   │ ───────────────▶ │ server/inference_server  │
│  decideBotAction│   中性状态 JSON   │  encode_state (复用训练) │
│  (AI 优先/回退)  │ ◀─────────────── │  strategy_net 前向推理    │
└─────────────────┘  botAction 语义   │  build_raise_action 钳制 │
                                     └──────────────────────────┘
```

- **Go 侧**（`backend/server/ai_client.go`）：把 `GameView` 翻译成中性 JSON 状态，
  解析响应并映射回 `botAction{kind, amount}`。`bot.go` 的决策入口改为
  `decideBotAction`：AI 可用走模型，出错自动回退启发式。
- **Python 侧**（`server/inference_server.py`）：FastAPI 服务。内部用 duck-typed
  伪 state **原样复用**训练期的 `encode_state` / 加注钳制逻辑，保证特征一致性；
  `create_agent_for_checkpoint` 自动选择普通或 OM agent，OM 请求会先重放对手动作历史。

## 启动

Python 侧（本仓库根目录，`poker-ai` 环境）：

```bash
python -m server.inference_server \
    --checkpoint models/standard/phase2/checkpoint_iter_1000.pt \
    --host 127.0.0.1 --port 8001
```

Go 侧（go-poker 目录）：

```bash
AI_INFERENCE_URL=http://127.0.0.1:8001 ./go-poker
```

不设置 `AI_INFERENCE_URL` 时行为与原来完全一致（启发式 bot）。

## 状态映射（go-poker → pokers）

| 概念 | go-poker | pokers（训练约定） |
|---|---|---|
| 花色 | `CardSuit` 位掩码 C/D/H/S | 0/1/2/3 |
| 点数 | `CardRank` 0..12 | 0..12（相同） |
| 街道 | `GameStage` 2..6（PreFlop..Showdown） | stage − 2 → 0..4 |
| 本轮下注 | `player.Bet` | `bet_chips` |
| 已收前轮 | `TotalBet − Bet` | `pot_chips` |
| 剩余筹码 | `player.Stack` | `stake` |
| 底池 | `Σ TotalBet`（含离场玩家） | `pot` |
| 需跟到额 | `max(Bet)` | `min_bet` |
| 最小加注 | `view.MinRaise` | 最小加注增量（伪 state 的 `bb`） |
| 加注合法 | `(!Called \|\| callAmount==0) && Stack > callAmount` | `Raise ∈ legal_actions` |

响应动作用 **go-poker 语义**：`raise` 的 `amount = call_amount + 加注额`，
即 `Bet(pn, amount)` 一次调用即可；引擎自身仍做最终合法性校验。

OM checkpoint 的请求还可携带当前手牌中各对手的动作序列：

```json
{
  "opponent_histories": [
    {
      "opponent_id": 1,
      "actions": [
        {"action_id": 3, "context": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
      ]
    }
  ]
}
```

`action_id` 为 0=fold、1=check/call、2=half-pot、3=pot、4=overbet；
`context` 必须恰好 25 维。历史只在单次请求内使用，不会跨房间残留。

## 注意事项与限制

1. **6 人桌**：网络输入是 156 维（6 槽玩家特征）。少于 6 座时空位补零
   （`active=false`），多于 6 座自动回退启发式。
2. **归一化基准**：特征归一化沿用训练期的 `players[0].stake`（0 号位剩余筹码）。
   建议 0 号位尽量有真人/bot 就座，空位时特征会轻微退化但不报错。
3. **采样**：默认按策略分布采样（`sample: true`），与训练评估行为一致；
   调试可传 `sample: false` 走贪心 argmax。
4. **延迟**：CPU 单次推理约 1–5 ms，Go 侧超时 3 s，远小于 bot 的思考延迟。
5. 旧 checkpoint（3 动作，如 `phase1`）会被服务的兼容性校验拒绝，
   请使用 `phase2` 及以后的 5 动作模型。
