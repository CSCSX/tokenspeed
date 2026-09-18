# Kimi-K3 Buffered Replay：统一 Decode 实现计划

评审入口请见 [Replay-SSM 设计](kda-replay-ssm-design.md)：汇总 cache/scheduler
职责、kernel 接口、当前 PR 状态与验收差距。本文保留实现和实验历史；涉及旧版
computed frontier 的记录，应以该设计引用的上游 #1597 修复为准。

状态：实现中，buffered replay 尚未通过完整模型验收，默认不启用。已实现 LCM-owned history、
固定地址 metadata、统一 decode/accepted commit、跨 rank 失败检查，以及
accepted/quiescent endpoint 物化。M14 接通 mixed batch 中的 buffered decode
部分，复用同一个 forward/commit；prefill 仍要求 cache owner 提供精确输入状态。
M15 接入实验性容量参数与 kernel 注册；真实 TP8 的首轮 L8/L16 测试都未通过
性能不退步的目标，输出也未与基线逐 token 一致。M16 已验证 decode checkpoint
延迟发布修复，并完成新旧实现的短 NSYS 对照。M17 验证 L16 的静态 kernel
tile 调整，并完成同条件 AIME：baseline 26/30，L16 官方 28/30、完整最终答案
27/30。匹配的端到端延迟在 C1/C4 分别增加 1.49%/13.69%，性能验收仍未通过；
新一组 GPU 上，M18/M19 已完成 L32/L64 与多次 baseline 重启对照。M19 的
静态 kernel tile 调优通过数值和回归测试，但相对前后两次 baseline，L32 的
C1/C4 延迟仍增加 1.84–1.92% / 10.02–10.36%，L64 增加
0.68–0.76% / 9.07–9.40%。按用户指示恢复后，M20 的跨层 flush 隔离实验
通过 40 个数值用例，但原有 endpoint writer 的 mixed C4 批次慢了
10.39–21.57%。随后定位到 persistent grid 的 request-row 负载不均；调整静态
stride 后，40 个对照用例保持 bitwise 一致，mixed C4 的独立 endpoint 延迟
降低 43.81–72.42%。此改动仅优化现有 writer，不把容量 flush 移到其他执行位置；
源码 `ad28ea43` 已通过数值和 runtime 回归，以及真实 TP8 工作负载的输出一致性
检查。但相对前后两次原始 baseline，C1/C4 延迟仍增加 0.99–1.11% /
10.82–11.03%；相对 M19 L64 本次观测也稍慢，性能验收未通过。容量 flush
跨层合并仍未接入，不能把 kernel 收益等同于模型收益。M21 已完成同条件 C4
NSYS 对照：L64 的 B4 graph 中位数增加 5.44–5.61%，并比原实现多执行
12 轮 B2 decode；相同输出预算下，两条较慢请求的 acceptance 从 3.64 降至
3.11。每层 history recurrence 明显慢于原 verify，而通常的 graph 间隔没有
变大。下一步优先降低 history reconstruction 成本，并追踪数值差异对
acceptance 的影响；不能仅凭较长的 CPU validation 区间推断它是主要瓶颈。
这些是带 profiler 的诊断结果，不替代 M20 的无 profiler 性能验收。
M22 已完成隔离的 TF32x3 history reconstruction 实验：tile sweep 的 198 个
数值用例通过，但扩大到全部可达 history 长度后，488 个用例中有 9 个未通过
与当前实现的 BF16 output 对照容差；两者均通过独立 FP32 reference，容差未放宽。
静态候选通过 109 个 kernel/reference 测试、201 个 runtime 测试和 77 个 subtest，
但短 history 仍有性能退步，未接入生产代码，也没有新的完整模型或 AIME 结果。
下一步结合实际 history 分布和同输入下的数值差异继续定位，不以局部最优 tile
推断模型收益。
M23 已完成真实 NVFP4 TP8 的 history/same-input 诊断，CUDA graph 和 overlap
保持开启。C4 每批有 70 轮 B4 和 13 轮 B2，其中 61 轮 B4 的 history 长度
不一致；所有 request/group 的位置、flush 和物化转移检查通过。10 条 continuation
输出与 M20 L64 一致，18 个单层快照通过未放宽的独立 reference。原 accepted
replay 的 FP32 conv/gate 与 buffered 的 BF16 producer 是这些快照中 state 差异
的主要来源；这不证明完整模型 acceptance 差异的因果关系。诊断含额外同步和
快照，不用于性能验收，也没有新 AIME 分数。下一步以真实 mixed history 和
跨层工作集验证 kernel 候选，单独追踪 producer 精度，不能直接推广均匀 history
的测试收益或把局部数值接近当成完整模型正确性。
M24 已用实际 cache recipe 的 strides/packing 完成 cache-hint 隔离实验，覆盖
单层热缓存和 69 层轮换工作集。全部 reference 与 bitwise 对照通过；但在
70 种实际 C4 history/alignment 模式上，checkpoint streaming 的加权 recurrence
收益只有 0.43%，不构成完整模型性能达标，也未合入。下一步直接分析并优化
重建 kernel 的计算/调度成本，保留现有精度、状态协议和完整模型验收门槛。
M25 的新 NSYS 再次复现 matched-prefill 场景的退步，同时保留未复现的 capture。
M26 的显式 register layout／FP32 scalar dot 候选在 70 种实际 C4 metadata
模式上降低 recurrence 微基准耗时 12.79%，并通过 reference 和 bitwise 对照；
部分热缓存控制用例仍有退步，不能视为 E2E 收益。M27 修正最小 history tile
的编译限制，并让 forward 与 endpoint 共用重建 helper；隔离的 109 项 kernel
测试、201 项 runtime 测试和 77 个 subtest 通过。候选已接入 rebase 后的工作区，
冻结的接入版本随后通过直接 GPU 回归：109 项 kernel 测试、201 项 runtime
测试和 527 项共享 cache/scheduler 测试；匹配的 scheduler 构建通过 490 项 C++
和 153 项 Python binding 测试。独立 endpoint 微基准仍有 L64 长 history
退步，不能据此宣称整体加速。首次完整模型启动因旧 FlashMLA 缺少新 API
而失败，未产生性能样本；按当前依赖版本建立隔离环境后，两节点模型注册检查
通过。首轮新实现完成 75 条 E2E 计时请求，无错误或抢占；30 个计时批次的
输出与 M20 L64 一致。同 GPU 的原始 baseline 随后完成：C1 请求延迟基本
持平（-0.09%），C4 请求／整个批次延迟仍增加 8.48%／13.51%，三轮均有
C4 退步。C4 acceptance 从 3.67 降为 3.405；请求统计推算的较慢分组轮数
从 70 增至 82，不将其当作直接测量的 GPU round 数。下一步优先分离验证输出
与 accepted history 的精度问题；独立重启、本版本 AIME 和性能验收仍未完成。
M28 的隔离实验通过 18 个单层快照与 90 个 accepted endpoint 检查。M29 将
BF16 验证输出与 FP32 history producer 合并到一次 recurrence，共享 history
重建，不增加持久 state 或 accepted replay。18 个快照的 140 个接受组合及
40 个 T1/T4 多轮数值用例通过，容差不变；接入代码和 scratch 预算已更新。
接入回归随后发现并修正 T1 的 GEMM 可选参数错误；修正版通过 157 项 kernel、
201 项 runtime 和 527 项共享 cache/scheduler 测试。新一轮原版／新版 E2E
对照已启动。首轮新版完成 75 条计时请求，无错误或抢占；C1/C4 延迟中位数为
1461.59/1791.41 ms，acceptance 为 2.58/3.23，未恢复历史观测的 acceptance。
同 GPU 原版首轮随后完成：新版 C1/C4 请求延迟增加 38.00%/11.51%，
整个批次延迟增加 38.00%/10.98%。随后完成两次独立重启／实现，共 300 条
计时请求，所有配对的 C1/C4 请求延迟仍增加 37.81–38.37%/11.22–12.30%，
未通过性能门槛。随后完成的 C4 NSYS 中，新版 prefill 分组由 1+1+2 变成 4，
acceptance 回到 3.64，未复现原有 decode 退步；B4 graph 中位数反而降低
2.07–2.18%，但 recurrent kernel 从约 8.06 us 增至 17.86 us。此结果完整保留，
不能替代重复的无 profiler 性能验收。固定 C1 NSYS 对照已启动，用于检查稳定
复现的较大退步并避开多请求 prefill 分组。该 C1 对照随后完成并复现退步：
新旧 capture 分别匹配各自全部 30 条无 profiler C1 输出，acceptance 为
3.59/2.58；实际目标 graph 数为 72/100。新版单轮中位数增加 3.67–3.92%，
decode 区间增加 44.16–46.16%。rank 0 的 415.917 ms 增量中，按原版均值
核算的额外轮数项为 358.125 ms；这不是因果分摊，但说明不能只依靠 kernel
微基准改善来推导整体达标。下一步保留固定容差与状态协议，验证短 history
重建候选并追踪 acceptance 差异；AIME 尚待验证。局部 state 误差降低不能
直接推导为模型性能或精度改善。M30 短 history 候选通过数值与实际 endpoint
writer 检查，但较长的短 history 和混合请求出现 kernel 退步，未接入。
按用户最新要求，下一阶段固定 L64/C1，优先定位 acceptance 差异，暂不运行
容量 sweep。先对照冻结原版、当前未启用 buffered replay、当前 L64 三组结果，
区分 rebase 与 buffered 路径的影响，再沿相同 token 前缀寻找最早的 state、
verify output 或 target/draft logits 差异。不同生成轨迹下的后续张量不能当作
同输入对照；数学等价也不能代替浮点执行一致性的验证。
三组对照现已完成：原版与当前未启用 buffered replay 的代码，各 4 次输出逐
token 一致，acceptance length/rate 为 3.59/0.8638；当前 L64 的 4 次输出也
保持一致，但为 2.58/0.5253，首次生成 token 差异位于 index 11。该 case 的
差异因此收敛到启用 buffered replay 后的路径。M32 的同源 on/off 张量对照
复现上述输出与 AR：prefill 的已采集字段在 8 个 rank、两次运行中全部一致，
首次 verify 在 history 为空、输入与初始 state 相同时已出现 KDA 输出差异。
Q 缩放位置和 FMA/reduction 顺序参与了分歧，局部调整尚未做到全部逐位一致。
M33 隔离实验保留新 history/state 管理，仅复用原 verify 输出：1,035 个
fixture/graph 检查通过，完整模型四次运行的 acceptance length/rate 回到
3.70/0.8986；新的未启用 buffered 控制仍复现 3.59/0.8638。但诊断组与控制组
的输出仍从 index 11 起不同，因此 AR 回升不等于结果一致。另有 1,104 个
真实 layer fixture 确认 PDL 开关在该对照中不改变 verify 数值。M34 的 552 个
同输入 layer fixture 中，原 replay 均逐位复现保存的下一轮 state；新实现的
history 生成阶段和重建阶段各自引入差异，state 最大绝对差异分别为
9.54e-7 与 1.91e-6，均在既定容差内，但不是逐位一致。M35 进一步确认，
同一组原 replay K/U/decay 按原顺序逐 token 做 FMA，可在全部 552 个样本
上逐位复现原 state；交给当前批量重建算法则产生最大 2.86e-6 的差异。
因此 verify 算术、系数生成与 state 重建顺序需要分别控制，不能只对齐
producer 就宣称消除 state 差异。下一步以同输入验证约束算术候选，再跑
完整模型的 AR 对照。M36 的单 kernel 候选把 verify 与 history 更新分成两个
内部循环，共用一次重建，不再需要 M33 的额外 verify launch 与 state scratch。
552 个空 history、2,208 个非空 history 样本的 verify 输出均逐位匹配原版，
graph 与无效行检查通过；四种短 history 的局部耗时约持平至快 4%，不代表
完整模型收益。M37 已将候选扩展到现有 T1/T4 与各输入类型；修正小 history
分支的 layout 转换后，161 项 kernel、201 项 runtime 和 77 个 subtest 通过。
M38 在新 GPU allocation 上准备同一 L64/C1 完整模型对照，先用保存的真实
tensor 验证最终 wrapper，再比较未启用 buffered replay 与候选版本的 AR。
M38 随后完成：2,760 个真实 tensor 用例通过；完整模型候选的 acceptance
length/rate 为 3.11/70.33%，比 M29 的 2.58/52.53% 改善，但仍低于
同 GPU 控制组的 3.59/86.38%。每组四次输出一致，两组首次差异仍在
generated index 11；三次重复的请求延迟中位数仍增加 17.58%。因此对齐
verify 尚未修复 AR，也未通过性能门槛，候选不接入正式代码。下一步继续
固定 L64/C1，单独检查候选的 history 系数生成与 state 重建，不扩展容量 sweep。
M39 已在当前候选上完成该检查：552 个样本的 verify 输出逐位匹配原版，
但候选系数按原顺序逐 token FMA 后的 state 仍有最大 9.54e-7 差异；
原版系数交给当前重建算法仍有最大 2.86e-6 差异。下一步分别隔离 FP32
conv 累加、gate 归约与 key normalization，再验证重建顺序；不能把局部
容差通过当作 AR 已恢复，也不能将慢速诊断 replay 直接当作正式实现。
M40–M43 继续固定同一组 552 个真实输入，逐项隔离 conv 顺序、gate、
normalization 的 reduction/FMA、state update 的 FMA，以及 SiLU 与 projection
相减的融合边界。对齐后，K/U/decay 与按原顺序重建的 endpoint 均逐位匹配
原 replay，verify 输出也保持一致。但这仍是首个 T4 window 的隔离诊断：
gate 当时使用原版输出，重建使用 serial reference。M44 后续通过原 descriptor
gate kernel 的单层独立计算，在全部 552 个样本保持系数与 endpoint 逐位一致。
M45 的分页、批量读取／逐 token 更新重建也通过这批真实样本及 260 个边界
用例，包含 CUDA graph；局部重建成本约持平到增加 10%，并非性能达标。
M46 的直接指针 gate 已去掉临时 CPU descriptor，独立计算仍在全部 552 个
样本上逐位匹配。M47 已在私有候选中接入实际 producer/重建路径，补入
预分配 history scratch 与显存预算；最终修正版通过 163 项 kernel、201 项
runtime 测试及 77 项 subtest（3 项原有 skip）。M48 的 552 个首窗口样本
与 M49 的 2,760 个连续窗口层级检查均精确匹配原版，包含 CUDA graph；
多轮测试只在起点载入一次 state，之后由候选自行维护 history 和 conv state。
前两轮回归发现并修复了注册、prepared 分支以及测试地址清单的接入遗漏，
没有放宽精度门槛。M50 的真实完整模型 L64/C1 对比已完成：修正版与原版
四次生成均逐 token 一致，AR 同为 0.8638、acceptance length 同为 3.59，
在这个固定场景下消除了此前的 AR 差异。但总延迟中位数仍增加 9.01%
（1140.08 vs. 1045.84 ms），性能验收没有通过，候选尚未接入正式代码。
M51 的同条件 timeline 已完成：双方仍逐 token 一致、AR 相同，且每个 rank 都是
72 轮目标 decode。新版单轮 graph 中位数增加 9.78–10.09%，主要可见开销是
逐层 history reconstruction 与独立 history gate。完整 profile 区间还包含一次
约 79 ms 的迟到 rank／allreduce 等待，不能将其等同于稳定 kernel 开销，也不能
替代 M50 的无 profiler 结果。M52 随后完成 L8/16/32/64、8 个 rank 的真实输入
连续状态检查，覆盖容量 flush 与 state-page 边界：26,496 个层级 verify 对照
和 24,288 个 accepted endpoint 对照均逐位一致，graph/eager 一致且未重置参考
state。M53 已按原有协议启动原版／L8／L16 的六次独立启动对比；首轮原版
完成全部 75 条计时请求，C1/C4 总延迟中位数为 1045.39/1580.175 ms。
首轮 L8 随后完成：C1 的 15 个批次仍逐 token 一致，但 C4 的 15 个批次
输出均不同，AR 中位数从 0.8898 降到 0.7985，整个批次延迟增加 13.61%。
这还不是六次启动的最终结论，也不能把包含 acceptance 差异的延迟归因于 kernel。
两边均关闭 mixed batch；下一项 AR 排查固定 L8/C4，补入同版本关闭 buffering
的控制，区分上游改动与 replay，并比较相同输入和 state 下的多请求数值计算。
不将 C1 的恢复推广到 C4。随后 L16 第一轮因部分请求复用到 51840 而非预定的
51328 cached tokens，被工作量一致性检查中止；所有样本保留，不补跑替换，
六次启动对比未完成。结束服务并确认节点空闲后，M55 开始固定 L8/B4 的同输入
数值诊断，比较 producer、verify、accepted state 与下一轮 flush。它使用真实
C1 张量构造 B4 并重新计算原版 B4 对照，不等同于完整模型 AR 验证。
M55 随后完成全部 1656 个层级用例：producer、accepted state 与 flush state
均逐位一致，但首次 verify 的 40,697,856 个元素中有 2448 个不同，替换成
原版 producer 后差异仍在。问题在这些样本中收敛到 B4 verify recurrence，
尚未证明其解释完整模型 AR。M56 确认 B4 单 warp 的 normalization 线程布局
不同；M57 只对齐这个布局后，差异元素从 2448 减少到 877，但仍未逐位一致，
不接入正式代码。producer、accepted/flush state 与 graph/eager 仍一致。
下一步固定同一 L8/B4，继续对照剩余归约/FMA 指令顺序，再验证完整模型 AR。
M58 随后确认原版编译结果在 projection、state update 和 output dot 中采用
不同的乘加融合位置。M59 的隔离诊断显式复现这些舍入位置后，全部 1656 个
L8/B4 层级用例均逐位一致，包括 accepted state、下一轮 verify/flush 和
graph/eager。它定位了这批同输入差异，但尚未证明完整模型 AR 已恢复；其中
依赖编译器结果的 value-row 特判不作为正式实现接入。M60 已完成 C1 连续状态
及共享回归：C1 与 163 项 kernel 测试通过；runtime 的两项失败随后在未修改
代码上复现，原因是测试继承双节点启动并选择了不安全的默认端口。恢复原有
单节点测试方式后，201 项 runtime 测试与 77 项 subtest 通过（3 项原有 skip）。
M61 已完成同一份源码的 buffering off、L8 on 与算术对齐诊断版对照，固定 C4、
真实 NVFP4 TP8 和 CUDA graph，不扩展容量性能 sweep。首次尝试在产生 C4 样本前
被 gateway 拒绝：该接口不支持批量 input IDs。失败记录保留；revision 2 恢复
runbook 的四个并发单请求，重新核对源码和进程归属后复用已加载的控制组，
从清空 cache、重新 prime 开始。三组各四批请求的 prefill 都是 1+1+2，cache
工作量一致；未对齐的 L8 稳定复现 AR 差异，对齐 verify 算术后，四批的全部
输出 token、AR 与 acceptance length 组合恢复到控制组。控制组和诊断版每批
各有两条请求 AR 为 0.881、两条为 0.8986；未对齐 L8 为 0.716 / 0.881。
结合同输入 state/verify 对照，这定位了该固定场景的 verify 浮点执行差异，
不是 replay state 协议错误的证据。八个 worker 的实际诊断 kernel 调用已核对。
下一步须把依赖编译器的特判换成可维护的正式实现，再通过同输入与端到端回归；
诊断版尚未接入，不能将这次 AR 恢复等同于性能验收或其他配置已通过。
M62–M64 随后各完成同一套 1656 个 L8/B4 用例，尝试去掉编译结果特判。
改用 Triton 自动布局使 verify 与 accepted state 的一致性更差；把 verify
循环改成固定 T，或仅把 verify 输入存储改为 BF16，则所有逐层指标与 M57
完全相同，仍有 877 个首次 verify 元素不同。三种方案均不接入，也不据此
重跑完整模型或宣称性能收益。Gluon 与 Triton 使用相同的 sum/reduce 定义，
两边也都启用乘加融合；后续应检查编译后的算术与向量分组，而不是假定存在
不同的归约默认开关。不得为消除差异而悄悄修改原版对照或放宽验收条件。
M65 的离线编译对照进一步确认，原版按 value row 不同的乘加融合依赖 SLP
向量化；关闭另外两项向量算术优化没有同样效果。原版和候选的未改动 PTX
均精确复现。该检查没有运行 GPU，也不是数值或性能通过：关闭 SLP 同样会
改变原版算术，不能以同时改动两边替代对冻结原版的验证。下一项有界实验是
在重建与 verify 之间加入保持数值的寄存器边界，先检查编译输出，再决定是否
进入既有数值与 AR 验证；尚无该候选结果。
M66 随后完成这个寄存器边界实验：未改动的两组控制 PTX 均精确复现，但加入
边界后仍未恢复原版 output dot 的 FMA，候选在离线编译阶段被拒绝，没有新
GPU 数值、AR 或性能结果。按已说明的取舍，下一步先请用户确认是否允许为
unbuffered/buffered 定义共享且明确的 verify 浮点运算顺序。若确认，验证须
保留冻结原版、共享算术的 unbuffered、共享算术的 buffered 三组；不能用改变
后的 unbuffered 取代原始性能/精度对照，也不能声称输出逐位恢复原版。该方向
尚未获确认或实现，现有验收门槛不作隐式变更。
用户随后指定先测试现有算术对齐诊断版，并将范围限制在 KDA、不跑 E2E。
M67 已完成 L8/B4/T4、69 层、TP8 每 rank 形状、CUDA graph 开启的局部对照。
八个 rank 的 1656 个 layer/case 与新 benchmark 的输出和 accepted state 均
逐元素通过。对齐版 KDA core 比未对齐版耗时低 5.82–5.92%，但仍比冻结原版
高 36.67–37.10%。计时包含 producers、verify/history 与 accepted commit，
平均一个无 history 窗口和一个 flush 窗口；不代表稳态 L8 分布或端到端性能。
因此对齐本身在此处没有性能惩罚，但 buffered 路径尚未达到不慢于原版的目标。
本次没有改写共享算术契约或采用诊断版，也没有新的 AR/AIME 测试。
M68 随后完成相同条件下的 KDA 单元测试 NSYS 对照。四份报告的预期 NVTX
区段与全部 kernel 均已核对，四次 GPU 测试逐元素通过。直接 kernel 计时显示
对齐版 recurrence 较慢，且每层 history-gate 合计约 238–242 µs／69 层窗口，
高于原版跨层 batched gate 的 12.2–12.4 µs；省掉约 113 µs accepted replay
仍不足以抵消。此局部差异不是由更大的 kernel 间 bubble 导致。NSYS 数字是
带 profiler 的诊断，不能替代 M67 的无 profiler 结果或端到端验收。
M54 的 history-gate token tile 实验暂缓，仅完成 CPU 准备，尚无 GPU 结果。
不得因局部 kernel 变快而放宽逐位对照、AR 或重复端到端性能门槛。
不得将诊断版的重复计算与临时 scratch 当成生产实现或性能达标。所有观测
同步、诊断拷贝与重复计算均不计入性能收益，当前版本 AIME 尚未验证。
M69 用户已确认采用共享 verify 算术：unbuffered 与 buffered 使用明确的归约与
FMA 规则，同时保留冻结旧版作独立对照。这允许新算术与旧编译结果有最后几位
差异，不等于放宽 replay 一致性或完整模型精度/性能门槛。正式代码已接入，
187 项 kernel 测试、240 项 runtime 测试与 117 项 subtest 通过（3 项既有 skip）。
8 个 rank 的 1656 个真实张量 layer/case 中，新 unbuffered/buffered 逐位一致，
包含 CUDA graph、accepted state 与下一窗口 flush。相对冻结旧版，verify 有
约 0.0124% 输出元素不同，最大绝对差 0.00048828125；这不是 AR/AIME 验证。
不采用 M59 的 value-row 特判，不新增用户开关。
M70 在 `5fc562dd` 上优化共享算术的执行方式：warp 内保留相同归约树、逐 token
交错 verify/history recurrence，并合并 conv/gate producer launch。小 row 数的
history gate 保留原 scalar kernel，避免改变 FP32 舍入。scratch 预算不增加。
最终代码通过 197 项 kernel 测试、240 项 runtime 测试与 117 项 subtest（3 项既有
skip），以及 8 个 rank 的全部 1656 个真实张量 layer/case。开启 CUDA graph 的
L8/B4/T4 KDA 局部对照已达到冻结原版耗时水平；详细重复测量见 M70 记录。
该测试使用保存的真实 NVFP4 激活，包含两个窗口的 producers、recurrence 和
accepted commit，不包含完整模型，旧 producer 也未模拟 runtime StreamFork
重叠。因此下一步仍需固定 L8/C4 的完整模型 token/AR、AIME 和 E2E 性能验证，
不能以局部结果替代这些门槛。
PD/任意 live endpoint 交接仍受限，尚未选择默认容量。各阶段 commit、环境和验证证据见
[implementation record](kda-buffered-replay-progress.md)。下文保留完整方案与验收要求。

## 1. 目标与范围

将 Kimi-K3 的 KDA decode 改为跨轮缓存历史输入、按需物化 recurrent state。
Standard decode 与 speculative decode 共用一套计算、metadata 和提交协议：
standard decode 是本轮输入宽度 `T=1` 的情况，speculative verify 是 `T>1`。

首阶段采用 **state-and-output route + 按需 flush**：在 kernel 内重建 state
并计算 output，但仅在需要 flush 或对外交付精确 state 时写回完整 state。
启动参数决定 history buffer 的容量。

本阶段不实现 output-only route，不扩展 GDN/Qwen3.5，不改变模型权重格式、
采样算法或 speculative acceptance 规则。共享基础设施的改动必须维持其他模型行为。
允许针对 `T=1` 做 kernel 编译特化，但不维护两套 Python 执行路径。

参考： [ReplaySSM: Cache SSM Inputs, Not State](https://dao-lab.ai/blog/2026/replayssm/)。
文章的性能数字不能直接作为 Kimi-K3 的预期收益；KDA 的 per-channel decay
需要单独推导和验证。

## 2. 当前实现与目标的区别

当前 KDA 普通 decode 每步更新并写回 state。Speculative verify 在 replay
启用时保存本轮 packed QKV、gate 相关输入与 beta，不保存每个 draft 的完整
state；acceptance 确定后重放 accepted tokens，并立即提交 recurrent/conv state。
现有 replay payload 是本轮工作区，不是跨轮 history buffer。

| 项目 | 当前 KDA | 本计划 |
| --- | --- | --- |
| 普通 decode | 每步写回 state | `T=1` 使用统一 buffered decode |
| Speculative commit | 重放 accepted prefix，写回 state | 提交候选 history entries，推进 GPU 指针 |
| 历史输入 | 仅保留本轮 replay payload | 保留 checkpoint 之后的 accepted history |
| State 写回 | 每步或每轮 | 容量触发 flush，或精确 state 交付边界 |
| Output 计算 | 显式计算 recurrent state | 首阶段仍显式重建 state，不增加 output-only |

`store_states=False` 只控制中间 state 是否写回，不等于 output-only。
State-and-output 与 flush 是两个独立维度：kernel 可以重建 state 用来计算
output，然后不写回它。Output-only 则是另一种代数计算方式，避免构建更新后的
完整 state，本阶段不实现。

## 3. 统一状态协议

每个 request 的每个 KDA layer 由以下内容共同表示当前状态：

- 已物化的 recurrent checkpoint `S_c` 及其绝对 token 位置 `c`。
- 从 `c` 到当前 accepted endpoint 的 history entries。
- 本轮尚未提交的 candidate entries。
- GPU metadata：ring 起点、已提交长度、本轮有效宽度和候选位置。
- 与绝对 token 位置一致的 conv history/窗口信息。

核心不变量：

> 当前逻辑 state = checkpoint + checkpoint 之后已提交的 history。

Rejected candidates 不属于逻辑 state。物理 checkpoint 可以落后于 accepted
endpoint，但任何消费者都不能把它误认成 endpoint 的完整 state。
请求位置 metadata 应尽量跨层共享；每层保存各自的 history 数据。

### 3.1 KDA history 存什么

先建立 FP32 reference，再确定 kernel 的缓存布局。优先验证缓存 normalized K、
delta correction U 与 decay 信息的方案；Q 用于本轮输出，不要求长期保存。
KDA 的 decay 按 key channel 变化，不能直接套用 Mamba-2 的标量 decay 公式。

Reference 需要证明：从 checkpoint 与历史重建得到的 state，和逐 token KDA
recurrence 一致；当前候选的 correction 只依赖已提交历史与同轮更早候选。
因此丢弃 rejected suffix 后，accepted prefix 的 history 仍然有效。

Conv 有独立的短窗口依赖，不能只用 recurrent history 替代。设计时明确保存
必要的卷积前输入或等价的 accepted window，并与 recurrent history 使用一致的
accepted endpoint。允许提交小型 conv window，但不保存每个 draft 的完整
recurrent state。最终字段、dtype 和精度策略在 reference 验证后固定。

### 3.2 Buffer 所有权

遵循 [cache-concepts.md](cache-concepts.md)：优先让 LCM cache group 管理
history storage 的分配、容量、回收及迁移，backend 只消费存储与 metadata。
不得先引入独立的 backend 私有长期 buffer，再补 request 生命周期。

先验证现有 sliding-history group 能否保护所有未被 checkpoint 覆盖的 entries。
保留范围必须覆盖历史和未提交窗口；若现有规则会提前回收，先扩展通用 cache
contract。Checkpoint 保持 state-family 语义，不给它虚构 row/page geometry。
Ring 是 kernel 的逻辑访问布局，不要求绕过 LCM 自建连续存储池。

## 4. 一套 Forward 与 Commit 路径

统一流程如下：

1. Refresh 固定地址 metadata，读取本轮有效宽度 `T`。
2. 从 checkpoint 与 committed history 重建本轮起点 state。
3. 按每个 request 的容量判断是否 flush 已提交历史。
4. 计算本轮 outputs 与 candidate history entries。
5. 获取本轮实际接受的输入 token 数 `a`。
6. 用同一个 GPU commit 操作推进 committed endpoint，丢弃 rejected suffix。

普通 decode 的有效行使用 `T=1, a=1`；speculative decode 的 `a` 来自验证。
Padding/idle 行不得推进状态。必须在调用边界统一 accepted-count 的定义，
不能把 draft matches 和实际推进 state 的输入 token 数混用，也不能盲目加一。

本轮 candidate 写入在下一轮读取前完成；commit 与后续 forward 必须有明确的
stream 顺序。普通 decode 可在无需等待 acceptance 时安排提交，但不得复制一套
计算或 metadata 实现。

共享 kernel 可以根据静态 `T` 选择优化配置。多 token correction 的并行计算是
后续 kernel 优化点，不能直接假定文章 GDN 的 triangular solve 已适用于 KDA。
第一版先以统一、可验证的 recurrence 保证语义，再减少串行计算成本。

## 5. Flush 与容量参数

新增启动参数，建议名称：

```bash
--ssm-replay-buffer-capacity 64
```

`64` 仅是配置示例，不是性能推荐值。容量 `L` 以每个 request、每层的 token
entries 计，包含 committed history 与本轮 candidates 占用的空间。
启动时固定，完成内存规划与分配后再捕获 CUDA graph；不支持运行中调整。

采用提前一个窗口 flush 的初始策略：已提交 history 长度为 `h`，服务配置的
最大执行窗口为 `T_max`，当 `h + 2 * T_max > L` 时 flush 旧的 committed history。
使用最大窗口可保证后续窗口不因剩余容量而缩短。启动校验 `L >= 2 * T_max`，
并单独检查 kernel 支持的容量/对齐约束；若需要额外物理 padding，日志分别报告
逻辑容量与实际分配。Standard decode 同样使用该规则，`T_max=1`。

Flush 只汇总本轮开始前已经接受的历史，不能把尚未验证的 candidates 写入
checkpoint。完成后通过 ring 指针推进回收历史空间，不搬移剩余 candidates。
所有判断使用 GPU 上的每请求 metadata，允许同一 batch 混合 flush/no-flush。

参数不绑定 `prefix_granularity`、state block granularity 或 KV kernel page size。
首阶段不定义 `0` 为关闭模式；默认容量在基准验证后决定。容量不足或硬件/kernel
不支持时启动报错，不静默退回另一套 decode 路径。

实验性接入阶段必须显式传入正容量；省略参数维持现有部署行为，并非用 `0`
切换实现。目前注册限定 NVIDIA Blackwell、BF16 模型输入、BF16/FP32 conv producer、
FP32 state、128 维
head、`T_max=1/4`、`2*T_max <= L <= 64`，并拒绝 PD。这个范围是初始验证
边界，不是性能推荐；扩大范围需要补充对应验证。

显存预算按最终布局计算：

```text
history bytes ≈ live requests × local KDA layers × L × bytes per entry
```

另外计入 checkpoint、conv、metadata、scratch、allocator packing 和 overlap
所需保护空间。容量增大可减少 flush，但也增加每步 history 读取/重建成本，
因此不保证越大越快。

## 6. 生命周期与 CUDA Graph

遵循 [unified_path.md](unified_path.md) 与 [scheduler.md](scheduler.md)：

- **Prefill → decode**：以 prefill final state 建立 checkpoint，history 清空。
- **Decode → incremental prefill**：prefill 输入必须是精确 checkpoint，
  不直接读取 replay history。当前 agentic 下一轮是新 request，通过 prefix
  cache 匹配已物化边界；FSM 没有 live `Decoding → Prefilling` 转换。
  若后续增加直接消费 live accepted endpoint 的接口，必须先物化该 endpoint，
  确认完成后才能重塑或回收 history，不能在 prefill metadata 阶段补做。
- **Prefix cache**：只发布确实物化且位置匹配的 checkpoint。Flush 不自动赋予
  prefix 边界 provenance，不能发布候选 state 或落后的 checkpoint。
- **Retraction/恢复/迁移**：交付 checkpoint + history + metadata 的完整表示，
  或在交付边界统一物化 endpoint；首阶段优先采用后者，明确同步完成后再释放。
  当前 retraction 的交付边界是可复用 prefix checkpoint，不是任意 live endpoint：
  沿用原有 best-effort L2 store 和尾部重算，history 不参与传输或 prefix 匹配。
  不增加重算范围，也不新增丢弃 live state 后冒充精确 endpoint 的恢复方式。
  任意 endpoint 迁移仍待集成，PD 配置继续明确拒绝。
- **结束/取消/槽位复用**：不需要交付 state 时直接回收，不做无用 flush；在
  in-flight readers 完成前不能复用 buffer。Prefix 发布另按其精确位置处理。
- **Pool rebind**：清除旧 descriptors/views，重建存储绑定并按现有协议重新捕获。

Eager 与 CUDA graph 使用同一个 `refresh_decode_metadata` 和 GPU 操作序列。
Buffer 按最大运行 batch 容量分配，不以 graph capture ladder 上限代替运行容量。
避免每步 `.item()`、CPU flush 判断和临时 tensor 分配。Flush 标志、ring 指针、
accepted count 为设备数据，不因不同请求的决定重新捕获 graph。
Commit 如独立于模型 forward 捕获，也必须保持同一语义、固定 buffer 与明确依赖。

## 7. 工作拆分与交付顺序

| 阶段 | 主要工作 | 进入下一阶段的条件 |
| --- | --- | --- |
| A：语义与 reference | KDA 重建公式、缓存字段、conv 协议、acceptance 单位 | 多轮重建、拒绝、flush 与逐 token reference 一致 |
| B：Cache contract | LCM ownership、retention、容量参数、GPU metadata、生命周期 | 未 flush 数据不会被回收，内存预算与实际分配吻合 |
| C：统一 kernel/runtime | `T=1/T>1` 共享计算与 commit；按需 flush 替代每轮 eager replay | 两种 decode 均正确，非 flush 轮没有完整 recurrent state 写回 |
| D：系统集成 | CUDA graph、overlap、agentic prefill、prefix reuse、恢复路径 | Eager/graph 和混合 batch 边界测试通过 |
| E：性能与精度 | Real NVFP4 TP8 full model 对比、容量 sweep、NSYS、数据集验证 | 形成可复现报告后决定默认容量与是否启用 |

主要代码范围：

- `runtime/layers/attention/backends/state/kda.py`：统一 buffered decode 与 commit。
- `runtime/layers/attention/backends/state/mamba.py`：共享调用协议和 metadata 接口，
  不改变 GDN 的默认行为。
- `runtime/layers/attention/kv_cache/recipes/kimi_k3.py` 与 cache/LCM bridge：history
  group 声明、预算和状态生命周期；具体 C++ 变更由阶段 B 的缺口决定。
- Runtime 配置和 decode runner：参数校验、固定 buffer、acceptance 与 commit 顺序。
- `tokenspeed-kernel` 的 KDA ops 与 kernel：重建、候选计算、flush、GPU 指针提交。

上述 runtime 路径均相对于 `python/tokenspeed/`。Runtime 通过
`tokenspeed-kernel` 边界调用 kernel，不新增直接第三方依赖。
实现完成后更新相应 design 文档、模型 recipe 和 runbook；本计划不替代已有设计规范。

## 8. 正确性验收

以参数化测试覆盖组合，不为每个微小分支新增独立测试：

- `T=1` 与多 token 窗口，各种接受长度和 rejected suffix。
- 最小合法容量、较大容量、临界 flush、连续多次 flush、ring wraparound。
- 同 batch 的不同 history 长度与 flush 决策，padding/idle、请求重排与槽位复用。
- Prefill → decode → incremental prefill、prefix hit、取消、恢复、pool rebind。
- Eager/CUDA graph、overlap 开关、实际 batch 超过 capture ladder。

比较逐步 outputs、物化 endpoint state、conv window，以及长序列误差累积。
用 reference 先确定 dtype 对应容差并记录理由，不能为让新实现通过而临时放宽。
数学等价不代表浮点 bitwise 一致，也不要求随机采样文本逐字相同。

端到端使用真实 NVFP4 权重、TP8、完整 Kimi-K3；记录目标与 draft 模型版本、
代码 commit、kernel backend、软件环境、CUDA graph 配置和 sampling 参数。
固定 AIME 2026 题集版本、prompt、seed、生成预算与评分器，旧/新实现同条件运行，
报告分数及逐题结果。小型 agentic smoke test 只能验证执行，不能替代精度验证。
共享接口改变后运行相关 GDN/普通 decode 回归测试。

## 9. 性能验收与报告

Baseline 使用冻结的当前代码：standard decode 为逐步 state 更新，speculative
decode 为当前 eager replay-and-commit。两者分别和新实现对比，不能用不同
acceptance 或不同生成 token 数直接推断 kernel 加速。

EAGLE3 是必须通过的验收场景：真实 NVFP4、TP8、完整模型，使用原有
四 token verify 配置与相同 CUDA graph/overlap 设置。新 buffered 路径必须
实际执行，且性能不得慢于冻结的原实现。比较范围包含 verify、accepted-prefix
commit 和 agentic 边界物化；只比较 prepared recurrence，或保持旧路径启用，
都不能满足这个门槛。重复测量并报告波动，结果不明确时继续验证，不视为通过。

固定 GPU、模型、输入、batch、后端、warmup 和 graph 设置；在持久分配的计算
节点上按 runbook 执行。覆盖 concurrency 1 与较大 batch，并测试多个合法容量。
分别测无 profiler 的重复端到端运行和短 NSYS，报告：

- 每步 decode/verify/commit 延迟，flush 与 no-flush 的延迟分布。
- 总 tokens/s、inter-token latency、峰值显存和最大可用 batch。
- 实际 acceptance、flush 频率、完整 state store 次数、kernel launch 数。
- Agentic incremental prefill 的边界物化开销。

提供包含 prefill、多个 decode/verify round，以及 flush/no-flush 的清晰命名
`.nsys-rep` 和 ZIP，附可复现命令。不同容量必须记录历史重建开销，不能只看
flush 次数下降。正式比较前约定可接受的性能波动与回归阈值；结果不足时不宣称
达到论文收益，不默认启用未经验证的配置。

## 10. 实施前必须关闭的问题

1. KDA 的缓存字段、dtype 与长历史重建误差是否满足精度要求？
2. LCM 能否完整表达 checkpoint 落后于 endpoint 时的 history retention？
3. Conv 与 recurrent history 如何保证拒绝、flush、迁移后的同位置一致性？
4. Endpoint 物化与 prefix publication 如何遵守现有 provenance 和 overlap 顺序？
5. 哪些容量与窗口由首版 kernel 支持，哪一档容量在 TP8 实测最合适？

这些问题属于阶段 A/B 的交付，不应通过引入 standard/speculative 两套实现绕过。
