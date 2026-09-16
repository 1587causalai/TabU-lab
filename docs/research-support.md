# TabU-lab: research, evidence, and support

This brief is the shared narrative for research grants, compute contributions,
open-source maintenance support, and research partnerships. It is grounded in the
project's scientific question and public work. Individual applications adapt the
scope and budget to the supporter without changing the project's evidence claims.

Reviewed against repository materials on **2026-09-16**.

## The question and its value

Can a model learn relationships specific to each unit in a table by reconstructing
available evidence, then reuse those representations across tables?

TabU investigates this question through typed value encoding, contextual cell
representations, and Unit-based geometry that organizes relevant observations.
The current restoration implementation separates learning which observations to
refer to from estimating values using those observations. Whole-table
reconstruction supplies the learning objective; prediction is a downstream use.

The intended beneficiaries are researchers developing tabular models and
maintainers building reproducible tabular learning software. Useful transfer is a
research goal to test. In parallel, inspectable code, controlled evaluation
protocols, reproducible recipes, and well-documented failures can already serve
as reusable research materials.

## Why support this team and this work

TabU-lab has a public Apache-2.0 implementation and an active line of experiments,
with Heyang Gong / WeHub Research maintaining the project. The work combines a
specific modeling hypothesis with software for tracing how an experiment was
constructed and evaluated. This gives a supporter concrete work to inspect and
bounded follow-up questions to fund.

| Existing asset | Evidence to inspect | Interpretation |
| --- | --- | --- |
| Five-step table-restoration implementation | [Guide](tutorials/table-restoration.md), [dated independent review](reviews/restoration-five-step-20260915/README.md) | Implementation and bounded correctness checks |
| Prepared restoration execution | [Independent review and recorded checks](reviews/restoration-prepared-20260916/README.md) | Equivalence and timing for the stated small configuration |
| Historical TabU-TAR shared fitting | [120-table report, curves, and protocol](research/tar-shared-fit-20260907/README.md) | Fixed-training-table fitting; no transfer of its results to restoration |
| Research execution and evaluation software | [Program kernel](architecture/evolvable-pretraining-programs.md), [evaluation protocol](architecture/real-evaluation-default-protocol.md) | Reusable manifests, source identities, and controls |

The catalog currently records zero formal receipts and zero accepted capability
claims. Published local findings remain useful within their stated scope.
Review findings, failed checks, and negative controls remain part of that record.
Adoption and downstream use should be documented as they occur; no usage estimate
is inferred from commit counts or automated repository traffic.

## What the next resources would resolve

The following are candidate scopes for discussion. They do not launch experiments
or fix a new project-wide schedule.

| Work package | Question to resolve | Resources | Public deliverable and decision |
| --- | --- | --- | --- |
| Reproducible baseline | Which restoration choices learn reliably under a fixed protocol? | Research and engineering time; bounded compute | Baseline recipe, controls, per-table curves, failure analysis; decide which configuration merits further evaluation |
| Transfer evaluation | Does learned structure help on held-out rows, new worlds, and separate real tables? | Evaluation compute, permitted data, independent reproduction | Matched comparisons and coverage/uncertainty reports; distinguish fit from useful transfer |
| Controlled scaling | When do added data diversity, model capacity, or compute improve the relevant outcome? | GPU/cloud resources and systems work | Learning and resource curves plus documented limits; decide whether to scale, revise, or stop |
| Sustainable open software | Can another researcher reproduce and extend the work reliably? | Maintainer time, coding tools/API credits, external contributors | Regression fixes, documented interfaces, reproduction instructions, and reviewed release preparation |

Each scoped proposal should name its hypothesis, baseline, data/split, source
version, budget, stop condition, and output before compute is spent. A negative
result is valuable when it identifies a limitation and leaves reusable evidence.
Candidate artifacts such as model weights require separate verification and
licensing before publication.

## One resource-to-result argument

A support request should make this chain explicit:

**Important question → existing evidence → remaining uncertainty → bounded work
and resources → inspectable result → value for others.**

Research funding buys time to test a mechanism and interpret outcomes. Compute
support buys controlled experimental coverage. Coding-tool support buys
maintainable implementations, review, and reproducibility. A data partnership
makes a relevant evaluation possible under agreed access and sharing rules.
The scientific question and evidence standard stay consistent across these forms
of support.

For each application, specify:

- the work package and named maintainer/research roles;
- why the work matters to its intended users or research community;
- which current assets reduce execution risk;
- the resource quantity and budget rationale, based on an actual pilot where needed;
- milestones with verifiable deliverables and explicit stop/revise criteria;
- what code, recipes, measurements, or reports can be shared, and when.

Amounts, dates, funder eligibility, and negotiated terms belong to the individual
proposal. This brief does not promise a particular scientific result, exclusivity,
or access to private data.

## Collaboration

The project welcomes research grants and fellowships, GPU/cloud contributions,
open-source maintenance support, and data or evaluation collaborations.
[Open a public discussion via an issue](https://github.com/1587causalai/TabU-lab/issues/new)
to identify a shared question and a bounded scope; keep confidential details out
of that initial public message.

[Project entry](../README.md) ·
[Research website](https://research.wehub.us/tabu-lab/) ·
[License](../LICENSE)

## 展示与申请共用的中文口径

TabU-lab 研究如何通过恢复表格中的可用证据，学习每个 Unit 的特有规律，
并检验这些表征能否迁移到新表。我们已经公开参考实现、局部验证、历史实验
报告与可复现研究工具；跨表泛化仍是需要实验回答的问题。

不同形式的支持共同服务于同一条逻辑：
**重要问题 → 已有证据 → 尚待回答的问题 → 有边界的工作与资源 → 可检查的成果
→ 他人可复用的价值。**
科研经费支持机制研究与分析，算力支持受控实验，工具额度支持软件维护与复现，
数据和研究合作支持有意义的独立评估。具体预算、交付时间和共享范围按合作约定；
项目展示保留真实的进展，也明确每项结果能说明到哪里。
