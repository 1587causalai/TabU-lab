# V8.1 feature64 correction — 2026-10-11

Owner corrected feature vectors from32 to64 to preserve C64/8 and the complete V8 value interface. New opt-in mode is `learned_feature64_v1`; default inheritedV8 and explicit historical `learned_pool_v1`32D mode remain readable and unchanged. Only the10000x64 bank is new. All old V8 parameters are copied exactly by `from_v8`; optimizer migration is a separate runner responsibility.

Focused completion suite:57passed including CPU/MPS two-step learning, finite-difference gradient, baseline numeric/category draws, partial recoding, temporal binding, donor output exactly restored when feature entries equal old origins, strict storage reload. Ruff and diff checks pass.

Full suite first run from outer research directory:1925passed,1optional-pandas skip,2failures caused solely by missing relative manifest paths. The affected test module is rerun from this repository root and passes; see cwd-retry.log. No implementation failure remains. CUDA performance is a separate one-hour experiment; do not infer its result here.

Normalized Gaussian initial feature norm1 differs from C64/8 normsqrt(8), despite matching64D shape. Keeping every donor weight does not imply identical predictions when feature origins change. Historical32D poor results are retained and not relabeled64D.
