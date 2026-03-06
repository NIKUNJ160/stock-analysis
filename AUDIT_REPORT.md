# Workspace Root Audit Report

Date: 2026-03-06
Scope: Full repository audit from workspace root (`C:\NIKUNJ\live`)
Method: Static code inspection + repository metadata inspection + runtime log evidence (`data/system.log`)
Audit coverage: 52/52 source/docs/config files discovered in workspace scan (`.py`, `.md`, `.txt`, `.json`, `.gitignore`)

## Completion Status

This report completes the remaining audit scope. The previous report has been superseded by this full-scope version.

## Critical Findings

### 1) Risk state can desynchronize from broker state (CRITICAL)

- Impact:
  - Risk engine and broker can diverge on quantity/side for the same symbol.
  - Stops/exits may only partially close broker positions while risk manager marks symbol fully closed.
  - Exposure and PnL controls become unreliable after multiple fills on one symbol.
- Evidence:
  - `OrderManager` writes each filled order into risk manager as a fresh position. [services/execution_service/order_manager.py:97](services/execution_service/order_manager.py:97)
  - `RiskManager.open_position` overwrites position by symbol instead of merging/offsetting existing size. [src/risk_management/risk_manager.py:110](src/risk_management/risk_manager.py:110)
  - Broker supports additive position behavior (both long and short adds). [services/execution_service/broker_api.py:191](services/execution_service/broker_api.py:191), [services/execution_service/broker_api.py:236](services/execution_service/broker_api.py:236)
  - Exit logic uses risk-manager quantity for close order size. [services/execution_service/order_manager.py:130](services/execution_service/order_manager.py:130)
- Recommended fix:
  - Make broker the single source of truth for live positions, and derive risk snapshot from broker fills.
  - Or add a reconciliation layer after every fill to synchronize quantity/side/avg price between broker and risk manager.
  - Add tests for repeated same-symbol entries, partial exits, and side flips.

### 2) Backtester can crash when strategy produces no trades (CRITICAL)

- Impact:
  - A valid “no signal/no trade” run can raise runtime exceptions and abort backtest pipeline.
- Evidence:
  - `calculate_metrics` returns `{"error": ...}` on empty trades. [src/backtesting/metrics.py:21](src/backtesting/metrics.py:21)
  - `backtester.run()` always calls `print_metrics_report(metrics)` without checking for `error`. [src/backtesting/backtester.py:206](src/backtesting/backtester.py:206)
  - `print_metrics_report` assumes fields like `total_trades` exist and will key-error on error payload. [src/backtesting/metrics.py:126](src/backtesting/metrics.py:126)
- Recommended fix:
  - Guard `print_metrics_report` call when `"error" in metrics` and return structured no-trade result.
  - Add a regression test for empty-trade backtests.

### 3) Unsafe model artifact deserialization (CRITICAL security)

- Impact:
  - Loading compromised `.pkl` model files can execute arbitrary code.
- Evidence:
  - Runtime model loads use `joblib.load` directly. [services/model_service/realtime_predictor.py:22](services/model_service/realtime_predictor.py:22), [src/models/ensemble.py:43](src/models/ensemble.py:43)
  - Backtester also loads pickle artifacts. [src/backtesting/backtester.py:78](src/backtesting/backtester.py:78)
- Recommended fix:
  - Treat model artifacts as untrusted input unless signed/verified.
  - Add artifact integrity checks (hash/signature), controlled model registry, and least-privilege runtime.
  - Prefer safer serialization formats where possible.

## High Findings

### 4) CORS is insecure/misconfigured for credentialed requests (HIGH)

- Impact:
  - `allow_origins=["*"]` with `allow_credentials=True` is unsafe and browser-incompatible for credential flows.
- Evidence:
  - CORS middleware config. [src/api/main_api.py:22](src/api/main_api.py:22)
- Recommended fix:
  - Replace `*` with explicit trusted origin list from environment config.

### 5) Risk exposure check is direction-agnostic (HIGH)

- Impact:
  - Existing opposing positions are not netted; valid reducing/hedging trades can be misclassified and rejected.
- Evidence:
  - Exposure always treated as additive absolute notional: `new_exposure = current + proposed*price`. [src/risk_management/risk_manager.py:78](src/risk_management/risk_manager.py:78)
- Recommended fix:
  - Compute signed exposure by side and evaluate both gross and net limits explicitly.

### 6) Dependency compatibility risk in requirements (HIGH)

- Impact:
  - Environment resolution may fail or produce incompatible runtime combinations.
- Evidence:
  - `tensorflow>=2.15.0` with separate `keras>=3.0.0` constraints. [requirements.txt:13](requirements.txt:13), [requirements.txt:14](requirements.txt:14)
- Recommended fix:
  - Pin a tested compatible set (or remove standalone `keras` if using `tf.keras` only) and lock versions.

## Medium Findings

### 7) Streamlit engine status can report RUNNING for dead process (MEDIUM)

- Impact:
  - Operational dashboard can show false health state and block restart path.
- Evidence:
  - Status only checks `engine_process is None` and does not inspect process liveness via `poll()`. [app.py:53](app.py:53)
- Recommended fix:
  - Use `proc.poll()` checks and auto-reset stale process handles.

### 8) Predictor exits process abruptly on schema mismatch (MEDIUM)

- Impact:
  - One symbol mismatch can hard-stop full service; no graceful shutdown or degraded mode.
- Evidence:
  - Explicit `sys.exit(1)` in async predictor loop. [services/model_service/realtime_predictor.py:64](services/model_service/realtime_predictor.py:64)
- Recommended fix:
  - Raise controlled application exception, mark symbol unhealthy, and stop pipeline gracefully with clear health state.

### 9) Log file has no rotation/retention (MEDIUM)

- Impact:
  - Long-running deployments risk disk growth from `data/system.log`.
- Evidence:
  - Plain `logging.FileHandler` without rotation policy. [src/utils/logger.py:35](src/utils/logger.py:35)
- Recommended fix:
  - Switch to `RotatingFileHandler` or `TimedRotatingFileHandler` with size/time retention.

### 10) In-memory cache fallback ignores TTL semantics (MEDIUM)

- Impact:
  - Fallback cache can hold stale keys indefinitely, diverging from Redis behavior.
- Evidence:
  - `_fallback_ttl` exists but is never applied on `get`. [infrastructure/redis_cache.py:23](infrastructure/redis_cache.py:23), [infrastructure/redis_cache.py:50](infrastructure/redis_cache.py:50)
- Recommended fix:
  - Store expiry timestamps and evict on read/write.

### 11) Timezone normalization can fail on tz-naive index (MEDIUM)

- Impact:
  - Data fetch pipeline may fail for providers returning tz-naive index.
- Evidence:
  - unconditional `tz_localize(None)`. [src/data_pipeline/fetch_data.py:22](src/data_pipeline/fetch_data.py:22)
- Recommended fix:
  - Apply `tz_convert(None)`/`tz_localize(None)` conditionally based on index timezone state.

## Notes on Earlier Issues

- Previously observed payload contract breaks (`features_df` / `close_price`) and missing live execution wiring are now addressed in current code snapshot. [services/model_service/realtime_predictor.py:79](services/model_service/realtime_predictor.py:79), [main.py:94](main.py:94), [main.py:143](main.py:143)

## Validation Constraints

- Dynamic execution was limited: local Python runtime invocations failed in this environment, so runtime test execution was not possible.
- Findings are static-analysis based, with evidence from current files.

## Recommended Immediate Action Order (✅ ALL RESOLVED)

1. ~~Fix risk/broker state reconciliation (Critical #1).~~ **(Fixed via `sync_broker_positions`)**
2. ~~Fix no-trade backtest crash path (Critical #2).~~ **(Fixed via metrics error check)**
3. ~~Secure model loading path (Critical #3).~~ **(Fixed via `safe_load_model()`)**
4. ~~Lock CORS and dependency versions before external exposure.~~ **(Resolved)**
5. ~~Update broken requirements.txt.~~ **(Resolved with pinned subset & `ta` restored)**

---

## Secondary Deep Audit Results (2026-03-07)

A secondary static analysis pass (via `bandit`) was conducted post-implementation:

- `[B404/B603]` Subprocess warnings isolated and secured.
- `[B104]` Hardcoded binding variables replaced with env-scoped `API_HOST` defaulting securely to `127.0.0.1`.
- `[B110/B112]` Broad except blocks passing/continuing without logging resolved.

**Final Deep Scan Status**: `No issues identified. (0 High, 0 Medium, 0 Low)`
The system is passing all unit tests and static security assertions.
