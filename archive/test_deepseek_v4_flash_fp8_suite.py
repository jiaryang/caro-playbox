"""DeepSeek-V4-Flash-FP8 combined test suite (8-GPU, AMD ROCm / MI325).

NOTE (caro-playbox): archived stray copy; not part of the GLM suite.
Canonical location is
``sglang/test/registered/amd/test_deepseek_v4_flash_fp8_suite.py`` in the
SGLang checkout. Prefer running it from there.

Consolidates the three previously separate DSV4-Flash-FP8 test files into one
module with a shared server recipe and an auto-generated report. All classes run
the same model (``sgl-project/DeepSeek-V4-Flash-FP8``) with TP=8, the ``dsv4``
attention backend, fp8_e4m3 KV cache, the AITER/ROCm env vars, and EAGLE (MTP)
speculative decoding.

Test categories
---------------
1. Accuracy & Performance   -- ``TestDSV4FlashFP8AccuracyPerf``
   GSM8K few-shot accuracy gate + 8k->1k throughput benchmark on one server.
2. Output Sanity & Correctness / Spec-decode  -- ``TestDSV4FlashFP8Sanity``
   Platform-agnostic HTTP probes (correctness, ascii/gibberish ratio,
   no-repetition, temp=0 determinism, max_token_one), GSM8K accuracy, and the
   bs=1 speed + acceptance-length spec-decode check (via shared mixins).
3. MTP Parity (losslessness) -- ``TestDSV4FlashFP8MTPParity``
   Teacher-forced argmax-agreement check that EAGLE/MTP does not shift the
   output distribution beyond intrinsic fp8 path noise.

Running the module writes a categorized report (markdown + json) to
``REPORT_DIR`` (default: repo root) summarizing per-test status, duration, and
captured metrics.

Run:
    export HF_HOME=/data/huggingface
    python test/registered/amd/test_deepseek_v4_flash_fp8_suite.py
    # single class, e.g.:
    python test/registered/amd/test_deepseek_v4_flash_fp8_suite.py \
        TestDSV4FlashFP8MTPParity
"""

import concurrent.futures as cf
import datetime
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=7200,
    suite="nightly-amd-8-gpu-mi35x-deepseek-v4-flash",
    nightly=True,
)

# --------------------------------------------------------------------------- #
# Shared configuration (identical across all three original test files).
# --------------------------------------------------------------------------- #
DEEPSEEK_V4_FP8_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_FP8_MODEL_PATH", "sgl-project/DeepSeek-V4-Flash-FP8"
)
SERVER_LAUNCH_TIMEOUT = 3600
FLASHMLA_BACKEND = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "unified_kv_triton")

# Common DeepSeek-V4 env vars (AMD ROCm path: AITER indexer + triton attn),
# aligned with the manual MI325 MTP server config.
COMMON_ENV_VARS = {
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_ROCM700A": "0",
    "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
    "SGLANG_OPT_USE_FUSED_COMPRESS_TRITON": "true",
    "SGLANG_HACK_FLASHMLA_BACKEND": FLASHMLA_BACKEND,
    "SGLANG_OPT_FP8_WO_A_GEMM": "false",
    "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "false",
    "SGLANG_OPT_USE_TOPK_V2": "false",
    "SGLANG_OPT_USE_AITER_INDEXER": "true",
    "SGLANG_OPT_USE_TILELANG_INDEXER": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
    "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "false",
    "SGLANG_ROCM_USE_MULTI_STREAM": "false",
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "SGLANG_EAGER_INPUT_NO_COPY": "true",
}

# FP8 variant (fp8 experts, not fp4).
FP8_ENV_VARS = {
    "SGLANG_DSV4_FP4_EXPERTS": "false",
}

# Server args shared by every launch (no speculative flags here).
BASE_ARGS = [
    "--trust-remote-code",
    "--tp",
    "8",
    "--disable-radix-cache",
    "--attention-backend",
    "dsv4",
    "--max-running-requests",
    "512",
    "--page-size",
    "256",
    "--mem-fraction-static",
    "0.80",
    "--swa-full-tokens-ratio",
    "0.1",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--chunked-prefill-size",
    "16384",
    "--cuda-graph-max-bs",
    "512",
    "--disable-shared-experts-fusion",
    "--tool-call-parser",
    "deepseekv4",
    "--reasoning-parser",
    "deepseek-v4",
]

# MTP / EAGLE speculative decoding (NextN head from the base model).
MTP_ARGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]

# The Accuracy/Perf class historically gated MTP behind an env var so it can
# also measure the non-spec baseline; the other classes keep MTP always on.
ENABLE_MTP_ACCURACY_CLASS = os.environ.get("SGLANG_DSV4_ENABLE_MTP", "1") == "1"


def _make_env():
    env = os.environ.copy()
    env.update(COMMON_ENV_VARS)
    env.update(FP8_ENV_VARS)
    return env


# --------------------------------------------------------------------------- #
# Report collection.
# --------------------------------------------------------------------------- #
# Tests we fully control push structured numbers here; the reporting runner also
# scrapes printed metrics from captured stdout as a fallback.
SUITE_METRICS: dict = {}

CATEGORY_ACCURACY = "1. Accuracy & Performance"
CATEGORY_SANITY = "2. Output Sanity & Correctness"
CATEGORY_SPEC = "3. Speculative Decoding (MTP)"
CATEGORY_PARITY = "4. MTP Parity (losslessness)"

CATEGORY_ORDER = [
    CATEGORY_ACCURACY,
    CATEGORY_SANITY,
    CATEGORY_SPEC,
    CATEGORY_PARITY,
]

CATEGORY_BY_METHOD = {
    "test_a_gsm8k": CATEGORY_ACCURACY,
    "test_b_perf_8k_1k": CATEGORY_ACCURACY,
    "test_gsm8k": CATEGORY_ACCURACY,
    "test_bs_1_speed": CATEGORY_SPEC,
    "test_capital_france": CATEGORY_SANITY,
    "test_basic_math": CATEGORY_SANITY,
    "test_color_completion": CATEGORY_SANITY,
    "test_ascii_ratio": CATEGORY_SANITY,
    "test_no_repetition_blowup": CATEGORY_SANITY,
    "test_determinism_temp_zero": CATEGORY_SANITY,
    "test_max_token_one": CATEGORY_SANITY,
    "test_a_mtp_active": CATEGORY_PARITY,
    "test_parity_argmax_agreement": CATEGORY_PARITY,
}


def record_metric(method, data):
    SUITE_METRICS.setdefault(method, {}).update(data)


# --------------------------------------------------------------------------- #
# 1. Accuracy & Performance.
# --------------------------------------------------------------------------- #
class TestDSV4FlashFP8AccuracyPerf(CustomTestCase):
    """GSM8K few-shot accuracy gate + 8k->1k throughput benchmark (one server)."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_FP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = list(BASE_ARGS)
        if ENABLE_MTP_ACCURACY_CLASS:
            other_args += MTP_ARGS
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=_make_env(),
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None):
            kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        # `a` prefix runs first (alphabetical) and warms up the server.
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=64,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")
        record_metric(
            "test_a_gsm8k",
            {
                "accuracy": round(float(metrics["accuracy"]), 4),
                "harness": "few_shot_gsm8k (8-shot, parallel=64, n=1319)",
                "threshold": 0.91,
            },
        )
        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v4-flash-fp8, {FLASHMLA_BACKEND})\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )
            self.assertGreater(metrics["accuracy"], 0.91)

    @unittest.skipIf(
        os.environ.get("SGLANG_DSV4_ACCURACY_ONLY") == "1",
        "SGLANG_DSV4_ACCURACY_ONLY=1: accuracy-only run (skipping perf)",
    )
    def test_b_perf_8k_1k(self):
        json_output = "/tmp/deepseek_v4_flash_fp8_perf.json"
        if os.path.exists(json_output):
            os.remove(json_output)

        # First "1" is a warmup; the report below skips it.
        batch_sizes = ["1", "1", "2", "4", "8", "16", "32"]
        cmd = [
            "python3",
            "-m",
            "sglang.bench_one_batch_server",
            "--model",
            "None",
            "--base-url",
            self.base_url,
            "--batch-size",
            *batch_sizes,
            "--input-len",
            "8192",
            "--output-len",
            "1024",
            "--show-report",
            f"--pydantic-result-filename={json_output}",
            "--no-append-to-github-summary",
            "--trust-remote-code",
        ]
        print(f"Running benchmark: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            self.fail(f"bench_one_batch_server failed (rc={result.returncode})")

        self.assertTrue(
            os.path.exists(json_output),
            f"Benchmark JSON output {json_output} not found",
        )
        with open(json_output) as f:
            results_data = json.load(f)
        self.assertTrue(results_data, "No benchmark results returned")

        if (
            len(results_data) > 1
            and results_data[0]["batch_size"] == results_data[1]["batch_size"]
        ):
            report_results = results_data[1:]
        else:
            report_results = results_data

        perf_rows = []
        summary_lines = [
            f"### test_perf_8k_1k (deepseek-v4-flash-fp8, {FLASHMLA_BACKEND})",
            "input_len=8192 output_len=1024",
            "",
            "| batch size | latency (s) | input throughput (tok/s) | output throughput (tok/s) | ITL (ms) |",
            "| ---------- | ----------- | ------------------------ | ------------------------- | -------- |",
        ]
        for r in report_results:
            bs = r["batch_size"]
            latency = r.get("latency", 0.0)
            in_tp = r.get("input_throughput", 0.0)
            out_tp = r.get("output_throughput", 0.0)
            itl = 1 / (out_tp / bs) * 1000 if out_tp > 0 else float("inf")
            summary_lines.append(
                f"| {bs} | {latency:.2f} | {in_tp:.2f} | {out_tp:.2f} | {itl:.2f} |"
            )
            perf_rows.append(
                {
                    "batch_size": bs,
                    "latency_s": round(latency, 2),
                    "input_tok_s": round(in_tp, 2),
                    "output_tok_s": round(out_tp, 2),
                    "itl_ms": round(itl, 2),
                }
            )
            print(
                f"bs={bs} latency={latency:.2f}s "
                f"in_tp={in_tp:.2f} tok/s out_tp={out_tp:.2f} tok/s ITL={itl:.2f}ms"
            )

        record_metric("test_b_perf_8k_1k", {"perf_table": perf_rows})
        if is_in_ci():
            write_github_step_summary("\n".join(summary_lines) + "\n")


# --------------------------------------------------------------------------- #
# 2/3. Output sanity + accuracy + spec-decode (mixin-driven, MTP always on).
# --------------------------------------------------------------------------- #
class TestDSV4FlashFP8Sanity(
    SpecDecodingMixin,
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """MI325 recipe probes: correctness sanity, GSM8K, and bs=1 spec speed."""

    # -- GSM8KMixin --
    gsm8k_accuracy_thres = 0.91
    gsm8k_num_questions = 1319
    gsm8k_num_shots = 8
    # 64 concurrent (matches the accuracy class). The mixin default (128) floods
    # the server with concurrent large prefills and trips a GPU memory access
    # fault on MI325 under this recipe.
    gsm8k_num_threads = 64
    gsm8k_accept_length_thres = None  # accept-length asserted by SpecDecodingMixin

    # -- SpecDecodingMixin --
    # Conservative floors. Manual DSV4-Flash MTP runs show accept length ~2.9-3.0
    # (num_steps=3, topk=1, num_draft=4).
    accept_length_thres = 1.8
    bs_1_speed_thres = 50.0

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_FP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=BASE_ARGS + MTP_ARGS,
            env=_make_env(),
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None):
            kill_process_tree(cls.process.pid)


# --------------------------------------------------------------------------- #
# 4. MTP parity (losslessness).
# --------------------------------------------------------------------------- #
MAX_NEW_TOKENS = 128
TOP_LOGPROBS_NUM = 5
# Fillers keep the reference decode batch >= 2. The plain (non-MTP) fp8 decode
# path degenerates at batch size 1 (uniform logits -> token-0 spam), so the
# SELF-baseline decode must never run alone.
NUM_FILLERS = 8

# MTP argmax agreement must stay within MARGIN of the model's own decode-vs-
# prefill argmax agreement (the intrinsic fp8 ceiling), and above an absolute
# floor as a backstop. Lossy speculative decoding would drop well below both.
ARGMAX_AGREE_MARGIN = 0.05
ARGMAX_AGREE_ABS_FLOOR = 0.85

PARITY_DUMP_PATH = os.environ.get(
    "PARITY_DUMP_PATH", "/sgl-workspace/sglang/parity_outputs.json"
)

PARITY_PROMPTS = [
    "Explain step by step how photosynthesis works in plants.",
    "Write a short story about a robot who learns to paint.",
    "Describe in detail how a bicycle converts pedaling into motion.",
    "def fibonacci(n):",
    "List the first ten prime numbers and briefly explain what a prime number is.",
    "Summarize the plot of Romeo and Juliet in a few sentences.",
]


def _greedy(base_url, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Free greedy generation. Returns prompt_ids + generated token ids/text."""
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()
    meta = data["meta_info"]
    return {
        "prompt_ids": [t[1] for t in meta["input_token_logprobs"]],
        "out_ids": [t[1] for t in meta["output_token_logprobs"]],
        "text": data["text"],
    }


def _prefill_argmax(base_url, prompt_ids, out_ids):
    """Teacher-force ``prompt_ids + out_ids`` through one prefill; return the
    model's argmax token id at each output position. None for the boundary."""
    full_ids = list(prompt_ids) + list(out_ids)
    start = len(prompt_ids)
    resp = requests.post(
        base_url + "/generate",
        json={
            "input_ids": full_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_logprob": True,
            "top_logprobs_num": TOP_LOGPROBS_NUM,
            "logprob_start_len": start,
        },
        timeout=600,
    )
    resp.raise_for_status()
    meta = resp.json()["meta_info"]
    argmax = []
    for pos_top in meta.get("input_top_logprobs", []) or []:
        argmax.append(max(pos_top, key=lambda t: t[0])[1] if pos_top else None)
    return argmax


def _agreement(out_ids, argmax):
    """Fraction of positions where the prefill argmax == the fed token.

    Returns (agree, positions, disagree_indices)."""
    n = min(len(out_ids), len(argmax))
    idx = [i for i in range(n) if argmax[i] is not None]
    agree = sum(1 for i in idx if argmax[i] == out_ids[i])
    disagree = [i for i in idx if argmax[i] != out_ids[i]]
    return agree, len(idx), disagree


def _filler(base_url, stop_flag):
    """Keep the decode batch >= 2 while the reference SELF-baseline generates."""
    while not stop_flag["stop"]:
        try:
            requests.post(
                base_url + "/generate",
                json={
                    "text": "Count slowly: one, two, three,",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 400,
                        "ignore_eos": True,
                    },
                },
                timeout=600,
            )
        except Exception:
            pass


class TestDSV4FlashFP8MTPParity(CustomTestCase):
    """MTP (EAGLE) must be lossless: every emitted token stays the greedy
    (argmax) choice of the plain model, at (nearly) the model's own decode-vs-
    prefill agreement rate. Exact token-equality is impossible under fp8 noise,
    so we use a teacher-forced argmax-agreement rate judged against that ceiling.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_FP8_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = _make_env()

        # 1) MTP server: capture free-greedy outputs, record spec algo, tear down.
        mtp_proc = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=BASE_ARGS + MTP_ARGS,
            env=env,
        )
        try:
            cls.spec_algo = (
                requests.get(cls.base_url + "/server_info")
                .json()
                .get("speculative_algorithm")
            )
            cls.mtp_outputs = {p: _greedy(cls.base_url, p) for p in PARITY_PROMPTS}
        finally:
            kill_process_tree(mtp_proc.pid, wait_timeout=120)

        # 2) Reference (non-MTP) server: kept alive for the test methods.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=BASE_ARGS,
            env=env,
        )

        # 2a) SELF baseline: batched greedy decode (fillers -> batch >= 2).
        stop_flag = {"stop": False}
        fillers = [
            threading.Thread(target=_filler, args=(cls.base_url, stop_flag), daemon=True)
            for _ in range(NUM_FILLERS)
        ]
        for t in fillers:
            t.start()
        time.sleep(2)
        try:
            with cf.ThreadPoolExecutor(max_workers=len(PARITY_PROMPTS)) as ex:
                futs = {ex.submit(_greedy, cls.base_url, p): p for p in PARITY_PROMPTS}
                cls.ref_outputs = {futs[f]: f.result() for f in cf.as_completed(futs)}
        finally:
            stop_flag["stop"] = True
            time.sleep(1)

        # 2b) Teacher-forced argmax agreement on the reference prefill.
        cls.per_prompt = {}
        self_agree = self_pos = mtp_agree = mtp_pos = 0
        for p in PARITY_PROMPTS:
            ref = cls.ref_outputs[p]
            mtp = cls.mtp_outputs[p]

            self_argmax = _prefill_argmax(cls.base_url, ref["prompt_ids"], ref["out_ids"])
            sa, sp, s_dis = _agreement(ref["out_ids"], self_argmax)
            mtp_argmax = _prefill_argmax(cls.base_url, mtp["prompt_ids"], mtp["out_ids"])
            ma, mp, m_dis = _agreement(mtp["out_ids"], mtp_argmax)

            self_agree += sa
            self_pos += sp
            mtp_agree += ma
            mtp_pos += mp
            cls.per_prompt[p] = {
                "mtp": {"text": mtp["text"], "token_ids": mtp["out_ids"]},
                "reference": {"text": ref["text"], "token_ids": ref["out_ids"]},
                "self_argmax_agree": sa,
                "self_argmax_positions": sp,
                "self_argmax_agree_rate": round(sa / sp, 4) if sp else 1.0,
                "self_disagree_pos": s_dis,
                "mtp_argmax_agree": ma,
                "mtp_argmax_positions": mp,
                "mtp_argmax_agree_rate": round(ma / mp, 4) if mp else 1.0,
                "mtp_disagree_pos": m_dis,
            }

        cls.self_rate = self_agree / self_pos if self_pos else 1.0
        cls.mtp_rate = mtp_agree / mtp_pos if mtp_pos else 1.0

        cls.summary = {
            "num_prompts": len(PARITY_PROMPTS),
            "self_argmax_agree_rate": round(cls.self_rate, 4),
            "mtp_argmax_agree_rate": round(cls.mtp_rate, 4),
            "margin": ARGMAX_AGREE_MARGIN,
            "abs_floor": ARGMAX_AGREE_ABS_FLOOR,
            "threshold": round(
                max(cls.self_rate - ARGMAX_AGREE_MARGIN, ARGMAX_AGREE_ABS_FLOOR), 4
            ),
        }
        with open(PARITY_DUMP_PATH, "w") as f:
            json.dump(
                {
                    "config": {"base_args": BASE_ARGS, "mtp_args": MTP_ARGS},
                    "summary": cls.summary,
                    "prompts": cls.per_prompt,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nParity dump written to {PARITY_DUMP_PATH}")
        print(f"summary: {cls.summary}")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None):
            kill_process_tree(cls.process.pid)

    def test_a_mtp_active(self):
        """Sanity: the captured outputs came from an EAGLE (MTP) server."""
        self.assertEqual(self.spec_algo, "EAGLE")

    def test_parity_argmax_agreement(self):
        """MTP tokens must be greedy-consistent with the non-MTP reference at
        (nearly) the same rate as the reference is with itself across paths."""
        threshold = self.summary["threshold"]
        for p, v in self.per_prompt.items():
            print(
                f"[{p[:50]!r}] self={v['self_argmax_agree_rate']:.2%} "
                f"mtp={v['mtp_argmax_agree_rate']:.2%} "
                f"mtp_disagree@{v['mtp_disagree_pos']}"
            )
        print(
            f"OVERALL self={self.self_rate:.2%} mtp={self.mtp_rate:.2%} "
            f"threshold={threshold:.2%}"
        )
        record_metric(
            "test_parity_argmax_agreement",
            {
                "self_rate": round(self.self_rate, 4),
                "mtp_rate": round(self.mtp_rate, 4),
                "threshold": round(threshold, 4),
            },
        )
        self.assertGreaterEqual(
            self.mtp_rate,
            threshold,
            f"MTP argmax agreement {self.mtp_rate:.2%} < threshold {threshold:.2%} "
            f"(self baseline {self.self_rate:.2%}). MTP may be introducing bias "
            f"beyond intrinsic fp8 path noise. See {PARITY_DUMP_PATH}.",
        )


# --------------------------------------------------------------------------- #
# Report generation.
# --------------------------------------------------------------------------- #
REPORT_DIR = os.environ.get("REPORT_DIR", "/sgl-workspace/sglang")
REPORT_MD = os.path.join(REPORT_DIR, "deepseek_v4_flash_fp8_report.md")
REPORT_JSON = os.path.join(REPORT_DIR, "deepseek_v4_flash_fp8_report.json")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


class ReportingResult(unittest.TextTestResult):
    """Records per-test status/duration and tees stdout so the report can scrape
    printed metrics (gsm8k score, accept length, parity rates)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self._status = {}
        self._buf = None
        self._real_stdout = None
        self._t0 = 0.0

    def startTest(self, test):
        super().startTest(test)
        self._buf = io.StringIO()
        self._real_stdout = sys.stdout
        sys.stdout = _Tee(self._real_stdout, self._buf)
        self._t0 = time.time()

    def stopTest(self, test):
        if self._real_stdout is not None:
            sys.stdout = self._real_stdout
        dur = time.time() - self._t0
        method = getattr(test, "_testMethodName", "?")
        captured = self._buf.getvalue() if self._buf else ""
        self.records.append(
            {
                "class": type(test).__name__,
                "method": method,
                "category": CATEGORY_BY_METHOD.get(method, "Other"),
                "status": self._status.get(id(test), "pass"),
                "duration_s": round(dur, 1),
                "metrics": _scrape_metrics(method, captured),
            }
        )
        super().stopTest(test)

    def addError(self, test, err):
        super().addError(test, err)
        self._status[id(test)] = "error"

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._status[id(test)] = "fail"

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._status[id(test)] = "skip"


def _scrape_metrics(method, text):
    """Merge structured metrics recorded by the test with values scraped from
    its captured stdout."""
    m = dict(SUITE_METRICS.get(method, {}))
    g = re.search(r"'score':\s*(?:np\.float64\()?\s*([0-9.]+)", text)
    if g and "gsm8k_score" not in m and "accuracy" not in m:
        m["gsm8k_score"] = round(float(g.group(1)), 4)
    a = re.search(r"avg_spec_accept_length=([0-9.]+)", text)
    if a:
        m["avg_spec_accept_length"] = round(float(a.group(1)), 4)
    speed = re.search(r"Speed \(token/s\)\s*\|\s*\n?.*?([0-9]+\.[0-9]+)", text)
    if speed:
        m.setdefault("bs1_speed_tok_s", round(float(speed.group(1)), 2))
    return m


def _status_icon(status):
    return {"pass": "PASS", "fail": "FAIL", "error": "ERROR", "skip": "SKIP"}.get(
        status, status.upper()
    )


def _write_report(result):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = getattr(result, "records", [])

    by_cat = {c: [] for c in CATEGORY_ORDER}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)

    def _counts(recs):
        c = {"pass": 0, "fail": 0, "error": 0, "skip": 0}
        for r in recs:
            c[r["status"]] = c.get(r["status"], 0) + 1
        return c

    lines = [
        "# DeepSeek-V4-Flash-FP8 Test Suite Report",
        "",
        f"- Generated: {ts}",
        f"- Model: `{DEEPSEEK_V4_FP8_MODEL_PATH}`",
        "- Recipe: TP=8, `--attention-backend dsv4`, KV `fp8_e4m3`, "
        "EAGLE/MTP (num_steps=3, eagle_topk=1, num_draft_tokens=4)",
        f"- Overall: {'OK' if result.wasSuccessful() else 'FAILED'} "
        f"({len(records)} tests)",
        "",
        "## Category summary",
        "",
        "| Category | Tests | Pass | Fail | Error | Skip |",
        "| -------- | ----- | ---- | ---- | ----- | ---- |",
    ]
    for cat in CATEGORY_ORDER:
        recs = by_cat.get(cat, [])
        if not recs:
            continue
        c = _counts(recs)
        lines.append(
            f"| {cat} | {len(recs)} | {c['pass']} | {c['fail']} | "
            f"{c['error']} | {c['skip']} |"
        )

    for cat in CATEGORY_ORDER:
        recs = by_cat.get(cat, [])
        if not recs:
            continue
        lines += ["", f"## {cat}", ""]
        lines += [
            "| Test | Status | Duration (s) | Metrics |",
            "| ---- | ------ | ------------ | ------- |",
        ]
        for r in recs:
            metrics = r.get("metrics") or {}
            mstr = _format_metrics(metrics)
            lines.append(
                f"| `{r['class']}.{r['method']}` | {_status_icon(r['status'])} "
                f"| {r['duration_s']} | {mstr} |"
            )
        # Expand the perf table if present.
        for r in recs:
            perf = (r.get("metrics") or {}).get("perf_table")
            if perf:
                lines += [
                    "",
                    "**Perf (input_len=8192, output_len=1024):**",
                    "",
                    "| bs | latency (s) | in tok/s | out tok/s | ITL (ms) |",
                    "| -- | ----------- | -------- | --------- | -------- |",
                ]
                for row in perf:
                    lines.append(
                        f"| {row['batch_size']} | {row['latency_s']} | "
                        f"{row['input_tok_s']} | {row['output_tok_s']} | "
                        f"{row['itl_ms']} |"
                    )

    lines.append("")
    report_md = "\n".join(lines)

    try:
        with open(REPORT_MD, "w") as f:
            f.write(report_md)
        with open(REPORT_JSON, "w") as f:
            json.dump(
                {
                    "generated": ts,
                    "model": DEEPSEEK_V4_FP8_MODEL_PATH,
                    "overall_ok": result.wasSuccessful(),
                    "records": records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\n[report] wrote {REPORT_MD} and {REPORT_JSON}")
    except OSError as e:
        print(f"[report] failed to write report: {e}")


def _format_metrics(metrics):
    parts = []
    for k, v in metrics.items():
        if k == "perf_table":
            parts.append("see perf table below")
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    return "; ".join(parts) if parts else "-"


if __name__ == "__main__":
    # run_suite.py launches each file with `python3 <file> -f` (fail-fast). Our
    # classes share very expensive server launches and record independent
    # measurements, so strip `-f` to keep collecting data/metrics after a
    # failure, and always emit the report.
    sys.argv = [a for a in sys.argv if a not in ("-f", "--failfast")]
    prog = unittest.main(
        exit=False,
        testRunner=unittest.TextTestRunner(
            verbosity=2, resultclass=ReportingResult
        ),
    )
    _write_report(prog.result)
    sys.exit(0 if prog.result.wasSuccessful() else 1)
