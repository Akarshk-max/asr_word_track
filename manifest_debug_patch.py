import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Debug logging config (provided by runtime)
DEBUG_LOG_PATH = "debug-8c96a7.log"
DEBUG_SESSION_ID = "8c96a7"


def _dbg(run_id: str, hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _extract_path(entry: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    # Keep candidate keys ordered by likelihood across user manifests.
    candidates = [
        "audio_filepath",
        "audiofilepath",
        "audio_file_path",
        "path",
        "filepath",
    ]
    for k in candidates:
        if k in entry:
            return entry.get(k), k
    return None, None


def _timed_isfile(p: str) -> Tuple[bool, float, Optional[str]]:
    t0 = time.time()
    try:
        ok = os.path.isfile(p)
        dt_ms = (time.time() - t0) * 1000.0
        return ok, dt_ms, None
    except Exception as e:
        dt_ms = (time.time() - t0) * 1000.0
        return False, dt_ms, str(e)


def read_manifest_with_hypothesis_logs(test_manifest_path: str) -> Dict[str, List]:
    """Drop-in replacement for your SECTION 10 manifest reader, with hypothesis logs."""
    run_id = f"manifest-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    audio_paths: List[str] = []
    reference_texts: List[str] = []
    durations: List[float] = []

    counters = {
        "total_lines": 0,
        "json_error": 0,
        "non_dict_entry": 0,
        "invalid_path_type": 0,
        "empty_path": 0,
        "null_byte_path": 0,
        "resolved_direct": 0,
        "resolved_prefixed": 0,
        "unresolved_path": 0,
        "alt_key_audiofilepath_used": 0,
        "missing_all_audio_path_keys": 0,
        "isfile_exception": 0,
        "slow_isfile_checks": 0,
    }
    key_counter: Dict[str, int] = {}

    # region agent log
    _dbg(
        run_id,
        "H0",
        "run_inference.py:manifest:start",
        "manifest read started",
        {"manifest_path": test_manifest_path, "exists": os.path.isfile(test_manifest_path)},
    )
    # endregion

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            counters["total_lines"] += 1
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                counters["json_error"] += 1
                # region agent log
                _dbg(
                    run_id,
                    "H3",
                    "run_inference.py:manifest:json_loads",
                    "json decode error",
                    {"line_no": line_no, "error": str(e), "preview": line[:160]},
                )
                # endregion
                continue

            if not isinstance(entry, dict):
                counters["non_dict_entry"] += 1
                # region agent log
                _dbg(
                    run_id,
                    "H2",
                    "run_inference.py:manifest:entry_type",
                    "manifest row is not a dict",
                    {"line_no": line_no, "entry_type": type(entry).__name__, "preview": str(entry)[:160]},
                )
                # endregion
                continue

            raw_path, picked_key = _extract_path(entry)
            if picked_key is None:
                counters["missing_all_audio_path_keys"] += 1
                # region agent log
                _dbg(
                    run_id,
                    "H1",
                    "run_inference.py:manifest:key_missing",
                    "no supported audio path key found",
                    {"line_no": line_no, "available_keys": list(entry.keys())[:20]},
                )
                # endregion
                continue
            key_counter[picked_key] = key_counter.get(picked_key, 0) + 1
            if picked_key == "audiofilepath":
                counters["alt_key_audiofilepath_used"] += 1
                if counters["alt_key_audiofilepath_used"] <= 5:
                    # region agent log
                    _dbg(
                        run_id,
                        "H1",
                        "run_inference.py:manifest:key_alias",
                        "using alias key audiofilepath",
                        {"line_no": line_no},
                    )
                    # endregion
            text = entry.get("text", "")
            dur_raw = entry.get("duration", 0.0)

            try:
                dur = float(dur_raw)
            except (TypeError, ValueError):
                dur = 0.0

            if not isinstance(raw_path, (str, bytes, os.PathLike)):
                counters["invalid_path_type"] += 1
                # region agent log
                _dbg(
                    run_id,
                    "H1",
                    "run_inference.py:manifest:path_type",
                    "audio_filepath has invalid type",
                    {"line_no": line_no, "path_type": type(raw_path).__name__, "value_preview": str(raw_path)[:160]},
                )
                # endregion
                continue

            path = os.fspath(raw_path).strip()
            if not path:
                counters["empty_path"] += 1
                # region agent log
                _dbg(
                    run_id,
                    "H1",
                    "run_inference.py:manifest:empty_path",
                    "audio_filepath empty after normalization",
                    {"line_no": line_no},
                )
                # endregion
                continue

            if "\x00" in path:
                counters["null_byte_path"] += 1
                # region agent log
                _dbg(
                    run_id,
                    "H4",
                    "run_inference.py:manifest:null_byte",
                    "audio_filepath contains null byte",
                    {"line_no": line_no, "path_preview": path[:160]},
                )
                # endregion
                continue

            ok, dt_ms, err = _timed_isfile(path)
            if err is not None:
                counters["isfile_exception"] += 1
                # region agent log
                _dbg(
                    run_id,
                    "H2",
                    "run_inference.py:manifest:isfile_exception",
                    "os.path.isfile raised exception",
                    {"line_no": line_no, "error": err, "path_preview": path[:160]},
                )
                # endregion
                continue
            if dt_ms > 50:
                counters["slow_isfile_checks"] += 1
                if counters["slow_isfile_checks"] <= 10:
                    # region agent log
                    _dbg(
                        run_id,
                        "H3",
                        "run_inference.py:manifest:isfile_slow",
                        "slow os.path.isfile check",
                        {"line_no": line_no, "latency_ms": round(dt_ms, 2), "path_preview": path[:160]},
                    )
                    # endregion
            if ok:
                counters["resolved_direct"] += 1
                audio_paths.append(path)
                reference_texts.append(text)
                durations.append(dur)
                continue

            resolved = False
            for prefix in ["/kaggle/input/", "/kaggle/input/datasets/"]:
                candidate = os.path.join(prefix, path.lstrip("/"))
                ok2, dt_ms2, err2 = _timed_isfile(candidate)
                if err2 is not None:
                    counters["isfile_exception"] += 1
                    if counters["isfile_exception"] <= 10:
                        # region agent log
                        _dbg(
                            run_id,
                            "H2",
                            "run_inference.py:manifest:candidate_isfile_exception",
                            "candidate os.path.isfile raised exception",
                            {"line_no": line_no, "error": err2, "candidate_preview": candidate[:160]},
                        )
                        # endregion
                    continue
                if dt_ms2 > 50:
                    counters["slow_isfile_checks"] += 1
                    if counters["slow_isfile_checks"] <= 10:
                        # region agent log
                        _dbg(
                            run_id,
                            "H3",
                            "run_inference.py:manifest:candidate_isfile_slow",
                            "slow candidate os.path.isfile check",
                            {"line_no": line_no, "latency_ms": round(dt_ms2, 2), "candidate_preview": candidate[:160]},
                        )
                        # endregion
                if ok2:
                    counters["resolved_prefixed"] += 1
                    audio_paths.append(candidate)
                    reference_texts.append(text)
                    durations.append(dur)
                    resolved = True
                    break

            if not resolved:
                counters["unresolved_path"] += 1
                # region agent log
                _dbg(
                    run_id,
                    "H5",
                    "run_inference.py:manifest:unresolved",
                    "audio path unresolved; kept as-is",
                    {"line_no": line_no, "path_preview": path[:160]},
                )
                # endregion
                audio_paths.append(path)
                reference_texts.append(text)
                durations.append(dur)

    # region agent log
    _dbg(
        run_id,
        "H0",
        "run_inference.py:manifest:end",
        "manifest read completed",
        {"counters": counters, "path_key_usage": key_counter},
    )
    # endregion

    return {
        "audio_paths": audio_paths,
        "reference_texts": reference_texts,
        "durations": durations,
        "run_id": run_id,
        "counters": counters,
        "path_key_usage": key_counter,
    }


def log_decode_device_hypothesis(device_type: str, cuda_available: bool, run_id: str) -> None:
    """Use before decode to test CUDA/AMP mismatch hypothesis."""
    # region agent log
    _dbg(
        run_id,
        "H6",
        "run_inference.py:decode:device_probe",
        "decode device probe",
        {
            "device_type": device_type,
            "cuda_available": bool(cuda_available),
            "cuda_autocast_should_be_enabled": bool(device_type == "cuda" and cuda_available),
        },
    )
    # endregion


def log_adapter_hypothesis(
    run_id: str,
    replaced_count: int,
    layer_counter_after_probe: int,
    decoded_text_sample: str,
) -> None:
    """Log whether adapters likely executed during the probe forward pass."""
    # region agent log
    _dbg(
        run_id,
        "H4",
        "run_inference.py:adapter:probe",
        "adapter execution probe",
        {
            "replaced_count": replaced_count,
            "layer_counter_after_probe": layer_counter_after_probe,
            "decoded_text_sample": decoded_text_sample[:80],
            "adapter_likely_executed": bool(layer_counter_after_probe >= 40),
        },
    )
    # endregion

