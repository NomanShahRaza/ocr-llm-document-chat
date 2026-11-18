from __future__ import annotations

import glob
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from typing import Any

_AMP_PATCHED = False
_SCALAR_PATCHED = False
_VLM_GUARDED = False

from .config import (
    MODEL_VERSION,
    ENGINE,
    ENGINES,
    DEVICE,
    ENABLE_ORIENTATION,
    ENABLE_UNWARPING,
)
from .textfix import literal_newlines


def _load_torch_before_paddle() -> None:
    """Paddle registers an older OpenMP DLL. Torch must load first or shm.dll fails."""
    import torch  # noqa: F401


_load_torch_before_paddle()


class CancelledInference(Exception):
    """Raised when a running OCR job is cancelled."""


def default_options() -> dict[str, bool]:
    return {
        'use_doc_orientation_classify': ENABLE_ORIENTATION,
        'use_doc_unwarping': ENABLE_UNWARPING,
    }


def resolve_options(options: dict[str, Any] | None) -> dict[str, bool]:
    resolved = default_options()
    for key, value in (options or {}).items():
        if key in resolved and value is not None:
            resolved[key] = bool(value)
    return resolved


def normalize_markdown_math(text: str) -> str:
    """Put display math on its own lines so `$` is not parsed inside `$$`."""

    def _display(match: re.Match[str]) -> str:
        tex = match.group(1).strip()
        tex = re.sub(r'\{r\s+l\}', '{rl}', tex)
        tex = re.sub(r'\{l\s+r\}', '{lr}', tex)
        return f'\n\n$$\n{tex}\n$$\n\n'

    normalized = re.sub(r'\$\$([\s\S]+?)\$\$', _display, literal_newlines(text))
    return re.sub(r'\n{3,}', '\n\n', normalized).strip() + '\n'


def rewrite_markdown_image_paths(text: str, root: Path, page_hint: str | None = None) -> str:
    """Point Markdown image URLs at the image file that actually exists under root."""
    if not text:
        return text
    root = root.resolve()

    def resolve(url: str) -> str:
        raw = url.strip()
        if not raw or raw.startswith(('#', 'data:')) or raw.lower().startswith(('http://', 'https://', '/')):
            return url
        rel = raw.replace('\\', '/')
        if rel.startswith('./'):
            rel = rel[2:]
        direct = root / rel
        if direct.is_file():
            return direct.resolve().relative_to(root).as_posix()
        name = Path(rel).name
        if not name or name in {'.', '..'}:
            return url
        matches = [
            path for path in root.glob(f'page_*/**/{glob.escape(name)}')
            if path.is_file()
        ]
        if page_hint:
            hinted = [path for path in matches if page_hint in path.parts]
            if hinted:
                matches = hinted
        if len(matches) == 1:
            return matches[0].resolve().relative_to(root).as_posix()
        return url

    text = re.sub(
        r'(<img\b[^>]*?\bsrc\s*=\s*["\'])([^"\']+)(["\'])',
        lambda match: f'{match.group(1)}{resolve(match.group(2))}{match.group(3)}',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'(!\[[^\]]*\]\()([^)\s]+)',
        lambda match: f'{match.group(1)}{resolve(match.group(2))}',
        text,
    )
    return text


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledInference('Inference cancelled')


def _choose_device() -> str:
    requested = (DEVICE or '').strip().lower()
    gpu_ok = False
    try:
        import paddle

        if paddle.is_compiled_with_cuda():
            gpu_ok = int(paddle.device.cuda.device_count()) > 0
    except Exception:
        gpu_ok = False

    if requested.startswith('cpu') or not gpu_ok:
        return 'cpu'
    if requested.startswith('gpu'):
        return requested if ':' in requested else 'gpu:0'
    return 'gpu:0' if gpu_ok else 'cpu'


def _concrete_place(device=None):
    import paddle
    from paddle.base import core
    from paddle.base.framework import _current_expected_place_, _get_paddle_place

    if device is None:
        place = _current_expected_place_()
    else:
        place = _get_paddle_place(device)
    if type(place) is core.Place:
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return core.CUDAPlace(0)
        return core.CPUPlace()
    return place


def _patch_paddle_amp_checks() -> None:
    """PIR mode makes `amp.is_bfloat16_supported()` see Place(undefined:0)."""
    global _AMP_PATCHED
    if _AMP_PATCHED:
        return

    import paddle
    from paddle.base import core

    def is_bfloat16_supported(device=None) -> bool:
        try:
            return bool(core.is_bfloat16_supported(_concrete_place(device)))
        except Exception:
            return False

    def is_float16_supported(device=None) -> bool:
        try:
            return bool(core.is_float16_supported(_concrete_place(device)))
        except Exception:
            return str(_choose_device()).startswith('gpu')

    paddle.amp.is_bfloat16_supported = is_bfloat16_supported
    paddle.amp.is_float16_supported = is_float16_supported
    _AMP_PATCHED = True


_CONVERTING = threading.local()


def _tensor_to_numpy(value):
    import numpy as np

    numpy_fn = getattr(value, 'numpy', None)
    if callable(numpy_fn):
        try:
            return np.asarray(numpy_fn())
        except Exception:
            pass
    try:
        return np.array(value)
    except Exception:
        return None


def _as_python_int(value) -> int:
    import numpy as np

    if value is None:
        return 0
    if isinstance(value, (int, np.integer, bool)):
        return int(value)
    if getattr(_CONVERTING, 'on', False):
        raise TypeError('int(Tensor) is not available')
    _CONVERTING.on = True
    try:
        arr = _tensor_to_numpy(value)
        if arr is None:
            raise TypeError('int(Tensor) is not available')
        return int(arr.reshape(-1)[0])
    finally:
        _CONVERTING.on = False


def _as_python_float(value) -> float:
    import numpy as np

    if value is None:
        return 0.0
    if isinstance(value, (float, int, np.floating, np.integer, bool)):
        return float(value)
    if getattr(_CONVERTING, 'on', False):
        raise TypeError('float(Tensor) is not available')
    _CONVERTING.on = True
    try:
        arr = _tensor_to_numpy(value)
        if arr is None:
            raise TypeError('float(Tensor) is not available')
        return float(arr.reshape(-1)[0])
    finally:
        _CONVERTING.on = False


def _patch_paddle_scalar_casts() -> None:
    """PaddleOCR-VL calls int(tensor) while PIR thinks the graph is static."""
    global _SCALAR_PATCHED
    if _SCALAR_PATCHED:
        return

    import paddle
    from paddle.pir import Value

    def _int_(self):
        return _as_python_int(self)

    def _float_(self):
        return _as_python_float(self)

    Value.__int__ = _int_
    Value.__index__ = _int_
    Value.__float__ = _float_

    tensor_cls = getattr(paddle, 'Tensor', None)
    if tensor_cls is not None and isinstance(tensor_cls, type):
        tensor_cls.__int__ = _int_
        tensor_cls.__index__ = _int_
        tensor_cls.__float__ = _float_

    _SCALAR_PATCHED = True


def _guard_vlm_process(pipeline) -> None:
    """Keep the VLM predictor on a real CUDA/CPU place in dynamic mode."""
    global _VLM_GUARDED
    if _VLM_GUARDED:
        return
    inner = getattr(pipeline, 'paddlex_pipeline', pipeline)
    model = getattr(inner, 'vl_rec_model', None)
    if model is None or not hasattr(model, 'process'):
        return
    original = model.process

    def wrapped(*args, **kwargs):
        ensure_paddle_device()
        return original(*args, **kwargs)

    model.process = wrapped
    _VLM_GUARDED = True


def ensure_paddle_device() -> str:
    """Bind Paddle to a real CUDA/CPU place on the current thread."""
    import paddle
    from paddle.base import core, framework

    if not paddle.in_dynamic_mode():
        paddle.disable_static()

    target = _choose_device()
    paddle.device.set_device(target)
    if target.startswith('gpu'):
        index = int(target.split(':')[1]) if ':' in target else 0
        framework._set_expected_place(core.CUDAPlace(index))
    else:
        framework._set_expected_place(core.CPUPlace())
    _patch_paddle_amp_checks()
    _patch_paddle_scalar_casts()
    return target


def detect_runtime() -> dict[str, Any]:
    """Return GPU/CUDA availability without loading the OCR model."""
    requested = DEVICE or 'auto'
    info: dict[str, Any] = {
        'requested_device': requested,
        'gpu_available': False,
        'cuda_compiled': False,
        'label': 'CPU',
        'name': None,
        'model_version': 'Xiaomi-OCR-0' if ENGINE.lower() == 'xiaomi' else MODEL_VERSION,
        'engine': ENGINE,
        'pipeline_loaded': False,
    }
    if ENGINE.lower() == 'xiaomi':
        name = _nvidia_gpu_name()
        if name:
            info.update({
                'gpu_available': True,
                'cuda_compiled': True,
                'label': 'GPU (CUDA)',
                'name': name,
            })
        return info

    try:
        import paddle

        cuda_compiled = bool(paddle.is_compiled_with_cuda())
        info['cuda_compiled'] = cuda_compiled
        device_count = 0
        if cuda_compiled:
            try:
                device_count = int(paddle.device.cuda.device_count())
            except Exception:
                device_count = 0

        if cuda_compiled and device_count > 0:
            name = None
            try:
                name = paddle.device.cuda.get_device_name(0)
            except Exception:
                name = 'CUDA GPU'
            info.update({
                'gpu_available': True,
                'label': 'GPU (CUDA)',
                'name': name,
                'device_count': device_count,
            })
        else:
            info['label'] = 'CPU'
            info['name'] = 'CPU'
    except Exception:
        info['label'] = 'CPU'
        info['name'] = 'CPU'

    return info


def _nvidia_gpu_name() -> str | None:
    import subprocess

    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            text=True,
            timeout=8,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[0] if lines else None


class OCRService:
    """Lazy-loaded singleton wrapper around PaddleOCR-VL.

    All pipeline load and inference runs on one dedicated thread so the CUDA
    context stays valid. The lock also prevents concurrent VLM runs from
    exhausting GPU memory.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._device: str | None = None
        self._runtime: dict[str, Any] | None = None
        self._engine = ENGINE.lower() if ENGINE.lower() in ENGINES else 'xiaomi'
        self._busy_count = 0
        self._state_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ocr-gpu')

    @property
    def engine(self) -> str:
        return self._engine

    def engine_options(self) -> list[dict[str, str]]:
        return [{'id': key, 'label': label} for key, label in ENGINES.items()]

    def set_engine(self, engine: str) -> str:
        chosen = (engine or '').lower()
        if chosen not in ENGINES:
            raise ValueError(f'Unknown OCR model: {engine}')
        with self._state_lock:
            if self._busy_count:
                raise RuntimeError('Wait for the current run to finish before switching models')
            if chosen == self._engine:
                return chosen
            self._engine = chosen
            future = self._executor.submit(self._release_other, chosen)
        future.result()
        return chosen

    def _release_other(self, keep: str) -> None:
        if keep != 'xiaomi':
            from .xiaomi_ocr import xiaomi_engine
            xiaomi_engine.release()
        if keep != 'paddle':
            self._drop_pipeline()

    def runtime(self) -> dict[str, Any]:
        if self._runtime is None:
            self._runtime = detect_runtime()
        snapshot = dict(self._runtime)
        snapshot['engine'] = self._engine
        snapshot['model_version'] = ENGINES[self._engine]
        snapshot['engines'] = self.engine_options()
        if self._engine == 'xiaomi':
            from .xiaomi_ocr import xiaomi_engine
            snapshot['pipeline_loaded'] = xiaomi_engine.loaded
        else:
            snapshot['pipeline_loaded'] = self._pipeline is not None
        snapshot['busy'] = self._busy_count > 0
        if self._device:
            snapshot['active_device'] = self._device
        return snapshot

    def _drop_pipeline(self) -> None:
        global _VLM_GUARDED
        if self._pipeline is None:
            return
        self._pipeline = None
        self._device = None
        _VLM_GUARDED = False
        try:
            import gc
            gc.collect()
            import paddle
            empty_cache = getattr(getattr(paddle.device, 'cuda', None), 'empty_cache', None)
            if callable(empty_cache):
                empty_cache()
        except Exception:
            pass

    def _get_pipeline(self, cancel_event: threading.Event | None = None):
        if self._pipeline is not None:
            ensure_paddle_device()
            _guard_vlm_process(self._pipeline)
            return self._pipeline

        _raise_if_cancelled(cancel_event)
        self._device = ensure_paddle_device()
        from paddleocr import PaddleOCRVL
        # Load both preprocessors so each run can switch them on or off via predict().
        kwargs = {
            'pipeline_version': MODEL_VERSION,
            'use_doc_orientation_classify': True,
            'use_doc_unwarping': True,
            'device': self._device,
            'use_queues': False,
        }
        self._pipeline = PaddleOCRVL(**kwargs)
        _guard_vlm_process(self._pipeline)
        return self._pipeline

    def process(
        self,
        input_path: Path,
        output_dir: Path,
        cancel_event: threading.Event | None = None,
        engine: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._state_lock:
            self._busy_count += 1
            chosen = (engine or self._engine).lower()
            future = self._executor.submit(
                self._process_on_gpu_thread,
                input_path,
                output_dir,
                cancel_event,
                chosen,
                resolve_options(options),
            )
        try:
            return future.result()
        finally:
            with self._state_lock:
                self._busy_count = max(0, self._busy_count - 1)

    def _process_on_gpu_thread(
        self,
        input_path: Path,
        output_dir: Path,
        cancel_event: threading.Event | None = None,
        engine: str | None = None,
        options: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        options = options or resolve_options(None)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        _raise_if_cancelled(cancel_event)
        chosen = (engine or self._engine).lower()
        if chosen == 'xiaomi':
            self._drop_pipeline()
            from .xiaomi_ocr import xiaomi_engine
            return xiaomi_engine.parse(input_path, output_dir, cancel_event)
        from .xiaomi_ocr import xiaomi_engine
        xiaomi_engine.release()
        pipeline = self._get_pipeline(cancel_event)
        _raise_if_cancelled(cancel_event)
        try:
            pages = list(pipeline.predict(
                input=str(input_path),
                use_queues=False,
                use_doc_orientation_classify=options['use_doc_orientation_classify'],
                use_doc_unwarping=options['use_doc_unwarping'],
            ))
        except Exception as exc:
            message = str(exc)
            if 'static graph' in message or 'int(Tensor)' in message:
                self._drop_pipeline()
            raise
        _raise_if_cancelled(cancel_event)

        # For multi-page PDFs, use the model's document restructuring stage.
        # This preserves/merges cross-page structure where supported.
        try:
            if input_path.suffix.lower() == '.pdf' and len(pages) > 1:
                structured_pages = list(
                    pipeline.restructure_pages(
                        pages,
                        merge_tables=True,
                        relevel_titles=True,
                        concatenate_pages=False,
                    )
                )
            else:
                structured_pages = pages
        except CancelledInference:
            raise
        except Exception:
            # Older PaddleOCR builds may not expose restructure_pages.
            structured_pages = pages

        page_json: list[dict[str, Any]] = []
        markdown_parts: list[str] = []

        for index, result in enumerate(structured_pages, start=1):
            _raise_if_cancelled(cancel_event)
            page_dir = output_dir / f'page_{index:04d}'
            page_dir.mkdir(parents=True, exist_ok=True)

            # Save all supported native outputs. Individual methods can vary
            # slightly by PaddleOCR version, so optional exporters are isolated.
            result.save_to_json(save_path=page_dir)
            result.save_to_markdown(save_path=page_dir)

            for method_name, target in (
                ('save_to_word', output_dir),
                ('save_to_html', page_dir),
                ('save_to_xlsx', page_dir),
                ('save_to_img', page_dir / 'visualizations'),
            ):
                method = getattr(result, method_name, None)
                if callable(method):
                    try:
                        Path(target).mkdir(parents=True, exist_ok=True)
                        method(save_path=target)
                    except Exception:
                        pass

            raw_json = _json_safe(getattr(result, 'json', {}))
            if isinstance(raw_json, dict):
                raw_json.setdefault('_app_page_number', index)
            page_json.append(raw_json)

            md_obj = getattr(result, 'markdown', None)
            md_text = ''
            if isinstance(md_obj, dict):
                md_text = md_obj.get('markdown_texts', '') or ''
            if not md_text:
                md_files = sorted(page_dir.glob('*.md'))
                if md_files:
                    md_text = md_files[0].read_text(encoding='utf-8', errors='replace')
            markdown_parts.append(
                rewrite_markdown_image_paths(
                    md_text.strip(),
                    output_dir,
                    page_hint=f'page_{index:04d}',
                )
            )

        canonical = {
            'source_file': input_path.name,
            'pipeline_version': MODEL_VERSION,
            'engine': chosen,
            'options': options,
            'page_count': len(page_json),
            'pages': page_json,
        }

        (output_dir / 'document.json').write_text(
            json.dumps(canonical, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        (output_dir / 'document.md').write_text(
            normalize_markdown_math('\n\n---\n\n'.join(p for p in markdown_parts if p)),
            encoding='utf-8',
        )

        inference_ms = int((time.perf_counter() - started) * 1000)
        return {
            'document': canonical,
            'markdown': (output_dir / 'document.md').read_text(encoding='utf-8', errors='replace'),
            'inference_ms': inference_ms,
        }


ocr_service = OCRService()
