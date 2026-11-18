from __future__ import annotations

import json
import mimetypes
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from .config import EXPORT_DIR, ENGINES, RESULT_DIR, UPLOAD_DIR
from .ocr_service import (
    CancelledInference,
    normalize_markdown_math,
    ocr_service,
    rewrite_markdown_image_paths,
)


def _now() -> float:
    return time.time()


def _iso(ts: float | None) -> str | None:
    if not ts:
        return None
    return time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(ts))


def collect_visualizations(job_id: str) -> list[str]:
    root = RESULT_DIR / job_id
    urls: list[str] = []
    if not root.exists():
        return urls
    for path in sorted(root.glob('page_*/visualizations/*')):
        rel = path.relative_to(RESULT_DIR).as_posix()
        urls.append(f'/results/{rel}')
    return urls


def job_urls(job_id: str, filename: str) -> dict[str, str]:
    extension = Path(filename).suffix.lower() or '.bin'
    return {
        'source': f'/results/{job_id}/source{extension}',
        'markdown': f'/results/{job_id}/document.md',
        'json': f'/results/{job_id}/document.json',
    }


def export_file_stem(filename: str) -> str:
    stem = Path(filename).stem.strip() or 'document'
    cleaned = ''.join('_' if ch in '<>:"/\\|?*' else ch for ch in stem).strip(' .')
    return cleaned or 'document'


def export_folder_name(filename: str, when: float | None = None) -> str:
    stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(when or time.time()))
    return f'{export_file_stem(filename)}_{stamp}'


def model_name(engine: str | None) -> str:
    if engine in ENGINES:
        return ENGINES[engine]
    if engine and 'xiaomi' in engine.lower():
        return ENGINES['xiaomi']
    return ENGINES['paddle']


def _engine_from_document(doc: dict[str, Any]) -> str:
    raw = str(doc.get('engine') or '')
    pipeline = str(doc.get('pipeline_version') or '')
    if raw in ENGINES:
        return raw
    if 'xiaomi' in raw.lower() or 'xiaomi' in pipeline.lower():
        return 'xiaomi'
    return 'paddle'


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._hydrate_from_disk()

    def _hydrate_from_disk(self) -> None:
        if not RESULT_DIR.exists():
            return
        for folder in RESULT_DIR.iterdir():
            if not folder.is_dir():
                continue
            meta_path = folder / 'job.json'
            doc_path = folder / 'document.json'
            if not doc_path.exists() and not meta_path.exists():
                continue
            job_id = folder.name
            filename = 'document'
            page_count = 0
            inference_ms = None
            created_at = folder.stat().st_mtime
            mime_type = 'application/octet-stream'
            meta: dict[str, Any] = {}

            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding='utf-8'))
                    filename = meta.get('filename') or filename
                    page_count = meta.get('page_count') or 0
                    inference_ms = meta.get('inference_ms')
                    created_at = meta.get('created_at') or created_at
                    mime_type = meta.get('mime_type') or mime_type
                except Exception:
                    meta = {}

            doc: dict[str, Any] = {}
            if doc_path.exists():
                try:
                    doc = json.loads(doc_path.read_text(encoding='utf-8'))
                    filename = doc.get('source_file') or filename
                    page_count = doc.get('page_count') or page_count
                    created_at = doc_path.stat().st_mtime
                except Exception:
                    doc = {}

            source = next(folder.glob('source.*'), None)
            if source is not None:
                guessed, _ = mimetypes.guess_type(source.name)
                mime_type = guessed or mime_type
                if filename == 'document':
                    filename = source.name.replace('source', filename, 1)

            status = 'completed' if doc_path.exists() else 'failed'
            error = None
            if meta_path.exists():
                try:
                    status = meta.get('status') or status
                    error = meta.get('error')
                except Exception:
                    pass
            engine = meta.get('engine') or _engine_from_document(doc)
            self._jobs[job_id] = {
                'job_id': job_id,
                'filename': filename,
                'status': status,
                'error': error,
                'created_at': created_at,
                'started_at': created_at,
                'finished_at': created_at,
                'inference_ms': inference_ms,
                'page_count': page_count,
                'mime_type': mime_type,
                'engine': engine,
                'options': meta.get('options') or doc.get('options'),
                'cancel_event': threading.Event(),
            }

    def _persist(self, job: dict[str, Any]) -> None:
        folder = RESULT_DIR / job['job_id']
        folder.mkdir(parents=True, exist_ok=True)
        payload = {
            'job_id': job['job_id'],
            'filename': job['filename'],
            'status': job['status'],
            'error': job['error'],
            'created_at': job['created_at'],
            'started_at': job.get('started_at'),
            'finished_at': job.get('finished_at'),
            'inference_ms': job.get('inference_ms'),
            'page_count': job.get('page_count') or 0,
            'mime_type': job.get('mime_type'),
            'engine': job.get('engine'),
            'options': job.get('options'),
        }
        (folder / 'job.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def summary(self, job: dict[str, Any]) -> dict[str, Any]:
        return {
            'job_id': job['job_id'],
            'filename': job['filename'],
            'status': job['status'],
            'error': job.get('error'),
            'created_at': _iso(job.get('created_at')),
            'started_at': _iso(job.get('started_at')),
            'finished_at': _iso(job.get('finished_at')),
            'inference_ms': job.get('inference_ms'),
            'page_count': job.get('page_count') or 0,
            'mime_type': job.get('mime_type'),
            'engine': job.get('engine'),
            'model_name': model_name(job.get('engine')),
            'options': job.get('options'),
            'urls': job_urls(job['job_id'], job['filename']),
            'visualizations': collect_visualizations(job['job_id']) if job['status'] == 'completed' else [],
        }

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda item: item.get('created_at') or 0, reverse=True)
        return [self.summary(job) for job in jobs]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return dict(job)

    def detail(self, job_id: str, include_result: bool = False) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None:
            return None
        payload = self.summary(job)
        if include_result and job['status'] == 'completed':
            output_dir = RESULT_DIR / job_id
            md_path = output_dir / 'document.md'
            json_path = output_dir / 'document.json'
            raw_md = md_path.read_text(encoding='utf-8', errors='replace') if md_path.exists() else ''
            payload['markdown'] = rewrite_markdown_image_paths(raw_md, output_dir)
            if json_path.exists():
                payload['document'] = json.loads(json_path.read_text(encoding='utf-8'))
            else:
                payload['document'] = {}
        return payload

    def create(
        self,
        job_id: str,
        filename: str,
        mime_type: str,
        input_path: Path,
        output_dir: Path,
        engine: str | None = None,
        options: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        job = {
            'options': options,
            'job_id': job_id,
            'filename': filename,
            'status': 'queued',
            'error': None,
            'created_at': _now(),
            'started_at': None,
            'finished_at': None,
            'inference_ms': None,
            'page_count': 0,
            'mime_type': mime_type,
            'engine': engine,
            'input_path': input_path,
            'output_dir': output_dir,
            'cancel_event': threading.Event(),
        }
        with self._lock:
            self._jobs[job_id] = job
        self._persist(job)
        return self.summary(job)

    def export_job(self, job_id: str) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None or job.get('status') != 'completed':
            return None
        source_dir = RESULT_DIR / job_id
        if not source_dir.exists():
            return None

        stem = export_file_stem(job['filename'])
        name = export_folder_name(job['filename'])
        dest = EXPORT_DIR / name
        suffix = 2
        while dest.exists():
            dest = EXPORT_DIR / f'{name}_{suffix}'
            suffix += 1
        dest.mkdir(parents=True, exist_ok=True)

        md_path = source_dir / 'document.md'
        if md_path.exists():
            exported_md = normalize_markdown_math(md_path.read_text(encoding='utf-8', errors='replace'))
            exported_md = rewrite_markdown_image_paths(exported_md, source_dir)
            (dest / f'{stem}.md').write_text(exported_md, encoding='utf-8')
        json_path = source_dir / 'document.json'
        if json_path.exists():
            shutil.copy2(json_path, dest / f'{stem}.json')
        for source in source_dir.glob('source.*'):
            shutil.copy2(source, dest / f'{stem}{source.suffix}')

        skip_files = {'document.md', 'document.json', 'job.json', 'all_results.zip'}
        for item in source_dir.iterdir():
            if item.is_dir() and item.name.startswith('page_'):
                shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
            elif item.is_file() and item.name not in skip_files and not item.name.startswith('source.'):
                shutil.copy2(item, dest / item.name)

        files = sorted(path.relative_to(dest).as_posix() for path in dest.rglob('*') if path.is_file())
        return {
            'folder': str(dest),
            'name': dest.name,
            'files': files,
        }

    def mark_completed(self, job_id: str, inference_ms: int | None, page_count: int) -> None:
        with self._lock:
            stored = self._jobs.get(job_id)
            if stored is None:
                self._remove_job_files(job_id)
                return
            stored['status'] = 'completed'
            stored['inference_ms'] = inference_ms
            stored['page_count'] = page_count
            stored['started_at'] = stored.get('started_at') or stored['created_at']
            stored['finished_at'] = _now()
            stored['error'] = None
            self._persist(stored)

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        folder = RESULT_DIR / job_id
        if job is None and not folder.exists():
            return False
        if job is not None:
            event = job.get('cancel_event')
            if event is not None:
                event.set()
        self._remove_job_files(job_id)
        try:
            from .rag import rag_service
            rag_service.remove_job(job_id)
        except Exception:
            pass
        return True

    def _remove_job_files(self, job_id: str) -> None:
        shutil.rmtree(RESULT_DIR / job_id, ignore_errors=True)
        shutil.rmtree(UPLOAD_DIR / job_id, ignore_errors=True)

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job['status'] in {'completed', 'failed', 'cancelled'}:
                return self.summary(job)
            job['cancel_event'].set()
            if job['status'] == 'queued':
                job['status'] = 'cancelled'
                job['finished_at'] = _now()
                job['error'] = 'Cancelled before start'
                self._persist(job)
            return self.summary(job)

    def clear_all(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
            self._jobs.clear()
        for job in jobs:
            job['cancel_event'].set()
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        shutil.rmtree(RESULT_DIR, ignore_errors=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            from .rag import rag_service
            rag_service.clear()
        except Exception:
            pass

    def run(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return
        if job['cancel_event'].is_set():
            with self._lock:
                stored = self._jobs.get(job_id)
                if stored and stored['status'] == 'queued':
                    stored['status'] = 'cancelled'
                    stored['finished_at'] = _now()
                    stored['error'] = 'Cancelled before start'
                    self._persist(stored)
            return

        with self._lock:
            stored = self._jobs.get(job_id)
            if stored is None or stored['status'] not in {'queued', 'running'}:
                return
            stored['status'] = 'running'
            stored['started_at'] = _now()
            stored['error'] = None
            self._persist(stored)
            cancel_event = stored['cancel_event']
            input_path = stored['input_path']
            output_dir = stored['output_dir']
            engine = stored.get('engine')
            options = stored.get('options')

        try:
            result = ocr_service.process(
                input_path,
                output_dir,
                cancel_event=cancel_event,
                engine=engine,
                options=options,
            )
            self.mark_completed(
                job_id,
                inference_ms=result.get('inference_ms'),
                page_count=result['document']['page_count'],
            )
            try:
                from .rag import rag_service
                rag_service.index_job_async(job_id, force=True)
            except Exception:
                pass
        except CancelledInference:
            with self._lock:
                stored = self._jobs.get(job_id)
                if stored is None:
                    self._remove_job_files(job_id)
                    return
                stored['status'] = 'cancelled'
                stored['error'] = 'Cancelled'
                stored['finished_at'] = _now()
                self._persist(stored)
        except Exception as exc:
            with self._lock:
                stored = self._jobs.get(job_id)
                if stored is None:
                    self._remove_job_files(job_id)
                    return
                stored['status'] = 'failed'
                stored['error'] = str(exc)
                stored['finished_at'] = _now()
                self._persist(stored)


job_store = JobStore()
