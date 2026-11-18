from __future__ import annotations

import mimetypes
import os
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ALLOWED_EXTENSIONS, EXPORT_DIR, MAX_UPLOAD_MB, RESULT_DIR, UPLOAD_DIR
from .jobs import job_store
from .ocr_service import CancelledInference, default_options, ocr_service, resolve_options
from .rag import rag_service

app = FastAPI(title='OCR-LLM', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

STATIC_DIR = Path(__file__).resolve().parent / 'static'
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
app.mount('/results', StaticFiles(directory=RESULT_DIR), name='results')


async def _save_upload(file: UploadFile, original_name: str, extension: str) -> tuple[str, Path, Path]:
    job_id = uuid.uuid4().hex
    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = RESULT_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_upload_dir / original_name
    written = 0
    chunk_size = 1024 * 1024
    with input_path.open('wb') as out:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_MB * 1024 * 1024:
                out.close()
                shutil.rmtree(job_upload_dir, ignore_errors=True)
                shutil.rmtree(job_output_dir, ignore_errors=True)
                raise HTTPException(status_code=413, detail=f'File exceeds {MAX_UPLOAD_MB} MB limit')
            out.write(chunk)

    preview_path = job_output_dir / f'source{extension}'
    shutil.copy2(input_path, preview_path)
    return job_id, input_path, job_output_dir


def _start_job(job_id: str) -> None:
    thread = threading.Thread(
        target=job_store.run,
        args=(job_id,),
        name=f'ocr-{job_id[:8]}',
        daemon=True,
    )
    thread.start()


@app.get('/')
def index():
    # The page pins versioned asset URLs, so it must be revalidated to pick up new ones.
    return FileResponse(STATIC_DIR / 'index.html', headers={'Cache-Control': 'no-cache'})


@app.get('/favicon.ico')
def favicon():
    return FileResponse(STATIC_DIR / 'icons' / 'icon.png')


@app.get('/api/health')
def health():
    runtime = ocr_service.runtime()
    return {
        'status': 'ok',
        'gpu_available': runtime['gpu_available'],
        'device_label': runtime['label'],
        'device_name': runtime['name'],
        'cuda_compiled': runtime['cuda_compiled'],
        'model_version': runtime['model_version'],
        'engine': runtime['engine'],
        'engines': runtime.get('engines') or [],
        'pipeline_loaded': runtime['pipeline_loaded'],
        'busy': runtime['busy'],
        'default_options': default_options(),
    }


@app.get('/api/jobs')
def list_jobs():
    return {'jobs': job_store.list_jobs()}


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str, include_result: bool = True):
    payload = job_store.detail(job_id, include_result=include_result)
    if payload is None:
        raise HTTPException(status_code=404, detail='Job not found')
    return payload


@app.post('/api/jobs')
async def create_job(
    file: UploadFile = File(...),
    use_doc_orientation_classify: bool | None = Form(None),
    use_doc_unwarping: bool | None = Form(None),
):
    original_name = Path(file.filename or 'upload').name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f'Unsupported file type: {extension}')

    job_id, input_path, output_dir = await _save_upload(file, original_name, extension)
    mime, _ = mimetypes.guess_type(original_name)
    job = job_store.create(
        job_id=job_id,
        filename=original_name,
        mime_type=mime or 'application/octet-stream',
        input_path=input_path,
        output_dir=output_dir,
        engine=ocr_service.engine,
        options=resolve_options({
            'use_doc_orientation_classify': use_doc_orientation_classify,
            'use_doc_unwarping': use_doc_unwarping,
        }),
    )
    _start_job(job_id)
    return job


@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id: str):
    job = job_store.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')
    return job


@app.post('/api/jobs/{job_id}/save')
def save_job(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')
    if job.get('status') != 'completed':
        raise HTTPException(status_code=400, detail='Run OCR before saving results')
    saved = job_store.export_job(job_id)
    if saved is None:
        raise HTTPException(status_code=404, detail='No results to save')
    return saved


class OpenFolderRequest(BaseModel):
    name: str


class EngineRequest(BaseModel):
    engine: str = Field(min_length=1, max_length=255)


class ChatTurn(BaseModel):
    role: str = Field(pattern='^(user|assistant)$')
    content: str = Field(max_length=8000)


class RagSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    mode: str = Field(default='hybrid', max_length=32)
    top_k: int = Field(default=8, ge=1, le=50)
    job_id: str | None = None
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)


class RagReindexRequest(BaseModel):
    job_id: str | None = None
    force: bool = True


@app.get('/api/engine')
def engine_status():
    return {
        'engine': ocr_service.engine,
        'engines': ocr_service.engine_options(),
        'busy': ocr_service.runtime().get('busy', False),
    }


@app.post('/api/engine')
def set_engine(payload: EngineRequest):
    try:
        chosen = ocr_service.set_engine(payload.engine)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    label = next(item['label'] for item in ocr_service.engine_options() if item['id'] == chosen)
    return {'engine': chosen, 'label': label, 'engines': ocr_service.engine_options()}


@app.post('/api/exports/open')
def open_export_folder(body: OpenFolderRequest):
    name = body.name.strip()
    if not name or name != Path(name).name or name in {'.', '..'}:
        raise HTTPException(status_code=400, detail='Invalid folder')
    dest = (EXPORT_DIR / name).resolve()
    root = EXPORT_DIR.resolve()
    if dest.parent != root or not dest.is_dir():
        raise HTTPException(status_code=404, detail='Folder not found')
    try:
        os.startfile(dest)  # Windows Explorer
    except OSError as exc:
        raise HTTPException(status_code=500, detail='Could not open the folder') from exc
    return {'opened': True, 'name': dest.name}


@app.delete('/api/jobs/{job_id}')
def delete_job(job_id: str):
    if not job_store.delete_job(job_id):
        raise HTTPException(status_code=404, detail='Job not found')
    return {'deleted': True, 'job_id': job_id}


@app.delete('/api/jobs')
def clear_jobs():
    job_store.clear_all()
    return {'cleared': True}


@app.get('/api/rag/health')
def rag_health():
    return rag_service.health()


@app.post('/api/rag/search')
def rag_search(payload: RagSearchRequest):
    try:
        return rag_service.search(
            query=payload.query,
            mode=payload.mode,
            top_k=payload.top_k,
            job_id=payload.job_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/rag/ask')
def rag_ask(payload: RagSearchRequest):
    try:
        return rag_service.ask(
            query=payload.query,
            mode=payload.mode,
            top_k=payload.top_k,
            job_id=payload.job_id,
            history=[turn.model_dump() for turn in payload.history],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/rag/reindex')
def rag_reindex(payload: RagReindexRequest):
    # Indexing loads BGE-M3 on first use; run in background so the API stays responsive.
    if payload.job_id:
        if not payload.force and payload.job_id in rag_service.store.job_ids():
            count = sum(1 for chunk in rag_service.store.chunks if chunk['job_id'] == payload.job_id)
            return {'status': 'ready', 'job_id': payload.job_id, 'chunk_count': count}
        rag_service.index_job_async(payload.job_id, force=payload.force)
        return {'status': 'started', 'job_id': payload.job_id}
    thread = threading.Thread(
        target=rag_service.reindex_all,
        name='rag-reindex-all',
        daemon=True,
    )
    thread.start()
    return {'status': 'started', 'job_id': None}


@app.post('/api/parse')
async def parse_document(
    file: UploadFile = File(...),
    use_doc_orientation_classify: bool | None = Form(None),
    use_doc_unwarping: bool | None = Form(None),
):
    original_name = Path(file.filename or 'upload').name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f'Unsupported file type: {extension}')

    job_id, input_path, job_output_dir = await _save_upload(file, original_name, extension)
    mime, _ = mimetypes.guess_type(original_name)
    options = resolve_options({
        'use_doc_orientation_classify': use_doc_orientation_classify,
        'use_doc_unwarping': use_doc_unwarping,
    })

    try:
        result = ocr_service.process(input_path, job_output_dir, options=options)
    except CancelledInference as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'OCR failed: {exc}') from exc

    job = job_store.create(
        job_id=job_id,
        filename=original_name,
        mime_type=mime or 'application/octet-stream',
        input_path=input_path,
        output_dir=job_output_dir,
        engine=ocr_service.engine,
        options=options,
    )
    job_store.mark_completed(
        job_id,
        inference_ms=result.get('inference_ms'),
        page_count=result['document']['page_count'],
    )
    try:
        rag_service.index_job_async(job_id, force=True)
    except Exception:
        pass

    return {
        'job_id': job['job_id'],
        'filename': original_name,
        'mime_type': mime or 'application/octet-stream',
        'page_count': result['document']['page_count'],
        'inference_ms': result.get('inference_ms'),
        'markdown': result['markdown'],
        'document': result['document'],
        'urls': {
            'source': f'/results/{job_id}/source{extension}',
            'markdown': f'/results/{job_id}/document.md',
            'json': f'/results/{job_id}/document.json',
        },
    }
