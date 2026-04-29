// ============================================================
// BJJ Procesador de Instruccionales — Frontend
// ============================================================

const API = {
    scan: (path) => fetch('/api/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path })
    }).then(r => r.json()),

    videoInfo: (path) => fetch(`/api/video-info?path=${encodeURIComponent(path)}`).then(r => r.json()),

    createJob: (type, path) => fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type, path })
    }).then(r => r.json()),

    listJobs: () => fetch('/api/jobs').then(r => r.json()),

    getSubtitles: (path) => fetch(`/api/subtitles?path=${encodeURIComponent(path)}`).then(r => r.json()),

    thumbnailUrl: (path, t = 5) => `/api/thumbnail?path=${encodeURIComponent(path)}&t=${t}`,
};

// ---- Estado ----
let state = {
    library: [],
    selectedVideo: null,
    activeJobs: {},
};

// ---- DOM ----
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ---- Rutas recientes ----
function getRecentPaths() {
    try { return JSON.parse(localStorage.getItem('bjj_recent_paths') || '[]'); }
    catch { return []; }
}

function addRecentPath(path) {
    let paths = getRecentPaths().filter(p => p !== path);
    paths.unshift(path);
    if (paths.length > 5) paths = paths.slice(0, 5);
    localStorage.setItem('bjj_recent_paths', JSON.stringify(paths));
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    const pathInput = $('#path-input');
    const scanBtn = $('#scan-btn');

    // Cargar ruta guardada
    const savedPath = localStorage.getItem('bjj_library_path');
    if (savedPath) pathInput.value = savedPath;

    scanBtn.addEventListener('click', () => scanLibrary());
    pathInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') scanLibrary();
    });

    // Cerrar panel
    $('#panel-close').addEventListener('click', closePanel);
    $('#overlay').addEventListener('click', closePanel);

    // Popup de ruta
    $('#browse-btn').addEventListener('click', openPathModal);
    $('#modal-close').addEventListener('click', closePathModal);
    $('#modal-cancel').addEventListener('click', closePathModal);
    $('#modal-confirm').addEventListener('click', confirmPathModal);
    $('#modal-path-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') confirmPathModal();
    });

    // Auto-escanear si ya hay ruta
    if (savedPath) scanLibrary();
});

// ---- Modal de ruta ----
function openPathModal() {
    const modal = $('#path-modal');
    const modalInput = $('#modal-path-input');
    const recentDiv = $('#recent-paths');

    modalInput.value = $('#path-input').value;

    // Mostrar rutas recientes
    const recents = getRecentPaths();
    if (recents.length) {
        recentDiv.innerHTML = recents.map(p =>
            `<button class="recent-path-btn" onclick="selectRecentPath('${escapeAttr(p)}')">${escapeHtml(p)}</button>`
        ).join('');
    } else {
        recentDiv.innerHTML = '<span style="color:var(--text-muted); font-size:0.75rem;">Sin rutas recientes</span>';
    }

    modal.style.display = 'flex';
    modalInput.focus();
    modalInput.select();
}

function closePathModal() {
    $('#path-modal').style.display = 'none';
}

function confirmPathModal() {
    const path = $('#modal-path-input').value.trim();
    if (!path) return;
    $('#path-input').value = path;
    closePathModal();
    scanLibrary();
}

function selectRecentPath(path) {
    $('#modal-path-input').value = path;
}

// ---- Escanear biblioteca ----
async function scanLibrary() {
    const path = $('#path-input').value.trim();
    if (!path) {
        openPathModal();
        return;
    }

    localStorage.setItem('bjj_library_path', path);
    addRecentPath(path);

    const scanBtn = $('#scan-btn');
    scanBtn.disabled = true;
    scanBtn.innerHTML = '<span class="spinner"></span> Escaneando...';

    try {
        const data = await API.scan(path);
        if (data.error) {
            showError(data.error);
            return;
        }
        state.library = data.instructionals || [];
        renderLibrary();
        updateStats();
    } catch (e) {
        showError('Error al escanear: ' + e.message);
    } finally {
        scanBtn.disabled = false;
        scanBtn.textContent = 'Escanear Biblioteca';
    }
}

function updateStats() {
    const lib = state.library;
    const totalVideos = lib.reduce((s, i) => s + i.total_videos, 0);
    const totalSubs = lib.reduce((s, i) => s + i.subtitled, 0);
    const totalDubbed = lib.reduce((s, i) => s + i.dubbed, 0);

    $('#stat-instructionals').textContent = lib.length;
    $('#stat-videos').textContent = totalVideos;
    $('#stat-subtitled').textContent = totalSubs;
    $('#stat-dubbed').textContent = totalDubbed;
}

// ---- Renderizado ----
function renderLibrary() {
    const container = $('#library-grid');

    if (!state.library.length) {
        container.innerHTML = `
            <div class="empty-state" style="grid-column: 1 / -1;">
                <div class="icon">&#128218;</div>
                <h3>No se encontraron instruccionales</h3>
                <p>Revisa la ruta e intenta de nuevo. Asegurate de que contenga archivos de video (.mp4, .mkv, .avi, .mov)</p>
            </div>`;
        return;
    }

    container.innerHTML = state.library.map(instr => renderInstructionalCard(instr)).join('');
}

function renderInstructionalCard(instr) {
    const statusBadge = getStatusBadge(instr);
    const progress = instr.total_videos > 0
        ? Math.round((instr.subtitled / instr.total_videos) * 100)
        : 0;

    return `
    <div class="card instructional-card">
        <div class="card-header">
            <h3>${escapeHtml(instr.name)}</h3>
            ${statusBadge}
        </div>
        <div class="card-body">
            <div class="instructional-meta">
                <div class="meta-item">
                    <span>Videos</span>
                    <span class="value">${instr.total_videos}</span>
                </div>
                <div class="meta-item">
                    <span>Subtitulados</span>
                    <span class="value">${instr.subtitled}</span>
                </div>
                <div class="meta-item">
                    <span>Doblados</span>
                    <span class="value">${instr.dubbed}</span>
                </div>
                <div class="meta-item">
                    <span>Capitulos</span>
                    <span class="value">${instr.chapters_detected}</span>
                </div>
            </div>
            <div class="progress-bar">
                <div class="fill" style="width: ${progress}%"></div>
            </div>
            <ul class="video-list">
                ${instr.videos.slice(0, 8).map(v => renderVideoItem(v)).join('')}
                ${instr.videos.length > 8 ? `<li class="video-item" style="justify-content:center; color:var(--text-muted); font-size:0.78rem;">+ ${instr.videos.length - 8} mas</li>` : ''}
            </ul>
        </div>
    </div>`;
}

function renderVideoItem(video) {
    return `
    <li class="video-item" onclick="openVideoPanel('${escapeAttr(video.path)}', '${escapeAttr(video.filename)}')">
        <span class="filename">${escapeHtml(video.filename)}</span>
        <span class="size">${video.size_mb} MB</span>
        <span class="badges">
            ${video.has_subtitles_en ? '<span class="badge badge-success">EN</span>' : ''}
            ${video.has_subtitles_es ? '<span class="badge badge-info">ES</span>' : ''}
            ${video.has_dubbing ? '<span class="badge badge-warning">DUB</span>' : ''}
        </span>
    </li>`;
}

function getStatusBadge(instr) {
    if (instr.dubbed > 0) return '<span class="badge badge-success">Doblado</span>';
    if (instr.subtitled > 0) return '<span class="badge badge-info">Subtitulado</span>';
    if (instr.chapters_detected > 0) return '<span class="badge badge-warning">Capitulos</span>';
    return '<span class="badge badge-neutral">Pendiente</span>';
}

// ---- Panel de detalle ----
async function openVideoPanel(path, filename) {
    state.selectedVideo = { path, filename };

    $('#panel-title').textContent = filename;
    $('#panel-body').innerHTML = '<div style="text-align:center; padding:40px;"><span class="spinner"></span></div>';

    $('#detail-panel').classList.add('open');
    $('#overlay').classList.add('active');

    try {
        const info = await API.videoInfo(path);
        renderPanelContent(path, filename, info);
    } catch (e) {
        $('#panel-body').innerHTML = `<p style="color:var(--error)">Error: ${e.message}</p>`;
    }
}

function closePanel() {
    $('#detail-panel').classList.remove('open');
    $('#overlay').classList.remove('active');
    state.selectedVideo = null;
}

function renderPanelContent(path, filename, info) {
    const basePath = path.replace(/\.[^.]+$/, '');
    const hasSrt = state.library.some(i =>
        i.videos.some(v => v.path === path && v.has_subtitles_en)
    );

    $('#panel-body').innerHTML = `
        <div class="detail-section">
            <h4>Informacion del Video</h4>
            <div class="detail-info-grid">
                <div class="info-box">
                    <div class="label">Duracion</div>
                    <div class="value">${info.duration_formatted || '---'}</div>
                </div>
                <div class="info-box">
                    <div class="label">Resolucion</div>
                    <div class="value">${info.width || '?'}x${info.height || '?'}</div>
                </div>
                <div class="info-box">
                    <div class="label">Tamano</div>
                    <div class="value">${info.size_mb || 0} MB</div>
                </div>
                <div class="info-box">
                    <div class="label">Codec</div>
                    <div class="value">${info.codec || '?'} @ ${info.fps || '?'}fps</div>
                </div>
            </div>
        </div>

        <div class="detail-section">
            <h4>Acciones</h4>
            <div class="detail-actions">
                <button class="btn btn-primary" onclick="startJob('chapters', '${escapeAttr(path)}')">
                    Detectar Capitulos
                </button>
                <button class="btn btn-secondary" onclick="startJob('subtitles', '${escapeAttr(path)}')">
                    Generar Subtitulos (EN)
                </button>
                ${hasSrt ? `
                <button class="btn btn-sm btn-secondary" onclick="viewSubtitles('${escapeAttr(basePath)}.srt')">
                    Ver Subtitulos
                </button>` : ''}
            </div>
        </div>

        <div id="job-area"></div>

        <div id="subtitle-preview"></div>
    `;
}

// ---- Trabajos ----
async function startJob(type, path) {
    const jobArea = $('#job-area');
    if (!jobArea) return;

    const typeLabels = { chapters: 'Detectando capitulos', subtitles: 'Generando subtitulos', dubbing: 'Generando doblaje' };

    try {
        const data = await API.createJob(type, path);
        if (data.error) {
            jobArea.innerHTML = `<p style="color:var(--error)">${data.error}</p>`;
            return;
        }

        const jobId = data.job_id;
        state.activeJobs[jobId] = { type, path };

        jobArea.innerHTML = `
        <div class="job-monitor" id="job-${jobId}">
            <div class="job-status">
                <span><span class="spinner" style="display:inline-block; vertical-align:middle; margin-right:8px;"></span>
                ${typeLabels[type] || 'Procesando'}...</span>
                <span class="badge badge-info" id="job-badge-${jobId}">En curso</span>
            </div>
            <div class="progress-bar" style="margin-bottom:8px;">
                <div class="fill" id="job-progress-${jobId}" style="width:0%"></div>
            </div>
            <div class="log-output" id="job-log-${jobId}"></div>
        </div>`;

        // Conectar SSE
        const evtSource = new EventSource(`/api/jobs/${jobId}/events`);
        evtSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.heartbeat) return;

            if (data.progress !== undefined) {
                const bar = $(`#job-progress-${jobId}`);
                if (bar) bar.style.width = data.progress + '%';
            }

            if (data.message) {
                const logEl = $(`#job-log-${jobId}`);
                if (logEl) {
                    const line = document.createElement('div');
                    line.className = 'log-line';
                    line.textContent = data.message;
                    logEl.appendChild(line);
                    logEl.scrollTop = logEl.scrollHeight;
                }
            }

            if (data.status === 'completed') {
                evtSource.close();
                const badge = $(`#job-badge-${jobId}`);
                if (badge) {
                    badge.className = 'badge badge-success';
                    badge.textContent = 'Completado';
                }
                const spinner = $(`#job-${jobId} .spinner`);
                if (spinner) spinner.style.display = 'none';
            }

            if (data.status === 'failed') {
                evtSource.close();
                const badge = $(`#job-badge-${jobId}`);
                if (badge) {
                    badge.className = 'badge badge-error';
                    badge.textContent = 'Error';
                }
                const spinner = $(`#job-${jobId} .spinner`);
                if (spinner) spinner.style.display = 'none';
            }
        };

        evtSource.onerror = () => { evtSource.close(); };

    } catch (e) {
        jobArea.innerHTML = `<p style="color:var(--error)">Error: ${e.message}</p>`;
    }
}

// ---- Vista previa de subtitulos ----
async function viewSubtitles(srtPath) {
    const preview = $('#subtitle-preview');
    if (!preview) return;

    preview.innerHTML = '<div style="text-align:center; padding:20px;"><span class="spinner"></span></div>';

    try {
        const data = await API.getSubtitles(srtPath);
        if (data.error) {
            preview.innerHTML = `<p style="color:var(--error)">${data.error}</p>`;
            return;
        }

        const subs = data.subtitles || [];
        preview.innerHTML = `
        <div class="detail-section">
            <h4>Subtitulos (${subs.length} bloques)</h4>
            <div style="max-height:300px; overflow-y:auto; background:var(--bg-primary); border-radius:var(--radius-sm); padding:12px;">
                ${subs.slice(0, 50).map(s => `
                    <div style="margin-bottom:10px; font-size:0.8rem;">
                        <span style="color:var(--accent); font-family:var(--font-mono); font-size:0.7rem;">${s.start} &rarr; ${s.end}</span>
                        <div style="color:var(--text-primary); margin-top:2px;">${escapeHtml(s.text)}</div>
                    </div>
                `).join('')}
                ${subs.length > 50 ? `<p style="color:var(--text-muted); text-align:center;">... y ${subs.length - 50} mas</p>` : ''}
            </div>
        </div>`;
    } catch (e) {
        preview.innerHTML = `<p style="color:var(--error)">Error: ${e.message}</p>`;
    }
}

// ---- Mostrar error ----
function showError(msg) {
    const container = $('#library-grid');
    container.innerHTML = `
        <div class="empty-state" style="grid-column: 1 / -1;">
            <div class="icon" style="color:var(--error);">&#9888;</div>
            <h3>Error</h3>
            <p>${escapeHtml(msg)}</p>
        </div>`;
}

// ---- Utilidades ----
function escapeHtml(str) {
    const el = document.createElement('span');
    el.textContent = str || '';
    return el.innerHTML;
}

function escapeAttr(str) {
    return (str || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
}
