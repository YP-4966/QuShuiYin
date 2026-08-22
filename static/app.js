/**
 * VSR Web - 前端交互逻辑
 */
(function () {
    'use strict';

    // ── Elements ──────────────────────────────────────────
    const $ = (s) => document.querySelector(s);
    const uploadZone = $('#uploadZone');
    const fileInput = $('#fileInput');
    const uploadContent = $('#uploadContent');
    const fileInfo = $('#fileInfo');
    const fileName = $('#fileName');
    const fileSize = $('#fileSize');
    const fileIcon = $('#fileIcon');
    const btnRemove = $('#btnRemove');
    const videoPreview = $('#videoPreview');
    const imagePreview = $('#imagePreview');
    const previewPlaceholder = $('#previewPlaceholder');
    const btnProcess = $('#btnProcess');
    const btnDownload = $('#btnDownload');
    const progressContainer = $('#progressContainer');
    const progressFill = $('#progressFill');
    const progressPercent = $('#progressPercent');
    const progressLabel = $('#progressLabel');
    const progressDetail = $('#progressDetail');
    const engineStatus = $('#engineStatus');
    const engineText = $('#engineText');
    const useArea = $('#useArea');

    let currentTaskId = null;
    let selectedMethod = 'telea';
    let ws = null;

    // ── 初始化 ────────────────────────────────────────────
    async function init() {
        try {
            const res = await fetch('/api/info');
            const info = await res.json();
            if (info.rapid_ocr_available) {
                engineStatus.classList.add('ready');
                engineText.textContent = 'RapidOCR 就绪';
            } else {
                engineText.textContent = 'OCR 引擎未加载';
            }
        } catch {
            engineText.textContent = '连接失败';
        }
    }

    // ── 文件上传 ──────────────────────────────────────────
    uploadZone.addEventListener('click', (e) => {
        if (e.target.closest('.btn-remove')) return;
        fileInput.click();
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    btnRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    async function handleFile(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        const videoExts = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'];
        const isVideo = videoExts.includes(ext);

        // 显示文件信息
        uploadZone.classList.add('has-file');
        uploadContent.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        fileName.textContent = file.name;
        fileSize.textContent = formatSize(file.size);
        fileIcon.textContent = isVideo ? '🎬' : '🖼️';

        // 预览
        const url = URL.createObjectURL(file);
        previewPlaceholder.classList.add('hidden');
        if (isVideo) {
            imagePreview.classList.add('hidden');
            videoPreview.classList.remove('hidden');
            videoPreview.src = url;
        } else {
            videoPreview.classList.add('hidden');
            imagePreview.classList.remove('hidden');
            imagePreview.src = url;
        }

        // 上传到服务器
        btnProcess.disabled = true;
        btnProcess.querySelector('.btn-text').textContent = '上传中...';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch('/api/upload', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.error) {
                alert(data.error);
                resetUpload();
                return;
            }
            currentTaskId = data.task_id;
            btnProcess.disabled = false;
            btnProcess.querySelector('.btn-text').textContent = '开始处理';
        } catch (err) {
            alert('上传失败: ' + err.message);
            resetUpload();
        }
    }

    function resetUpload() {
        currentTaskId = null;
        uploadZone.classList.remove('has-file');
        uploadContent.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        fileInput.value = '';
        videoPreview.classList.add('hidden');
        videoPreview.src = '';
        imagePreview.classList.add('hidden');
        imagePreview.src = '';
        previewPlaceholder.classList.remove('hidden');
        btnProcess.disabled = true;
        btnProcess.querySelector('.btn-text').textContent = '开始处理';
        btnProcess.querySelector('.btn-loading').classList.add('hidden');
        btnProcess.querySelector('.btn-text').classList.remove('hidden');
        progressContainer.classList.add('hidden');
        btnDownload.classList.add('hidden');
    }

    // ── 模式选择 ──────────────────────────────────────────
    document.querySelectorAll('.mode-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mode-btn').forEach((b) => b.classList.remove('active'));
            btn.classList.add('active');
            selectedMethod = btn.dataset.method;
        });
    });

    // ── 开始处理 ──────────────────────────────────────────
    btnProcess.addEventListener('click', async () => {
        if (!currentTaskId) return;

        // UI 状态
        btnProcess.disabled = true;
        btnProcess.querySelector('.btn-text').classList.add('hidden');
        btnProcess.querySelector('.btn-loading').classList.remove('hidden');
        progressContainer.classList.remove('hidden');
        btnDownload.classList.add('hidden');
        progressFill.style.width = '0%';
        progressPercent.textContent = '0%';
        progressLabel.textContent = '准备处理...';
        progressDetail.textContent = '';

        // 连接 WebSocket 接收进度
        connectWS(currentTaskId);

        // 构建表单
        const formData = new FormData();
        formData.append('method', selectedMethod);
        if (useArea.checked) {
            const yMin = $('#areaYMin').value;
            const yMax = $('#areaYMax').value;
            const xMin = $('#areaXMin').value;
            const xMax = $('#areaXMax').value;
            formData.append('sub_area', `${yMin},${yMax},${xMin},${xMax}`);
        }

        try {
            const res = await fetch(`/api/process/${currentTaskId}`, {
                method: 'POST',
                body: formData,
            });
            const data = await res.json();

            if (data.status === 'completed') {
                updateProgress(100, '处理完成！', '');
                btnDownload.classList.remove('hidden');
            } else if (data.status === 'failed') {
                updateProgress(0, '处理失败', data.message || data.error || '未知错误');
            }
        } catch (err) {
            updateProgress(0, '请求失败', err.message);
        } finally {
            btnProcess.disabled = false;
            btnProcess.querySelector('.btn-text').classList.remove('hidden');
            btnProcess.querySelector('.btn-loading').classList.add('hidden');
        }
    });

    // ── 下载 ──────────────────────────────────────────────
    btnDownload.addEventListener('click', () => {
        if (currentTaskId) {
            window.location.href = `/api/download/${currentTaskId}`;
        }
    });

    // ── WebSocket ─────────────────────────────────────────
    function connectWS(taskId) {
        if (ws) { try { ws.close(); } catch {} }
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${location.host}/ws/progress/${taskId}`);
        ws.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                updateProgress(data.progress, data.message, '');
                if (data.status === 'completed') {
                    btnDownload.classList.remove('hidden');
                }
            } catch {}
        };
        ws.onerror = () => {};
        ws.onclose = () => {};
    }

    function updateProgress(percent, label, detail) {
        progressFill.style.width = percent + '%';
        progressPercent.textContent = percent + '%';
        if (label) progressLabel.textContent = label;
        if (detail) progressDetail.textContent = detail;
    }

    // ── 工具函数 ──────────────────────────────────────────
    function formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + ' MB';
        return (bytes / 1073741824).toFixed(2) + ' GB';
    }

    init();
})();
