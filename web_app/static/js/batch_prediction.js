/**
 * batch_prediction.js
 * Controller for Batch Fmax Prediction
 * Handles CSV file selection, upload, validation, table rendering,
 * and result CSV download.
 */

let selectedFile = null;
let currentBatchResults = [];

document.addEventListener('DOMContentLoaded', () => {
    setupFileInput();
    setupFormSubmission();
    setupDownloadButton();
});

// ---------------------------------------------------------------------------
// 1. Setup File Input & Drag/Drop
// ---------------------------------------------------------------------------
function setupFileInput() {
    const fileInput = document.getElementById('batchCsvInput');
    const btnBrowse = document.getElementById('btnBrowseFile');
    const dropZone = document.getElementById('dropZone');

    btnBrowse.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.style.borderColor = 'var(--primary-blue)';
            dropZone.style.backgroundColor = '#0e172a';
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.style.borderColor = 'var(--border-color)';
            dropZone.style.backgroundColor = '#0b111e';
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) {
            handleFileSelection(dt.files[0]);
        }
    });
}

function handleFileSelection(file) {
    clearAlerts();

    if (!file.name.toLowerCase().endsWith('.csv')) {
        showError('Please select a valid CSV file (.csv extension).');
        selectedFile = null;
        document.getElementById('selectedFileInfo').classList.add('d-none');
        document.getElementById('btnRunBatch').disabled = true;
        return;
    }

    selectedFile = file;

    // Display selected file info
    const sizeStr = formatFileSize(file.size);
    document.getElementById('selectedFileName').textContent = file.name;
    document.getElementById('selectedFileSize').textContent = sizeStr;
    document.getElementById('selectedFileInfo').classList.remove('d-none');

    // Enable Run button
    document.getElementById('btnRunBatch').disabled = false;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// ---------------------------------------------------------------------------
// 2. Setup Form Submission & Batch Prediction
// ---------------------------------------------------------------------------
function setupFormSubmission() {
    const form = document.getElementById('batchUploadForm');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!selectedFile) {
            showError('Please choose a CSV file first.');
            return;
        }

        const btn = document.getElementById('btnRunBatch');
        const spinner = document.getElementById('batchSpinner');
        const playIcon = document.getElementById('batchPlayIcon');

        btn.disabled = true;
        spinner.classList.remove('d-none');
        playIcon.classList.add('d-none');
        clearAlerts();

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('/api/batch-predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.message || 'Batch prediction failed.');
            }

            currentBatchResults = result.data || [];
            renderResultsTable(currentBatchResults, result.summary);
            showSuccess(`Successfully generated predictions for ${currentBatchResults.length} chip records.`);

        } catch (err) {
            console.error('Batch prediction error:', err);
            showError(err.message || 'An error occurred during batch prediction.');
        } finally {
            btn.disabled = false;
            spinner.classList.add('d-none');
            playIcon.classList.remove('d-none');
        }
    });
}

// ---------------------------------------------------------------------------
// 3. Render Results Table & Summary
// ---------------------------------------------------------------------------
function renderResultsTable(records, summary) {
    const tbody = document.getElementById('batchTableBody');
    tbody.innerHTML = '';

    if (!records || records.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="text-center py-5 text-secondary">
                    No records found in prediction results.
                </td>
            </tr>
        `;
        document.getElementById('btnDownloadResultsCsv').disabled = true;
        document.getElementById('batchSummaryStrip').classList.add('d-none');
        return;
    }

    // Update Summary Strip
    if (summary) {
        document.getElementById('summaryTotalChips').textContent = summary.total_records.toLocaleString();
        document.getElementById('summaryAvgFmax').textContent = `${summary.avg_pred_fmax} MHz`;
        document.getElementById('summaryMinFmax').textContent = `${summary.min_pred_fmax} MHz`;
        document.getElementById('summaryMaxFmax').textContent = `${summary.max_pred_fmax} MHz`;
        document.getElementById('batchSummaryStrip').classList.remove('d-none');
    }

    document.getElementById('batchStatusText').textContent = 
        `Processed ${records.length} records using GBR model and calibrated window logic.`;

    // Populate rows
    records.forEach((row, idx) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="text-secondary">${idx + 1}</td>
            <td class="fw-bold text-white">${escapeHtml(row.chip_id)}</td>
            <td class="fw-bold text-cyan">${row.predicted_fmax_mhz.toFixed(1)} MHz</td>
            <td class="text-white">${row.search_window_lower_mhz} MHz</td>
            <td class="text-white">${row.search_window_upper_mhz} MHz</td>
            <td>
                <span class="badge bg-primary-subtle text-primary border border-primary-subtle px-2 py-1">
                    [${row.search_window_lower_mhz}, ${row.search_window_upper_mhz}] MHz
                </span>
            </td>
        `;
        tbody.appendChild(tr);
    });

    // Enable download button
    document.getElementById('btnDownloadResultsCsv').disabled = false;
}

// ---------------------------------------------------------------------------
// 4. Setup CSV Download
// ---------------------------------------------------------------------------
function setupDownloadButton() {
    const btn = document.getElementById('btnDownloadResultsCsv');
    btn.addEventListener('click', () => {
        if (!currentBatchResults || currentBatchResults.length === 0) {
            showError('No prediction results available to download.');
            return;
        }

        // Required output columns:
        // chip_id, predicted_fmax_mhz, search_window_lower_mhz, search_window_upper_mhz
        const headers = ['chip_id', 'predicted_fmax_mhz', 'search_window_lower_mhz', 'search_window_upper_mhz'];
        const csvRows = [headers.join(',')];

        currentBatchResults.forEach(row => {
            const rowValues = [
                row.chip_id,
                row.predicted_fmax_mhz,
                row.search_window_lower_mhz,
                row.search_window_upper_mhz
            ];
            csvRows.push(rowValues.join(','));
        });

        const csvContent = csvRows.join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'batch_fmax_predictions.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function showError(message) {
    const errorAlert = document.getElementById('batchErrorAlert');
    const errorMessage = document.getElementById('batchErrorMessage');
    if (errorMessage) errorMessage.textContent = message;
    if (errorAlert) errorAlert.classList.remove('d-none');
}

function showSuccess(message) {
    const successAlert = document.getElementById('batchSuccessAlert');
    const successMessage = document.getElementById('batchSuccessMessage');
    if (successMessage) successMessage.textContent = message;
    if (successAlert) successAlert.classList.remove('d-none');
}

function clearAlerts() {
    const errorAlert = document.getElementById('batchErrorAlert');
    const successAlert = document.getElementById('batchSuccessAlert');
    if (errorAlert) errorAlert.classList.add('d-none');
    if (successAlert) successAlert.classList.add('d-none');
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

