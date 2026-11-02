/**
 * results.js
 * Handles paginated fetching and rendering of benchmark results from /api/results.
 */

let currentPage = 1;
let currentLimit = 25;
let currentFilter = 'all';
let totalPages = 1;

document.addEventListener('DOMContentLoaded', () => {
    setupResultsControls();
    loadResults();
});

function setupResultsControls() {
    // Filter change
    document.getElementById('filterSearchType').addEventListener('change', (e) => {
        currentFilter = e.target.value;
        currentPage = 1;
        loadResults();
    });

    // Limit change
    document.getElementById('selectPageLimit').addEventListener('change', (e) => {
        currentLimit = parseInt(e.target.value, 10);
        currentPage = 1;
        loadResults();
    });

    // Pagination buttons
    document.getElementById('btnPrevPage').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            loadResults();
        }
    });

    document.getElementById('btnNextPage').addEventListener('click', () => {
        if (currentPage < totalPages) {
            currentPage++;
            loadResults();
        }
    });
}

async function loadResults() {
    const tbody = document.getElementById('resultsTableBody');
    tbody.innerHTML = '<tr><td colspan="9" class="text-center py-4 text-muted"><span class="spinner-border spinner-border-sm me-2"></span>Loading benchmark records...</td></tr>';

    try {
        const url = `/api/results?page=${currentPage}&limit=${currentLimit}&search_type=${currentFilter}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        renderTableRows(data);
    } catch (err) {
        tbody.innerHTML = `<tr><td colspan="9" class="text-center py-4 text-danger">Error loading records: ${err.message}</td></tr>`;
    }
}

function renderTableRows(data) {
    const records = data.records || [];
    const tbody = document.getElementById('resultsTableBody');
    tbody.innerHTML = '';

    currentPage = data.page || 1;
    totalPages = data.total_pages || 1;
    const totalRecords = data.total_records || 0;

    // Update count indicator
    const startIdx = totalRecords > 0 ? (currentPage - 1) * currentLimit + 1 : 0;
    const endIdx = Math.min(currentPage * currentLimit, totalRecords);
    document.getElementById('resultsCountText').textContent = `Showing ${startIdx}-${endIdx} of ${totalRecords}`;
    document.getElementById('pageIndicatorText').textContent = `Page ${currentPage} of ${totalPages}`;

    // Enable/disable pagination buttons
    document.getElementById('btnPrevPage').disabled = currentPage <= 1;
    document.getElementById('btnNextPage').disabled = currentPage >= totalPages;

    if (records.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" class="text-center py-4 text-muted">No records found matching filter.</td></tr>';
        return;
    }

    records.forEach((row) => {
        const tr = document.createElement('tr');

        // Search type badge
        let typeBadge = '';
        if (row.search_type === 'direct_window' || row.search_type === 'window_direct') {
            typeBadge = '<span class="badge badge-success-subtle">Direct Window</span>';
        } else if (row.search_type === 'upper_extension') {
            typeBadge = '<span class="badge badge-warning-subtle">Upper Extension</span>';
        } else if (row.search_type === 'lower_fallback') {
            typeBadge = '<span class="badge badge-tech" style="color: #06b6d4; border-color: rgba(6, 182, 212, 0.3);">Lower Fallback</span>';
        } else {
            typeBadge = `<span class="badge badge-tech">${row.search_type}</span>`;
        }

        // Speed bin indicator
        const isBinMatch = row.speed_bin_match !== undefined ? row.speed_bin_match : true;
        const binBadge = isBinMatch
            ? '<span class="text-success"><i class="bi bi-check-circle-fill me-1"></i>Match</span>'
            : '<span class="text-warning"><i class="bi bi-exclamation-triangle-fill me-1"></i>Mismatch</span>';

        // Preserved reduction calculation: reduction = ((conv_tests - ai_tests) / conv_tests) * 100
        let reductionVal = 0.0;
        const conv = Number(row.conv_tests);
        const ai = Number(row.ai_tests);

        if (!isNaN(conv) && !isNaN(ai) && conv > 0) {
            reductionVal = ((conv - ai) / conv) * 100.0;
        } else if (row.step_reduction_pct !== undefined && row.step_reduction_pct !== null && !isNaN(Number(row.step_reduction_pct))) {
            reductionVal = Number(row.step_reduction_pct);
        }

        const redDisplay = `${reductionVal.toFixed(1)}%`;

        // High-contrast readable colors:
        // Positive: Green (#10b981 / text-success)
        // Zero: Neutral white (#f8fafc / text-white)
        // Negative: Red (#ef4444 / text-danger)
        let redClass = 'reduction-zero text-white';
        if (reductionVal > 0.0001) {
            redClass = 'reduction-positive text-success';
        } else if (reductionVal < -0.0001) {
            redClass = 'reduction-negative text-danger';
        }

        tr.innerHTML = `
            <td class="fw-semibold text-white">${row.unit_id}</td>
            <td>${row.fmax_true_mhz !== undefined ? row.fmax_true_mhz.toFixed(1) : '--'} MHz</td>
            <td>${row.pred_fmax_mhz !== undefined ? row.pred_fmax_mhz.toFixed(1) : '--'} MHz</td>
            <td class="small text-secondary">[${row.window_low !== undefined ? row.window_low.toFixed(0) : '--'}, ${row.window_high !== undefined ? row.window_high.toFixed(0) : '--'}]</td>
            <td>${row.conv_tests ?? '--'}</td>
            <td class="fw-bold text-success">${row.ai_tests ?? '--'}</td>
            <td class="${redClass} fw-bold">${redDisplay}</td>
            <td>${typeBadge}</td>
            <td>${binBadge}</td>
        `;

        tbody.appendChild(tr);
    });
}

