/**
 * search.js
 * Controller for Page 3: Search Comparison
 * Supports both automatic curated chip loading and manual editing of
 * Predicted Fmax and Actual (Ground Truth) Fmax values.
 */

let curatedUnitsCache = [];
let currentCuratedUnit = null;

document.addEventListener('DOMContentLoaded', () => {
    loadBenchmarkMetrics();
    loadCuratedUnits();
    setupEventListeners();
});

// ---------------------------------------------------------------------------
// 1. Load Dataset-Level Benchmark Metrics (4 Metrics Only)
// ---------------------------------------------------------------------------
async function loadBenchmarkMetrics() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) throw new Error('Failed to fetch dataset metrics');
        const data = await response.json();
        const search = data.search_metrics || data.search_comparison || {};

        if (search.conv_avg_tests !== undefined) {
            document.getElementById('panelConvAvgTests').textContent = search.conv_avg_tests.toFixed(2);
        }
        if (search.ai_avg_tests !== undefined) {
            document.getElementById('panelAiAvgTests').textContent = search.ai_avg_tests.toFixed(2);
        }
        if (search.step_reduction_pct !== undefined) {
            document.getElementById('panelReductionPct').textContent = `${search.step_reduction_pct.toFixed(2)}%`;
        }
        if (search.speed_bin_accuracy_pct !== undefined) {
            document.getElementById('panelBinAccuracy').textContent = `${Number(search.speed_bin_accuracy_pct).toFixed(2)}%`;
        }
    } catch (err) {
        console.warn('Could not load benchmark metrics:', err);
    }
}

// ---------------------------------------------------------------------------
// 2. Load Curated Test Chips
// ---------------------------------------------------------------------------
async function loadCuratedUnits() {
    const select = document.getElementById('curatedUnitSelect');
    try {
        const response = await fetch('/api/sample-units');
        if (!response.ok) throw new Error('Failed to fetch sample units');
        const data = await response.json();
        curatedUnitsCache = data.samples || [];

        select.innerHTML = '<option value="" disabled selected>-- Select a chip --</option>';
        curatedUnitsCache.forEach((sample, idx) => {
            const opt = document.createElement('option');
            opt.value = idx;

            let behaviorLabel = 'Direct Search';
            if (sample.unit_id === 'U02024') behaviorLabel = 'Upper Extension';
            else if (sample.unit_id === 'U02008') behaviorLabel = 'Lower Fallback';
            else if (sample.expected_behavior) behaviorLabel = sample.expected_behavior;

            opt.textContent = `${sample.unit_id} (${behaviorLabel})`;
            select.appendChild(opt);
        });

        // Default to first chip (U02001)
        if (curatedUnitsCache.length > 0) {
            select.selectedIndex = 1;
            applyCuratedUnit(0);
            executeSearch();
        }
    } catch (err) {
        console.warn('Could not load curated units:', err);
        select.innerHTML = '<option value="" disabled>Error loading chips</option>';
    }
}

function applyCuratedUnit(index) {
    if (index < 0 || index >= curatedUnitsCache.length) return;
    const sample = curatedUnitsCache[index];
    currentCuratedUnit = sample;

    clearError();
    document.getElementById('inputPredFmax').value = sample.fmax_pred_mhz.toFixed(1);
    document.getElementById('inputTrueFmax').value = sample.fmax_true_mhz.toFixed(1);
}

// ---------------------------------------------------------------------------
// Error Display Helpers
// ---------------------------------------------------------------------------
function showError(message) {
    const errorAlert = document.getElementById('searchErrorAlert');
    const errorMessage = document.getElementById('searchErrorMessage');
    if (errorMessage) errorMessage.textContent = message;
    if (errorAlert) errorAlert.classList.remove('d-none');
}

function clearError() {
    const errorAlert = document.getElementById('searchErrorAlert');
    if (errorAlert) errorAlert.classList.add('d-none');
}

// ---------------------------------------------------------------------------
// 3. Event Listeners
// ---------------------------------------------------------------------------
function setupEventListeners() {
    const select = document.getElementById('curatedUnitSelect');
    select.addEventListener('change', (e) => {
        const idx = parseInt(e.target.value, 10);
        if (!isNaN(idx)) {
            applyCuratedUnit(idx);
            executeSearch();
        }
    });

    const predInput = document.getElementById('inputPredFmax');
    const trueInput = document.getElementById('inputTrueFmax');

    if (predInput) {
        predInput.addEventListener('input', clearError);
    }
    if (trueInput) {
        trueInput.addEventListener('input', clearError);
    }

    const form = document.getElementById('searchConfigForm');
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        executeSearch();
    });
}

// ---------------------------------------------------------------------------
// 4. Run Comparative Search via /api/search
// ---------------------------------------------------------------------------
async function executeSearch() {
    const predInput = document.getElementById('inputPredFmax');
    const trueInput = document.getElementById('inputTrueFmax');

    const predStr = predInput ? predInput.value.trim() : '';
    const trueStr = trueInput ? trueInput.value.trim() : '';

    // Validation
    if (!predStr) {
        showError('Please provide a Predicted Fmax value.');
        if (predInput) predInput.focus();
        return;
    }

    const predVal = parseFloat(predStr);
    if (isNaN(predVal) || predVal <= 0) {
        showError('Predicted Fmax must be a valid positive number.');
        if (predInput) predInput.focus();
        return;
    }

    if (!trueStr) {
        showError('Please provide an Actual (Ground Truth) Fmax value.');
        if (trueInput) trueInput.focus();
        return;
    }

    const trueVal = parseFloat(trueStr);
    if (isNaN(trueVal) || trueVal <= 0) {
        showError('Actual (Ground Truth) Fmax must be a valid positive number.');
        if (trueInput) trueInput.focus();
        return;
    }

    clearError();

    const btn = document.getElementById('btnRunSearch');
    const spinner = document.getElementById('searchSpinner');
    const playIcon = document.getElementById('searchPlayIcon');

    btn.disabled = true;
    spinner.classList.remove('d-none');
    playIcon.classList.add('d-none');

    try {
        const payload = {
            unit_id: currentCuratedUnit ? currentCuratedUnit.unit_id : 'Manual',
            pred_fmax_mhz: predVal,
            window_half_mhz: 39.80,
            fmax_true_mhz: trueVal,
            noise_sigma: 4.0,
            random_seed: 42,
            is_synthetic: !!currentCuratedUnit,
            features: currentCuratedUnit ? currentCuratedUnit.features : null,
            simulate_ood_drift: false
        };

        const res = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.message || 'Search execution failed');
        }

        const resData = await res.json();
        const data = resData.data || resData;
        renderResults(data);

    } catch (err) {
        console.error('Error during search execution:', err);
        showError(`Error executing search: ${err.message}`);
    } finally {
        btn.disabled = false;
        spinner.classList.add('d-none');
        playIcon.classList.remove('d-none');
    }
}

// ---------------------------------------------------------------------------
// 5. Render Comparison Cards, Banner, and Logs
// ---------------------------------------------------------------------------
function renderResults(data) {
    const conv = data.conventional_search || {};
    const ai = data.ai_windowed_search || {};

    const convTests = conv.total_tests !== undefined ? conv.total_tests : (conv.n_tests || 0);
    const aiTests = ai.total_tests !== undefined ? ai.total_tests : (ai.n_tests || 0);

    // --- Section 2: Cards ---
    // Conventional Card
    document.getElementById('cardConvTests').textContent = `${convTests} tests`;
    document.getElementById('cardConvFmax').textContent = conv.final_fmax_mhz !== null ? `${conv.final_fmax_mhz} MHz` : 'None';
    document.getElementById('cardConvBin').textContent = conv.speed_bin ? `Bin ${conv.speed_bin}` : '-';

    // AI Card
    document.getElementById('cardAiTests').textContent = `${aiTests} tests`;
    document.getElementById('cardAiFmax').textContent = ai.final_fmax_mhz !== null ? `${ai.final_fmax_mhz} MHz` : 'None';
    document.getElementById('cardAiBin').textContent = ai.speed_bin ? `Bin ${ai.speed_bin}` : '-';

    // Method badge
    const badge = document.getElementById('cardAiMethodBadge');
    const searchType = ai.search_type || 'window_direct';
    if (searchType === 'window_direct') {
        badge.textContent = 'Direct Search';
        badge.className = 'badge bg-primary-subtle text-primary border border-primary-subtle';
    } else if (searchType === 'upper_extension') {
        badge.textContent = 'Upper Extension';
        badge.className = 'badge bg-warning-subtle text-warning border border-warning-subtle';
    } else if (searchType === 'lower_fallback') {
        badge.textContent = 'Lower Fallback';
        badge.className = 'badge bg-danger-subtle text-danger border border-danger-subtle';
    } else {
        badge.textContent = 'Full Grid Search';
        badge.className = 'badge bg-secondary-subtle text-light border border-secondary-subtle';
    }

    // --- Section 2: Savings / Recovery Banner ---
    const banner = document.getElementById('savingsBanner');
    const mainText = document.getElementById('savingsMainText');
    const subText = document.getElementById('savingsSubText');

    if (aiTests < convTests) {
        const saved = convTests - aiTests;
        const pct = Math.round((saved / convTests) * 1000) / 10;
        mainText.textContent = `${saved} Tests Saved (${pct}% Reduction)`;
        mainText.className = 'fs-4 fw-bold text-teal';
        subText.textContent = `AI narrowed the search range, reducing required ATE test steps from ${convTests} to ${aiTests}.`;
        banner.style.borderColor = 'rgba(16, 185, 129, 0.4)';
    } else if (aiTests > convTests) {
        const extra = aiTests - convTests;
        const pct = Math.round((extra / convTests) * 1000) / 10;
        mainText.textContent = `+${extra} Additional Recovery Tests`;
        mainText.className = 'fs-4 fw-bold text-warning';
        subText.textContent = `Chip required search boundary extension (${pct}% recovery overhead); safely located true frequency in ${aiTests} tests.`;
        banner.style.borderColor = 'rgba(245, 158, 11, 0.5)';
    } else {
        mainText.textContent = 'Equal Test Steps (0 Saved)';
        mainText.className = 'fs-4 fw-bold text-white';
        subText.textContent = `Both conventional and AI methods completed in ${convTests} tests.`;
        banner.style.borderColor = 'var(--border-color)';
    }

    // --- Section 3: Execution Logs ---
    renderExecutionLogs(conv.steps || conv.trace || [], ai.steps || ai.trace || []);
}

// ---------------------------------------------------------------------------
// 6. Render Execution Logs (3 Columns: Step, Frequency Tested, Result)
// ---------------------------------------------------------------------------
function renderExecutionLogs(convSteps, aiSteps) {
    document.getElementById('badgeConvTotalSteps').textContent = `${convSteps.length} Steps`;
    document.getElementById('badgeAiTotalSteps').textContent = `${aiSteps.length} Steps`;

    // Conventional Log
    const convBody = document.getElementById('tableConvTraceBody');
    convBody.innerHTML = '';
    convSteps.forEach((step) => {
        const tr = document.createElement('tr');
        const stepNum = step.step || step.step_num || 0;
        const badge = step.passed
            ? `<span class="badge bg-teal-subtle text-teal px-2 py-1"><i class="bi bi-check-lg me-1"></i>PASS</span>`
            : `<span class="badge bg-rose-subtle text-rose px-2 py-1"><i class="bi bi-x-lg me-1"></i>FAIL</span>`;

        tr.innerHTML = `
            <td class="text-secondary fw-semibold">Step ${stepNum}</td>
            <td class="text-white fw-bold">${step.freq_mhz} MHz</td>
            <td>${badge}</td>
        `;
        convBody.appendChild(tr);
    });

    // AI-Windowed Log
    const aiBody = document.getElementById('tableAiTraceBody');
    aiBody.innerHTML = '';
    aiSteps.forEach((step) => {
        const tr = document.createElement('tr');
        const stepNum = step.step || step.step_num || 0;
        const badge = step.passed
            ? `<span class="badge bg-teal-subtle text-teal px-2 py-1"><i class="bi bi-check-lg me-1"></i>PASS</span>`
            : `<span class="badge bg-rose-subtle text-rose px-2 py-1"><i class="bi bi-x-lg me-1"></i>FAIL</span>`;

        tr.innerHTML = `
            <td class="text-secondary fw-semibold">Step ${stepNum}</td>
            <td class="text-white fw-bold">${step.freq_mhz} MHz</td>
            <td>${badge}</td>
        `;
        aiBody.appendChild(tr);
    });
}
