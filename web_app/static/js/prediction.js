/**
 * prediction.js
 * Handles Fmax prediction form, curated sample preloading, input validation,
 * API communication with /api/predict, and search space visualization.
 */

let sampleUnitsCache = [];
let currentSelectedSample = null;
let lastPredictionResult = null;

document.addEventListener('DOMContentLoaded', () => {
    // Explicitly guarantee all 15 inputs permit arbitrary floating-point precision
    const featureNames = [
        'ro_01', 'ro_02', 'ro_03', 'ro_04', 'ro_05',
        'ro_06', 'ro_07', 'ro_08', 'ro_09', 'ro_10',
        'iddq_norm', 'vmin_norm', 'die_x', 'die_y', 'radius_sq'
    ];
    for (const name of featureNames) {
        const el = document.getElementById(name);
        if (el) {
            el.setAttribute('step', 'any');
            el.step = 'any';
        }
    }

    loadSampleUnits();
    setupEventListeners();
});

async function loadSampleUnits() {
    try {
        const response = await fetch('/api/sample-units');
        if (!response.ok) throw new Error('Failed to load sample units');
        const data = await response.json();
        sampleUnitsCache = data.samples || [];

        const select = document.getElementById('sampleUnitSelect');
        sampleUnitsCache.forEach((sample, idx) => {
            const opt = document.createElement('option');
            opt.value = idx;
            const behavior = sample.expected_behavior || sample.category || 'Curated Sample';
            opt.textContent = `${sample.unit_id} (${behavior})`;
            select.appendChild(opt);
        });
    } catch (err) {
        console.warn('Could not populate sample units dropdown:', err);
    }
}

function setupEventListeners() {
    // Curated sample selector
    const select = document.getElementById('sampleUnitSelect');
    select.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === '') {
            currentSelectedSample = null;
            return;
        }
        const sample = sampleUnitsCache[parseInt(val, 10)];
        if (sample) {
            currentSelectedSample = sample;
            populateFormWithSample(sample);
        }
    });

    // Auto calculate radius squared when die_x or die_y changes
    const dieX = document.getElementById('die_x');
    const dieY = document.getElementById('die_y');
    const radiusSq = document.getElementById('radius_sq');

    function updateRadius() {
        const x = parseFloat(dieX.value);
        const y = parseFloat(dieY.value);
        if (!isNaN(x) && !isNaN(y)) {
            radiusSq.value = (x * x + y * y).toFixed(4);
        }
    }
    dieX.addEventListener('input', updateRadius);
    dieY.addEventListener('input', updateRadius);

    // Reset button
    document.getElementById('btnResetForm').addEventListener('click', () => {
        document.getElementById('predictionForm').reset();
        currentSelectedSample = null;
        select.value = '';
        document.getElementById('predictionResultCard').classList.add('d-none');
        document.getElementById('predictionEmptyState').classList.remove('d-none');
    });

    // Form submission
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await handlePredictionSubmit();
    });
}

function populateFormWithSample(sample) {
    const features = sample.features || {};
    for (const [key, val] of Object.entries(features)) {
        const input = document.getElementById(key);
        if (input) {
            input.value = typeof val === 'number' ? val.toFixed(4) : val;
        }
    }
}

async function handlePredictionSubmit() {
    const btn = document.getElementById('btnPredict');
    const spinner = document.getElementById('btnPredictSpinner');

    // Collect 15 features
    const payload = {};
    const featureNames = [
        'ro_01', 'ro_02', 'ro_03', 'ro_04', 'ro_05',
        'ro_06', 'ro_07', 'ro_08', 'ro_09', 'ro_10',
        'iddq_norm', 'vmin_norm', 'die_x', 'die_y', 'radius_sq'
    ];

    for (const name of featureNames) {
        const el = document.getElementById(name);
        if (!el || el.value.trim() === '') {
            alert(`Please provide a valid value for ${name}`);
            el?.focus();
            return;
        }
        const val = parseFloat(el.value);
        if (isNaN(val)) {
            alert(`Value for ${name} must be numeric.`);
            el.focus();
            return;
        }
        payload[name] = val;
    }

    // Set UI loading state
    btn.disabled = true;
    spinner.classList.remove('d-none');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const resData = await response.json();
        if (!response.ok || resData.status !== 'success') {
            throw new Error(resData.message || 'Prediction failed');
        }

        lastPredictionResult = resData.data;
        displayPredictionResult(resData.data);
    } catch (err) {
        alert(`Error executing model prediction: ${err.message}`);
    } finally {
        btn.disabled = false;
        spinner.classList.add('d-none');
    }
}

function validatePredictionResponse(res) {
    if (!res || typeof res !== 'object') {
        throw new Error('Invalid prediction response: expected a data object.');
    }
    const predVal = res.predicted_fmax_mhz !== undefined ? res.predicted_fmax_mhz : res.pred_fmax_mhz;
    if (predVal === undefined || typeof predVal !== 'number' || isNaN(predVal)) {
        throw new Error("Missing or invalid 'predicted_fmax_mhz' in API response.");
    }

    const requiredNumerics = ['window_low_mhz', 'window_high_mhz', 'window_half_mhz', 'sigma_calib_mhz'];
    for (const key of requiredNumerics) {
        if (res[key] === undefined || typeof res[key] !== 'number' || isNaN(res[key])) {
            throw new Error(`Missing or invalid '${key}' in API response.`);
        }
    }
}

function displayPredictionResult(res) {
    // Validate response schema before updating DOM
    validatePredictionResponse(res);

    document.getElementById('predictionEmptyState').classList.add('d-none');
    const resultCard = document.getElementById('predictionResultCard');
    resultCard.classList.remove('d-none');

    const predictedFmax = res.predicted_fmax_mhz !== undefined ? res.predicted_fmax_mhz : res.pred_fmax_mhz;
    const windowLow = res.window_low_mhz;
    const windowHigh = res.window_high_mhz;
    const windowHalf = res.window_half_mhz;
    const sigmaCalib = res.sigma_calib_mhz;

    // Derived values with fallback computation
    const gridCenter = res.grid_center_mhz !== undefined ? res.grid_center_mhz : Math.round(predictedFmax / 10.0) * 10.0;
    const candidatePoints = res.candidate_points !== undefined ? res.candidate_points : Math.round((windowHigh - windowLow) / 10.0) + 1;
    const spaceReduction = res.space_reduction_pct !== undefined ? res.space_reduction_pct : (1.0 - (windowHigh - windowLow) / 800.0) * 100.0;

    document.getElementById('resPredFmax').textContent = predictedFmax.toFixed(1);
    document.getElementById('resGridCenter').textContent = gridCenter.toFixed(1);
    document.getElementById('resWindowBounds').textContent = `[${windowLow.toFixed(0)}, ${windowHigh.toFixed(0)}] MHz`;
    document.getElementById('resWindowHalf').textContent = `± ${windowHalf.toFixed(2)} MHz (k=1.5σ)`;
    document.getElementById('resCandidatePoints').textContent = `${candidatePoints} points`;
    document.getElementById('resSpaceReduction').textContent = `${spaceReduction.toFixed(1)}%`;

    const badge = document.getElementById('badgeUnitId');
    if (currentSelectedSample) {
        badge.textContent = `${currentSelectedSample.unit_id} (True: ${currentSelectedSample.fmax_true_mhz.toFixed(1)} MHz)`;
    } else {
        badge.textContent = 'Custom Die Input';
    }
}
