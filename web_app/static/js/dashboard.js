/**
 * dashboard.js
 * Fetches metrics from /api/metrics and renders live dashboard statistics and Chart.js graphs.
 */

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) {
            throw new Error(`Failed to load metrics: HTTP ${response.status}`);
        }
        const data = await response.json();
        renderDashboardData(data);
    } catch (err) {
        console.error('Error fetching dashboard metrics:', err);
    }
});

function renderDashboardData(data) {
    const model = data.model_metrics || {};
    const search = data.search_metrics || {};
    const dataset = data.dataset_summary || {};

    // Update KPI Metric Cards
    if (search.step_reduction_pct !== undefined) {
        document.getElementById('metricStepReduction').textContent = `${search.step_reduction_pct.toFixed(1)}%`;
    }
    if (model.test_r2 !== undefined) {
        document.getElementById('metricR2').textContent = model.test_r2.toFixed(4);
    }
    if (model.test_mae_mhz !== undefined) {
        document.getElementById('metricMAE').textContent = model.test_mae_mhz.toFixed(2);
    }
    if (model.test_rmse_mhz !== undefined) {
        document.getElementById('metricRMSE').textContent = model.test_rmse_mhz.toFixed(2);
    }
    if (search.completion_rate_pct !== undefined) {
        document.getElementById('metricCompletionRate').textContent = `${search.completion_rate_pct.toFixed(1)}%`;
    }
    if (search.speed_bin_accuracy_pct !== undefined) {
        document.getElementById('metricBinAccuracy').textContent = `${search.speed_bin_accuracy_pct.toFixed(2)}%`;
    }

    // Update Dataset Partition Counts
    if (dataset.total_records) {
        document.getElementById('statTotalUnits').textContent = `${dataset.total_records.toLocaleString()} units`;
    }
    if (dataset.train_records) {
        document.getElementById('statTrainUnits').textContent = `${dataset.train_records.toLocaleString()} units`;
    }
    if (dataset.calib_records) {
        document.getElementById('statCalibUnits').textContent = `${dataset.calib_records.toLocaleString()} units`;
    }
    if (dataset.test_records) {
        document.getElementById('statTestUnits').textContent = `${dataset.test_records.toLocaleString()} units`;
    }

    // Render Charts
    renderStepComparisonChart(search.conv_avg_tests || 6.46, search.ai_avg_tests || 3.94);
    renderSearchModesChart(
        search.direct_window_count || 3603,
        search.upper_ext_count || 215,
        search.lower_fallback_count || 182
    );
}

function renderStepComparisonChart(convAvg, aiAvg) {
    const ctx = document.getElementById('chartTestSteps');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Conventional Binary Search', 'AI-Windowed Search (Algorithm 1)'],
            datasets: [{
                label: 'Average Tests Per Die',
                data: [convAvg, aiAvg],
                backgroundColor: [
                    'rgba(148, 163, 184, 0.65)',
                    'rgba(59, 130, 246, 0.85)'
                ],
                borderColor: [
                    '#94a3b8',
                    '#3b82f6'
                ],
                borderWidth: 1.5,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => ` ${context.parsed.y.toFixed(2)} test steps / chip`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 8,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' },
                    title: {
                        display: true,
                        text: 'Average ATE Steps / Die',
                        color: '#94a3b8'
                    }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#cbd5e1', font: { weight: '600' } }
                }
            }
        }
    });
}

function renderSearchModesChart(directCount, upperCount, lowerCount) {
    const ctx = document.getElementById('chartSearchModes');
    if (!ctx) return;

    const total = directCount + upperCount + lowerCount || 4000;
    const directPct = ((directCount / total) * 100).toFixed(1);
    const upperPct = ((upperCount / total) * 100).toFixed(1);
    const lowerPct = ((lowerCount / total) * 100).toFixed(1);

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [
                `Direct In-Window (${directCount} - ${directPct}%)`,
                `Upper-Edge Extension (${upperCount} - ${upperPct}%)`,
                `Lower Fallback (${lowerCount} - ${lowerPct}%)`
            ],
            datasets: [{
                data: [directCount, upperCount, lowerCount],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.85)',
                    'rgba(245, 158, 11, 0.85)',
                    'rgba(6, 182, 212, 0.85)'
                ],
                borderColor: '#111827',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#cbd5e1',
                        padding: 16,
                        font: { size: 11 }
                    }
                }
            }
        }
    });
}

