/**
 * AI - Irrigation Optimization System
 * Frontend Logic
 */

// Global state
let appState = {
    filesUploaded: false,
    modelTrained: false,
    predictionMade: false,
    charts: {
        soilMoisture: null,
        irrigation: null,
        featureImportance: null
    }
};

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('Irrigad AI - Frontend Initialized');

    // Setup event listeners
    setupEventListeners();
});

/**
 * Setup all event listeners
 */
function setupEventListeners() {
    // Upload form
    document.getElementById('upload-form').addEventListener('submit', handleUpload);

    // Train button
    document.getElementById('train-btn').addEventListener('click', handleTrain);

    // Predict button
    document.getElementById('predict-btn').addEventListener('click', handlePredict);
}

/**
 * Handle file upload
 */
async function handleUpload(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);
    const uploadBtn = document.getElementById('upload-btn');
    const statusDiv = document.getElementById('upload-status');

    // Check if all files are selected
    const requiredFiles = ['sm', 'meteo', 'valve', 'previsioni'];
    for (const file of requiredFiles) {
        if (!formData.get(file)) {
            showStatus('danger', `Seleziona il file ${file} prima di caricare.`);
            return;
        }
    }

    try {
        // Disable button and show loading
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Caricamento...';
        showStatus('info', 'Caricamento e validazione file in corso...');

        // Upload files
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            showStatus('success', result.message);

            // Show summary
            if (result.summary) {
                let summaryHtml = '<div class="mt-2"><strong>Riepilogo dati caricati:</strong><ul class="mb-0">';
                for (const [key, value] of Object.entries(result.summary)) {
                    summaryHtml += `<li>${key}: ${value.days} giorni (${value.start_date} - ${value.end_date})</li>`;
                }
                summaryHtml += '</ul></div>';
                statusDiv.querySelector('.alert').innerHTML += summaryHtml;
            }

            // Show cleaning report
            if (result.cleaning_report) {
                displayCleaningReport(result.cleaning_report);
            }

            // Update state
            appState.filesUploaded = true;

            // Enable train button
            document.getElementById('train-btn').disabled = false;

        } else {
            showStatus('danger', 'Errore: ' + (result.error || result.errors.join(', ')));
        }

    } catch (error) {
        showStatus('danger', 'Errore di connessione: ' + error.message);
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="bi bi-upload me-2"></i>Carica e Valida File';
    }
}

/**
 * Handle model training
 */
async function handleTrain() {
    const trainBtn = document.getElementById('train-btn');
    const statusDiv = document.getElementById('upload-status');

    try {
        // Disable button and show loading
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Training...';
        showStatus('info', 'Training modello ML in corso... Questo potrebbe richiedere alcuni secondi.');

        // Train model
        const response = await fetch('/train', {
            method: 'POST'
        });

        const result = await response.json();

        if (result.success) {
            const modelType = result.model_type;
            const metrics = result.training_metrics;

            let message = `Modello trainato con successo!<br>`;
            message += `<strong>Tipo:</strong> ${modelType}<br>`;
            message += `<strong>R² Score:</strong> ${metrics.r2_score.toFixed(3)}<br>`;
            message += `<strong>MAE:</strong> ${metrics.mae_m3.toFixed(1)} m³<br>`;
            message += `<strong>Training samples:</strong> ${metrics.training_samples}`;

            showStatus('success', message);

            // Update state
            appState.modelTrained = true;

            // Enable predict button
            document.getElementById('predict-btn').disabled = false;

        } else {
            showStatus('danger', 'Errore durante training: ' + result.error);
        }

    } catch (error) {
        showStatus('danger', 'Errore di connessione: ' + error.message);
    } finally {
        trainBtn.disabled = false;
        trainBtn.innerHTML = '<i class="bi bi-cpu me-2"></i>Traina Modello ML';
    }
}

/**
 * Handle prediction request
 */
async function handlePredict() {
    const predictBtn = document.getElementById('predict-btn');
    const statusDiv = document.getElementById('upload-status');

    try {
        // Disable button and show loading
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Calcolo...';
        showStatus('info', 'Calcolo raccomandazione irrigazione...');

        // Get prediction
        const response = await fetch('/predict', {
            method: 'POST'
        });

        const result = await response.json();

        if (result.success) {
            showStatus('success', 'Raccomandazione calcolata con successo!');

            // Display results
            displayPredictionResults(result);

            // Load and display charts
            loadAnalytics();

            // Update state
            appState.predictionMade = true;

            // Show results section
            document.getElementById('results-section').style.display = 'block';

            // Scroll to results
            document.getElementById('results-section').scrollIntoView({
                behavior: 'smooth'
            });

        } else {
            showStatus('danger', 'Errore durante predizione: ' + result.error);
        }

    } catch (error) {
        showStatus('danger', 'Errore di connessione: ' + error.message);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<i class="bi bi-lightning-charge me-2"></i>Calcola Raccomandazione';
    }
}

/**
 * Display prediction results
 */
function displayPredictionResults(result) {
    const prediction = result.prediction;
    const currentSituation = result.current_situation;
    const forecast = result.forecast_3days;
    const explanation = result.decision_explanation;

    // Main recommendation
    document.getElementById('water-recommendation').innerHTML =
        `<strong>${prediction.water_mm.toFixed(1)}</strong> mm`;
    document.getElementById('water-m3').textContent =
        `(${prediction.water_m3.toFixed(0)} m³ totali)`;

    // Badges
    const confidenceBadge = document.getElementById('confidence-badge');
    confidenceBadge.textContent = `Confidenza: ${prediction.confidence}`;
    confidenceBadge.className = 'badge fs-6 px-3 py-2 ' + getConfidenceBadgeClass(prediction.confidence);

    const priorityBadge = document.getElementById('priority-badge');
    priorityBadge.textContent = `Priorità: ${prediction.priority}`;
    priorityBadge.className = 'badge fs-6 px-3 py-2 ' + getPriorityBadgeClass(prediction.priority);

    // Recommendation text
    document.getElementById('recommendation-text').textContent = prediction.recommendation;

    // Key metrics
    document.getElementById('soil-moisture-value').textContent =
        `${currentSituation.soil_moisture_avg}%`;

    document.getElementById('rain-forecast-value').textContent =
        `${forecast.rain_expected_mm} mm`;

    document.getElementById('et0-forecast-value').textContent =
        `${forecast.et0_expected_mm} mm`;

    document.getElementById('temp-min-value').textContent =
        `${forecast.temp_min_c}°C`;

    const frostWarning = document.getElementById('frost-warning');
    if (forecast.frost_risk) {
        frostWarning.innerHTML = '<i class="bi bi-exclamation-triangle-fill text-danger"></i> Rischio gelo';
        frostWarning.className = 'small text-danger fw-bold';
    } else {
        frostWarning.textContent = 'Nessun rischio';
        frostWarning.className = 'small text-muted';
    }

    // Detailed explanation
    if (explanation) {
        displayExplanation(explanation);
    }

    // Model info
    document.getElementById('model-type').textContent = result.model_used;

    const metrics = result.training_metrics;
    let metricsHtml = `R² Score: ${metrics.r2_score.toFixed(3)} | `;
    metricsHtml += `MAE: ${metrics.mae_m3.toFixed(1)} m³ | `;
    metricsHtml += `RMSE: ${metrics.rmse_m3.toFixed(1)} m³<br>`;
    metricsHtml += `Training samples: ${metrics.training_samples}`;
    document.getElementById('model-metrics').innerHTML = metricsHtml;

    // Feature importance
    if (result.feature_importance) {
        renderFeatureImportanceChart(result.feature_importance);
    }

    // Debug information
    if (result.debug) {
        displayDebugInfo(result.debug);
    }
}

/**
 * Display detailed explanation
 */
function displayExplanation(explanation) {
    // Summary
    document.getElementById('explanation-summary').textContent = explanation.summary;

    // Steps
    const stepsList = document.getElementById('explanation-steps');
    stepsList.innerHTML = '';
    if (explanation.steps) {
        explanation.steps.forEach(step => {
            const li = document.createElement('li');
            li.textContent = step;
            stepsList.appendChild(li);
        });
    }

    // Key factors
    const factorsList = document.getElementById('key-factors-list');
    factorsList.innerHTML = '';
    if (explanation.key_factors) {
        explanation.key_factors.forEach(factor => {
            const factorDiv = document.createElement('div');
            factorDiv.className = 'factor-card';
            factorDiv.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <div class="factor-name">${factor.description || factor.factor}</div>
                        <div class="factor-value">${factor.value}</div>
                        ${factor.note ? `<small class="text-muted">${factor.note}</small>` : ''}
                    </div>
                    <div class="text-end">
                        <span class="factor-impact ${factor.impact}">${factor.impact}</span>
                        <div><small class="text-muted">Peso: ${factor.weight}</small></div>
                    </div>
                </div>
            `;
            factorsList.appendChild(factorDiv);
        });
    }

    // Warnings
    const warningsList = document.getElementById('warnings-list');
    warningsList.innerHTML = '';
    if (explanation.warnings) {
        explanation.warnings.forEach(warning => {
            const warningDiv = document.createElement('div');

            if (warning.includes('ATTENZIONE') || warning.includes('URGENTE')) {
                warningDiv.className = 'warning-box';
            } else if (warning.includes('ℹ️')) {
                warningDiv.className = 'info-box';
            } else {
                warningDiv.className = 'info-box';
            }

            warningDiv.textContent = warning;
            warningsList.appendChild(warningDiv);
        });
    }
}

/**
 * Load and display analytics charts
 */
async function loadAnalytics() {
    try {
        const response = await fetch('/analytics');
        const result = await response.json();

        if (result.success) {
            // Soil moisture chart
            renderSoilMoistureChart(result.soil_moisture_trend);

            // Irrigation history chart
            renderIrrigationChart(result.irrigation_history);

            // Feature importance chart (if available)
            if (result.feature_importance && result.feature_importance.length > 0) {
                renderFeatureImportanceChart(result.feature_importance);
            }
        }

    } catch (error) {
        console.error('Error loading analytics:', error);
    }
}

/**
 * Render soil moisture trend chart
 */
function renderSoilMoistureChart(data) {
    const ctx = document.getElementById('soil-moisture-chart').getContext('2d');

    // Destroy existing chart
    if (appState.charts.soilMoisture) {
        appState.charts.soilMoisture.destroy();
    }

    appState.charts.soilMoisture = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'SM1',
                    data: data.sm1,
                    borderColor: '#2196F3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    tension: 0.3
                },
                {
                    label: 'SM2',
                    data: data.sm2,
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    tension: 0.3
                },
                {
                    label: 'Media',
                    data: data.avg,
                    borderColor: '#FF9800',
                    backgroundColor: 'rgba(255, 152, 0, 0.1)',
                    borderWidth: 2,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'top'
                },
                title: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'Umidità Suolo (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Data'
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                }
            }
        }
    });
}

/**
 * Render irrigation history chart
 */
function renderIrrigationChart(data) {
    const ctx = document.getElementById('irrigation-chart').getContext('2d');

    // Destroy existing chart
    if (appState.charts.irrigation) {
        appState.charts.irrigation.destroy();
    }

    appState.charts.irrigation = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.dates,
            datasets: [{
                label: 'Irrigazione (mm)',
                data: data.values,
                backgroundColor: '#2196F3',
                borderColor: '#1976D2',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Irrigazione (mm)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Data'
                    },
                    ticks: {
                        maxTicksLimit: 10
                    }
                }
            }
        }
    });
}

/**
 * Render feature importance chart
 */
function renderFeatureImportanceChart(data) {
    const ctx = document.getElementById('feature-importance-chart').getContext('2d');

    // Destroy existing chart
    if (appState.charts.featureImportance) {
        appState.charts.featureImportance.destroy();
    }

    // Take top 5
    const topFeatures = data.slice(0, 5);

    appState.charts.featureImportance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: topFeatures.map(f => f.feature),
            datasets: [{
                label: 'Importance',
                data: topFeatures.map(f => f.importance),
                backgroundColor: '#4CAF50',
                borderColor: '#388E3C',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Importance'
                    }
                }
            }
        }
    });
}

/**
 * Show status message
 */
function showStatus(type, message) {
    const statusDiv = document.getElementById('upload-status');
    const alertClass = `alert alert-${type}`;

    statusDiv.innerHTML = `
        <div class="${alertClass} fade-in" role="alert">
            ${message}
        </div>
    `;
}

/**
 * Get confidence badge class
 */
function getConfidenceBadgeClass(confidence) {
    switch(confidence) {
        case 'ALTA':
            return 'bg-success';
        case 'MEDIA':
            return 'bg-warning text-dark';
        case 'BASSA':
            return 'bg-danger';
        default:
            return 'bg-secondary';
    }
}

/**
 * Get priority badge class
 */
function getPriorityBadgeClass(priority) {
    switch(priority) {
        case 'ALTA':
            return 'bg-danger';
        case 'MEDIA':
            return 'bg-warning text-dark';
        case 'BASSA':
            return 'bg-success';
        default:
            return 'bg-secondary';
    }
}

/**
 * Display data cleaning report
 */
function displayCleaningReport(report) {
    // Populate valve data
    document.getElementById('valve-outliers').textContent = report.valve.outliers;
    document.getElementById('variance-reduction').textContent = report.valve.variance_reduction;
    document.getElementById('valve-negative').textContent = report.valve.negative_values || 0;
    document.getElementById('valve-high').textContent = report.valve.too_high_values || 0;

    // Populate soil moisture data
    document.getElementById('sm-outliers').textContent = report.sm.outliers;
    const smSensors = report.sm.sensors_cleaned || [];
    document.getElementById('sm-sensors').textContent = smSensors.length > 0 ? smSensors.join(', ') : 'nessuno';

    // Populate meteo data
    document.getElementById('meteo-outliers').textContent = report.meteo.outliers;
    const meteoSensors = report.meteo.sensors_cleaned || [];
    document.getElementById('meteo-sensors').textContent = meteoSensors.length > 0 ? meteoSensors.join(', ') : 'nessuno';

    // Total outliers
    document.getElementById('total-outliers').textContent = report.total_outliers_removed;

    // Show warning if many outliers
    if (report.total_outliers_removed > 50) {
        document.getElementById('cleaning-warning').style.display = 'block';
    } else {
        document.getElementById('cleaning-warning').style.display = 'none';
    }

    // Show the cleaning report section
    document.getElementById('cleaning-report').style.display = 'block';
}

/**
 * Display debug information
 */
function displayDebugInfo(debug) {
    if (!debug) return;

    // Show debug panel
    const debugPanel = document.getElementById('debug-panel');
    debugPanel.style.display = 'block';

    // Sanity status
    const sanityStatus = document.getElementById('sanity-status');
    if (debug.sanity_check) {
        sanityStatus.innerHTML = '<span class="badge bg-success">✓ PASS</span>';
    } else {
        sanityStatus.innerHTML = '<span class="badge bg-danger">✗ FAIL - Predizione non plausibile</span>';
    }

    // Warnings
    if (debug.warnings && debug.warnings.length > 0) {
        document.getElementById('debug-warnings-section').style.display = 'block';
        const warningsList = document.getElementById('debug-warnings');
        warningsList.innerHTML = '';
        debug.warnings.forEach(w => {
            const emoji = w.level === 'critical' ? '🔴' : (w.level === 'warning' ? '🟡' : 'ℹ️');
            const li = document.createElement('li');
            li.innerHTML = `${emoji} <strong>[${w.level.toUpperCase()}]</strong> ${w.message}`;
            if (w.level === 'critical') {
                li.style.color = '#C62828';
                li.style.fontWeight = 'bold';
            }
            warningsList.appendChild(li);
        });
    } else {
        document.getElementById('debug-warnings-section').style.display = 'none';
    }

    // Recommendations
    if (debug.recommendations && debug.recommendations.length > 0) {
        document.getElementById('debug-recommendations-section').style.display = 'block';
        const recList = document.getElementById('debug-recommendations');
        recList.innerHTML = '';
        debug.recommendations.forEach(r => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${r.action}:</strong> ${r.details}`;
            recList.appendChild(li);
        });
    } else {
        document.getElementById('debug-recommendations-section').style.display = 'none';
    }

    // Detailed info (for collapsed sections)
    if (debug.checks) {
        document.getElementById('debug-features').textContent = JSON.stringify(debug.checks.features || {}, null, 2);
        document.getElementById('debug-forecast').textContent = JSON.stringify(debug.checks.forecast || {}, null, 2);
        document.getElementById('debug-full').textContent = JSON.stringify(debug, null, 2);
    }
}
