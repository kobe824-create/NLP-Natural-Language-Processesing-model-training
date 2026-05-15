/* ══════════════════════════════════════════════════════════════
   SentimentAI — Dashboard Logic
   Handles: API calls, chart rendering, DOM updates, animations
   ══════════════════════════════════════════════════════════════ */

// ── DOM Refs ──
const textInput      = document.getElementById('textInput');
const charCount      = document.getElementById('charCount');
const analyzeBtn     = document.getElementById('analyzeBtn');
const btnLoader      = document.getElementById('btnLoader');
const resultsContainer = document.getElementById('resultsContainer');
const emptyState     = document.getElementById('emptyState');
const errorToast     = document.getElementById('errorToast');
const errorMsg       = document.getElementById('errorMsg');

// Metric card refs
const sentimentValue  = document.getElementById('sentimentValue');
const confidenceValue = document.getElementById('confidenceValue');
const sentencesValue  = document.getElementById('sentencesValue');
const sentimentCard   = document.getElementById('sentimentCard');

// Section refs
const confidenceBars  = document.getElementById('confidenceBars');
const translationText = document.getElementById('translationText');
const sentenceTableBody = document.getElementById('sentenceTableBody');

let chartInstance = null;

// ── Character Counter ──
textInput.addEventListener('input', () => {
    const len = textInput.value.length;
    charCount.textContent = `${len.toLocaleString()} character${len !== 1 ? 's' : ''}`;
});

// ── Submit on Ctrl/Cmd + Enter ──
textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        analyzeText();
    }
});

// ── Show Error Toast ──
function showError(msg) {
    errorMsg.textContent = msg;
    errorToast.classList.add('show');
    setTimeout(() => errorToast.classList.remove('show'), 5000);
}

// ── Color helpers ──
function sentimentColor(sentiment) {
    const s = sentiment.toLowerCase();
    if (s === 'positive') return 'var(--positive)';
    if (s === 'negative') return 'var(--negative)';
    return 'var(--neutral)';
}

function sentimentClass(sentiment) {
    return sentiment.toLowerCase();
}

// ── Main Analyze Function ──
async function analyzeText() {
    const text = textInput.value.trim();
    if (!text) {
        showError('Please enter some text to analyze.');
        return;
    }

    // Set loading state
    analyzeBtn.classList.add('loading');
    analyzeBtn.disabled = true;

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }

        renderResults(data);

    } catch (err) {
        showError(err.message || 'Failed to connect to the analysis API.');
    } finally {
        analyzeBtn.classList.remove('loading');
        analyzeBtn.disabled = false;
    }
}

// ── Render All Results ──
function renderResults(data) {
    // Hide empty state, show results
    emptyState.style.display = 'none';
    resultsContainer.style.display = 'grid';

    // Force re-animation
    resultsContainer.style.animation = 'none';
    resultsContainer.offsetHeight; // trigger reflow
    resultsContainer.style.animation = '';

    renderMetrics(data);
    renderConfidenceBars(data.confidence_breakdown);
    renderTranslation(data.translation_rw);
    renderChart(data.chart_data);
    renderTable(data.sentence_details);

    // Scroll to results smoothly
    setTimeout(() => {
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 200);
}

// ── Metric Cards ──
function renderMetrics(data) {
    const sent = data.prediction;
    sentimentValue.textContent = sent;
    sentimentValue.className = 'metric-value ' + sentimentClass(sent);

    confidenceValue.textContent = (data.confidence * 100).toFixed(1) + '%';
    sentencesValue.textContent = data.sentences_count;
}

// ── Confidence Breakdown Bars ──
function renderConfidenceBars(breakdown) {
    confidenceBars.innerHTML = '';

    const order = ['Positive', 'Neutral', 'Negative'];
    const sorted = order.filter(k => k in breakdown);
    // Add any keys not in order
    Object.keys(breakdown).forEach(k => { if (!sorted.includes(k)) sorted.push(k); });

    sorted.forEach((label) => {
        const pct = breakdown[label];
        const cls = sentimentClass(label);

        const row = document.createElement('div');
        row.className = 'conf-row';
        row.innerHTML = `
            <span class="conf-label ${cls}">${label}</span>
            <div class="conf-bar-track">
                <div class="conf-bar-fill ${cls}" style="width: 0%"></div>
            </div>
            <span class="conf-pct">${pct}%</span>
        `;
        confidenceBars.appendChild(row);

        // Animate bar after a tiny delay
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                row.querySelector('.conf-bar-fill').style.width = pct + '%';
            });
        });
    });
}

// ── Translation ──
function renderTranslation(text) {
    translationText.textContent = text || '[Translation unavailable]';
}

// ── Sentiment Trajectory Chart ──
function renderChart(chartData) {
    const ctx = document.getElementById('sentimentChart').getContext('2d');

    if (chartInstance) {
        chartInstance.destroy();
    }

    const labels = chartData.map((_, i) => `Sentence ${i + 1}`);

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, 280);
    gradient.addColorStop(0, 'rgba(129, 140, 248, 0.2)');
    gradient.addColorStop(1, 'rgba(129, 140, 248, 0)');

    const positiveColor = '#34d399';
    const negativeColor = '#f87171';
    const neutralColor  = '#fbbf24';

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Sentiment Score',
                data: chartData,
                borderColor: '#818cf8',
                borderWidth: 2.5,
                pointBackgroundColor: chartData.map(val => {
                    if (val > 0.3) return positiveColor;
                    if (val < -0.3) return negativeColor;
                    return neutralColor;
                }),
                pointBorderColor: 'rgba(9,9,11,0.8)',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 8,
                fill: true,
                backgroundColor: gradient,
                tension: 0.35
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(20,20,26,0.95)',
                    titleColor: '#f0f0f5',
                    bodyColor: '#a1a1aa',
                    padding: 14,
                    cornerRadius: 10,
                    borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            const val = context.parsed.y;
                            const label = val > 0.3 ? 'Positive' : (val < -0.3 ? 'Negative' : 'Neutral');
                            return `Score: ${val.toFixed(3)} — ${label}`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    min: -1.2, max: 1.2,
                    grid: {
                        color: 'rgba(255,255,255,0.04)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#6b7280',
                        font: { size: 11, weight: '500', family: 'Inter' },
                        callback: function(value) {
                            if (value === 1)  return 'Positive';
                            if (value === 0)  return 'Neutral';
                            if (value === -1) return 'Negative';
                            return '';
                        }
                    }
                },
                x: {
                    grid: { display: false, drawBorder: false },
                    ticks: {
                        color: '#6b7280',
                        font: { size: 11, family: 'Inter' }
                    }
                }
            },
            animation: {
                duration: 1200,
                easing: 'easeOutQuart'
            }
        }
    });
}

// ── Sentence Detail Table ──
function renderTable(details) {
    sentenceTableBody.innerHTML = '';

    details.forEach((item, idx) => {
        const cls = sentimentClass(item.sentiment);
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td class="td-idx">${idx + 1}</td>
            <td>${escapeHtml(item.text)}</td>
            <td>
                <span class="badge badge-${cls}">
                    <span class="badge-dot"></span>
                    ${item.sentiment}
                </span>
            </td>
            <td class="td-conf">${item.confidence}%</td>
            <td class="td-translation">${escapeHtml(item.translation)}</td>
        `;
        sentenceTableBody.appendChild(tr);
    });
}

// ── Utility: Escape HTML ──
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
