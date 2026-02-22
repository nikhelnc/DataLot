from jinja2 import Template
from app.db.models import Analysis


class HTMLReporter:
    def generate(self, analysis: Analysis) -> str:
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lotto Analysis Report - {{ analysis.name }}</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1, h2, h3 { color: #333; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .warning-box { background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .limits-box { background: #f8d7da; border: 1px solid #dc3545; padding: 15px; border-radius: 5px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f5f5f5; font-weight: bold; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        .metric-label { font-weight: bold; color: #666; }
        .metric-value { font-size: 1.2em; color: #333; }
        @media print { .no-print { display: none; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>Lotto Analysis Report</h1>
        <p><strong>Analysis:</strong> {{ analysis.name }}</p>
        <p><strong>Date:</strong> {{ analysis.created_at.strftime('%Y-%m-%d %H:%M:%S') }}</p>
        <p><strong>Code Version:</strong> {{ analysis.code_version }}</p>
        <p><strong>Dataset Hash:</strong> {{ analysis.dataset_hash }}</p>
    </div>

    <div class="limits-box">
        <h3>⚠️ Limites & Interprétation (OBLIGATOIRE)</h3>
        <ul>
            <li><strong>Hypothèse nulle (H0):</strong> Une loterie saine suit un processus uniforme i.i.d. sans remise.</li>
            <li><strong>Baseline uniforme:</strong> Toute "tendance" doit être comparée au hasard pur (baseline M0).</li>
            <li><strong>Corrections multi-tests:</strong> Les p-values sont corrigées (FDR) pour éviter les faux positifs.</li>
            <li><strong>Incertitude:</strong> Les modèles probabilistes incluent intervalles de confiance et calibration.</li>
            <li><strong>Aucune garantie:</strong> Cette analyse est à valeur scientifique uniquement. Aucun modèle ne garantit de gain.</li>
            <li><strong>Biais possibles:</strong> Fenêtre temporelle, multiples tests, données insuffisantes, changements de règles.</li>
        </ul>
    </div>

    {% if results.summary %}
    <div class="section">
        <h2>Summary</h2>
        <p>{{ results.summary }}</p>
    </div>
    {% endif %}

    {% if results.warnings %}
    <div class="warning-box">
        <h3>⚠️ Warnings</h3>
        <ul>
        {% for warning in results.warnings %}
            <li>{{ warning }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}

    {% if results.metrics %}
    <div class="section">
        <h2>Descriptive Metrics</h2>
        
        {% if results.metrics.frequencies %}
        <h3>Frequency Analysis</h3>
        <div class="metric">
            <span class="metric-label">Entropy:</span>
            <span class="metric-value">{{ "%.3f"|format(results.metrics.frequencies.entropy) }}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Max Entropy:</span>
            <span class="metric-value">{{ "%.3f"|format(results.metrics.frequencies.max_entropy) }}</span>
        </div>
        <div class="metric">
            <span class="metric-label">KL Divergence:</span>
            <span class="metric-value">{{ "%.4f"|format(results.metrics.frequencies.kl_divergence) }}</span>
        </div>
        {% endif %}

        {% if results.metrics.quality %}
        <h3>Dataset Quality</h3>
        <div class="metric">
            <span class="metric-label">Total Draws:</span>
            <span class="metric-value">{{ results.metrics.quality.total_draws }}</span>
        </div>
        {% endif %}
    </div>
    {% endif %}

    {% if results.tests %}
    <div class="section">
        <h2>Randomness Tests</h2>
        
        {% if results.tests.uniformity %}
        <h3>Uniformity Tests</h3>
        <table>
            <tr>
                <th>Test</th>
                <th>Statistic</th>
                <th>P-Value</th>
                <th>Interpretation</th>
            </tr>
            {% for test_name, test_data in results.tests.uniformity.items() %}
            {% if test_data.statistic is defined %}
            <tr>
                <td>{{ test_name }}</td>
                <td>{{ "%.4f"|format(test_data.statistic) }}</td>
                <td>{{ "%.4f"|format(test_data.p_value) }}</td>
                <td>{{ test_data.interpretation }}</td>
            </tr>
            {% endif %}
            {% endfor %}
        </table>
        {% endif %}

        {% if results.tests.fdr_correction %}
        <h3>FDR Correction (Benjamini-Hochberg)</h3>
        <table>
            <tr>
                <th>Test</th>
                <th>Corrected P-Value</th>
                <th>Rejected</th>
            </tr>
            {% for test_name, pval in results.tests.fdr_correction.corrected_pvalues.items() %}
            <tr>
                <td>{{ test_name }}</td>
                <td>{{ "%.4f"|format(pval) }}</td>
                <td>{{ results.tests.fdr_correction.rejected[test_name] }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
    {% endif %}

    {% if results.anomalies %}
    <div class="section">
        <h2>Anomaly Detection</h2>
        
        {% if results.anomalies.drift %}
        <h3>Drift Detection</h3>
        <div class="metric">
            <span class="metric-label">Max PSI:</span>
            <span class="metric-value">{{ "%.3f"|format(results.anomalies.drift.max_psi) }}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Max z(KL):</span>
            <span class="metric-value">{{ "%.2f"|format(results.anomalies.drift.max_z_kl) }}</span>
        </div>
        {% endif %}

        {% if results.anomalies.change_points and results.anomalies.change_points.count > 0 %}
        <h3>Change Points ({{ results.anomalies.change_points.count }})</h3>
        <p>Method: {{ results.anomalies.change_points.method }}</p>
        <ul>
        {% for date in results.anomalies.change_points.change_points[:10] %}
            <li>{{ date }}</li>
        {% endfor %}
        </ul>
        {% endif %}

        {% if results.anomalies.outliers and results.anomalies.outliers.count > 0 %}
        <h3>Outliers ({{ results.anomalies.outliers.count }})</h3>
        <p>Method: {{ results.anomalies.outliers.method }}, Threshold: {{ results.anomalies.outliers.threshold }}</p>
        <table>
            <tr>
                <th>Date</th>
                <th>Numbers</th>
                <th>Sum</th>
                <th>Z-Score</th>
            </tr>
            {% for outlier in results.anomalies.outliers.outliers[:10] %}
            <tr>
                <td>{{ outlier.date }}</td>
                <td>{{ outlier.numbers }}</td>
                <td>{{ outlier.sum }}</td>
                <td>{{ "%.2f"|format(outlier.z_score) }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
    {% endif %}

    {% if results.probabilities %}
    <div class="section">
        <h2>Probability Models</h2>
        
        {% for model_name, model_data in results.probabilities.items() %}
        <h3>{{ model_name }} - {{ model_data.method_id }}</h3>
        
        {% if model_data.warnings %}
        <div class="warning-box">
            <strong>Warnings:</strong>
            <ul>
            {% for warning in model_data.warnings %}
                <li>{{ warning }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}

        {% if model_data.evaluation %}
        <h4>Evaluation (Walk-Forward)</h4>
        <div class="metric">
            <span class="metric-label">Brier Score:</span>
            <span class="metric-value">{{ "%.6f"|format(model_data.evaluation.brier_score) }}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Baseline Brier:</span>
            <span class="metric-value">{{ "%.6f"|format(model_data.evaluation.baseline_brier) }}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Lift:</span>
            <span class="metric-value">{{ "%.6f"|format(model_data.evaluation.lift) }}</span>
        </div>
        <div class="metric">
            <span class="metric-label">ECE:</span>
            <span class="metric-value">{{ "%.6f"|format(model_data.evaluation.ece) }}</span>
        </div>
        {% endif %}

        {% if model_data.top_numbers %}
        <h4>Top Numbers</h4>
        <p>{{ model_data.top_numbers }}</p>
        {% endif %}
        {% endfor %}
    </div>
    {% endif %}

    <div class="section no-print">
        <p><em>Report generated by Lotto Analyzer {{ analysis.code_version }}</em></p>
        <button onclick="window.print()">Print Report</button>
    </div>
</body>
</html>
        """
        
        template = Template(template_str)
        return template.render(analysis=analysis, results=analysis.results_json or {})
