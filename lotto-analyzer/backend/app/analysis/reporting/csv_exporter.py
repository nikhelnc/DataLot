import csv
from io import StringIO
from typing import Dict, Any

from app.db.models import Analysis


class CSVExporter:
    def export(self, analysis: Analysis) -> str:
        output = StringIO()
        writer = csv.writer(output)
        
        writer.writerow(["Analysis Export"])
        writer.writerow(["Analysis ID", str(analysis.id)])
        writer.writerow(["Game ID", str(analysis.game_id)])
        writer.writerow(["Analysis Name", analysis.name])
        writer.writerow(["Code Version", analysis.code_version])
        writer.writerow(["Dataset Hash", analysis.dataset_hash])
        writer.writerow(["Created At", analysis.created_at.isoformat()])
        writer.writerow([])
        
        results = analysis.results_json or {}
        
        if "metrics" in results and "frequencies" in results["metrics"]:
            writer.writerow(["Number Frequencies"])
            writer.writerow(["Number", "Count", "Z-Score"])
            
            freqs = results["metrics"]["frequencies"]
            counts = freqs.get("counts", {})
            z_scores = freqs.get("z_scores", {})
            
            for num in sorted(counts.keys(), key=int):
                writer.writerow([num, counts[num], f"{z_scores.get(num, 0):.3f}"])
            writer.writerow([])
        
        if "tests" in results and "uniformity" in results["tests"]:
            writer.writerow(["Uniformity Tests"])
            writer.writerow(["Test", "Statistic", "P-Value", "Interpretation"])
            
            for test_name, test_data in results["tests"]["uniformity"].items():
                if isinstance(test_data, dict) and "statistic" in test_data:
                    writer.writerow([
                        test_name,
                        f"{test_data['statistic']:.4f}",
                        f"{test_data['p_value']:.4f}",
                        test_data.get("interpretation", "")
                    ])
            writer.writerow([])
        
        if "anomalies" in results and "outliers" in results["anomalies"]:
            outliers = results["anomalies"]["outliers"]
            if outliers.get("count", 0) > 0:
                writer.writerow(["Outliers"])
                writer.writerow(["Date", "Numbers", "Sum", "Z-Score"])
                
                for outlier in outliers.get("outliers", [])[:20]:
                    writer.writerow([
                        outlier["date"],
                        str(outlier["numbers"]),
                        outlier["sum"],
                        f"{outlier['z_score']:.3f}"
                    ])
                writer.writerow([])
        
        if "probabilities" in results:
            for model_name, model_data in results["probabilities"].items():
                if isinstance(model_data, dict) and "number_probs" in model_data:
                    writer.writerow([f"Model {model_name} - Probabilities"])
                    writer.writerow(["Number", "Probability"])
                    
                    probs = model_data["number_probs"]
                    for num in sorted(probs.keys(), key=int):
                        writer.writerow([num, f"{probs[num]:.6f}"])
                    
                    if "evaluation" in model_data:
                        eval_data = model_data["evaluation"]
                        writer.writerow([])
                        writer.writerow(["Evaluation Metrics"])
                        writer.writerow(["Brier Score", f"{eval_data.get('brier_score', 0):.6f}"])
                        writer.writerow(["Baseline Brier", f"{eval_data.get('baseline_brier', 0):.6f}"])
                        writer.writerow(["Lift", f"{eval_data.get('lift', 0):.6f}"])
                        writer.writerow(["ECE", f"{eval_data.get('ece', 0):.6f}"])
                    writer.writerow([])
        
        return output.getvalue()
