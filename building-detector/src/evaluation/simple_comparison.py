#!/usr/bin/env python3
"""
Simple Percentage Comparison Module

This module provides clear percentage-based comparison between ground truth and model results.
"""

from typing import Dict, Any

class PercentageComparisonAnalyzer:
    """Simple analyzer for percentage-based comparison"""
    
    def __init__(self):
        pass
    
    def generate_simple_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a simple, clear percentage summary
        
        Args:
            metrics: Dictionary containing evaluation metrics
            
        Returns:
            Formatted string with percentage comparison
        """
        
        total_ground_truth = metrics['total_osm_buildings']
        total_detections = metrics['total_detections']
        successfully_matched = metrics['true_positives']
        missed_buildings = metrics['false_negatives']
        false_detections = metrics['false_positives']
        
        # Calculate percentages
        detection_rate = (successfully_matched / total_ground_truth * 100) if total_ground_truth > 0 else 0
        miss_rate = (missed_buildings / total_ground_truth * 100) if total_ground_truth > 0 else 0
        precision_pct = (successfully_matched / total_detections * 100) if total_detections > 0 else 0
        
        summary = f"""
🏠 GROUND TRUTH vs MODEL COMPARISON
{'='*50}
📊 Ground Truth (OSM Buildings): {total_ground_truth} buildings
🎯 Model Detections: {total_detections} detections
✅ Successfully Matched: {successfully_matched} buildings

📈 ACCURACY METRICS:
{'='*30}
✅ Detection Rate: {detection_rate:.1f}% ({successfully_matched}/{total_ground_truth})
❌ Miss Rate: {miss_rate:.1f}% ({missed_buildings}/{total_ground_truth})
🎯 Precision: {precision_pct:.1f}% ({successfully_matched}/{total_detections})
📊 Overall Accuracy: {detection_rate:.1f}%

📋 DETAILED BREAKDOWN:
{'='*25}
🟢 Buildings Successfully Detected: {successfully_matched}
🔴 Buildings Missed by Model: {missed_buildings}
🔵 False Detections: {false_detections}

💡 INTERPRETATION:
{'='*20}"""
        
        if detection_rate >= 95:
            summary += f"\n🌟 EXCELLENT: Model achieves {detection_rate:.1f}% detection rate!"
        elif detection_rate >= 90:
            summary += f"\n👍 GOOD: Model achieves {detection_rate:.1f}% detection rate."
        elif detection_rate >= 80:
            summary += f"\n⚠️  FAIR: Model achieves {detection_rate:.1f}% detection rate. Consider improvement."
        else:
            summary += f"\n❌ POOR: Model only achieves {detection_rate:.1f}% detection rate. Needs significant improvement."
        
        if missed_buildings > 0:
            summary += f"\n🔍 {missed_buildings} buildings were missed - check red polygons on map for analysis."
        else:
            summary += f"\n🎯 Perfect detection - no buildings missed!"
            
        if false_detections > 0:
            summary += f"\n⚠️  {false_detections} false detections found - model may be over-detecting."
        else:
            summary += f"\n✅ No false detections - excellent precision!"
        
        return summary
    
    def generate_simple_text_export(self, metrics: Dict[str, Any], output_path: str) -> None:
        """
        Export simple summary to text file
        
        Args:
            metrics: Dictionary containing evaluation metrics
            output_path: Path to save the summary file
        """
        summary = self.generate_simple_summary(metrics)
        
        # Add timestamp and metadata
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        full_summary = f"""Building Detection Evaluation Summary
Generated: {timestamp}
Threshold: 5.0 meters

{summary}

📁 FILES GENERATED:
- Interactive Map: improved_colors_evaluation_map.html
- This Summary: evaluation_summary.txt

🔗 For detailed analysis, open the interactive map to see:
- 🔵 Blue circles: Model detections
- 🟢 Green polygons: Successfully detected buildings
- 🔴 Red polygons: Missed buildings
"""
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_summary)
            print(f"📄 Summary exported to: {output_path}")
        except Exception as e:
            print(f"❌ Error exporting summary: {e}")
