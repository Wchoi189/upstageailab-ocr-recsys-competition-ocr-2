#!/usr/bin/env python3
"""
Compliance Trend Tracking System

This script provides comprehensive trend analysis and tracking for compliance metrics:
- Historical data analysis and visualization
- Trend calculation and prediction
- Performance metrics and KPIs
- Compliance forecasting
- Integration with monitoring and alerting systems

Usage:
    python compliance_trend_tracker.py --analyze-trends
    python compliance_trend_tracker.py --generate-report
    python compliance_trend_tracker.py --forecast-compliance
    python compliance_trend_tracker.py --export-data
"""

import argparse
import json
import sqlite3
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass
class TrendDataPoint:
    """Single data point in trend analysis"""

    date: str
    compliance_rate: float
    total_files: int
    total_issues: int
    issues_by_type: dict[str, int]
    auto_fixes_applied: int
    manual_fixes_needed: int


@dataclass
class TrendAnalysis:
    """Comprehensive trend analysis results"""

    period: str
    start_date: str
    end_date: str
    data_points: list[TrendDataPoint]

    # Statistical metrics
    avg_compliance_rate: float
    min_compliance_rate: float
    max_compliance_rate: float
    compliance_std_dev: float

    # Trend metrics
    overall_trend: float  # Linear regression slope
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_strength: str  # 'strong', 'moderate', 'weak'

    # Performance metrics
    total_improvement: float
    improvement_rate: float
    volatility: float

    # Predictions
    next_week_prediction: float
    next_month_prediction: float
    confidence_level: float


class ComplianceTrendTracker:
    """Comprehensive compliance trend tracking system"""

    def __init__(self, db_path: str = "compliance_monitoring.db"):
        self.db_path = db_path
        self.trend_cache = {}

        # Trend analysis parameters
        self.trend_params = {
            "min_data_points": 3,
            "trend_threshold": 0.01,  # 1% change threshold
            "volatility_threshold": 0.05,  # 5% volatility threshold
            "prediction_days": 30,
            "confidence_threshold": 0.7,
        }

    def analyze_trends(self, days: int = 30) -> TrendAnalysis:
        """Analyze compliance trends for specified period"""
        print(f"📊 Analyzing compliance trends for {days} days...")

        # Get historical data
        data_points = self._get_historical_data(days)

        if len(data_points) < self.trend_params["min_data_points"]:
            raise ValueError(
                f"Insufficient data points: {len(data_points)} (minimum: {self.trend_params['min_data_points']})"
            )

        # Calculate statistical metrics
        compliance_rates = [dp.compliance_rate for dp in data_points]
        avg_compliance_rate = statistics.mean(compliance_rates)
        min_compliance_rate = min(compliance_rates)
        max_compliance_rate = max(compliance_rates)
        compliance_std_dev = (
            statistics.stdev(compliance_rates) if len(compliance_rates) > 1 else 0.0
        )

        # Calculate trend metrics
        overall_trend = self._calculate_linear_trend(data_points)
        trend_direction = self._determine_trend_direction(overall_trend)
        trend_strength = self._determine_trend_strength(
            overall_trend, compliance_std_dev
        )

        # Calculate performance metrics
        total_improvement = compliance_rates[-1] - compliance_rates[0]
        improvement_rate = (
            total_improvement / len(data_points) if len(data_points) > 0 else 0.0
        )
        volatility = (
            compliance_std_dev / avg_compliance_rate if avg_compliance_rate > 0 else 0.0
        )

        # Generate predictions
        next_week_prediction, next_month_prediction, confidence_level = (
            self._generate_predictions(data_points)
        )

        # Create trend analysis
        analysis = TrendAnalysis(
            period=f"{days} days",
            start_date=data_points[0].date,
            end_date=data_points[-1].date,
            data_points=data_points,
            avg_compliance_rate=avg_compliance_rate,
            min_compliance_rate=min_compliance_rate,
            max_compliance_rate=max_compliance_rate,
            compliance_std_dev=compliance_std_dev,
            overall_trend=overall_trend,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            total_improvement=total_improvement,
            improvement_rate=improvement_rate,
            volatility=volatility,
            next_week_prediction=next_week_prediction,
            next_month_prediction=next_month_prediction,
            confidence_level=confidence_level,
        )

        # Cache analysis
        cache_key = f"trend_{days}_{data_points[-1].date}"
        self.trend_cache[cache_key] = analysis

        return analysis

    def _get_historical_data(self, days: int) -> list[TrendDataPoint]:
        """Get historical compliance data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT date, compliance_rate, total_files, total_issues,
                   issues_by_type, auto_fixes_applied, manual_fixes_needed
            FROM daily_reports
            WHERE date >= date('now', '-{days} days')
            ORDER BY date ASC
        """)

        data_points = []
        for row in cursor.fetchall():
            data_points.append(
                TrendDataPoint(
                    date=row[0],
                    compliance_rate=row[1],
                    total_files=row[2],
                    total_issues=row[3],
                    issues_by_type=json.loads(row[4]) if row[4] else {},
                    auto_fixes_applied=row[5],
                    manual_fixes_needed=row[6],
                )
            )

        conn.close()
        return data_points

    def _calculate_linear_trend(self, data_points: list[TrendDataPoint]) -> float:
        """Calculate linear trend using least squares regression"""
        if len(data_points) < 2:
            return 0.0

        # Convert dates to numeric values (days since first date)
        first_date = datetime.fromisoformat(data_points[0].date)
        x_values = []
        y_values = []

        for dp in data_points:
            current_date = datetime.fromisoformat(dp.date)
            days_diff = (current_date - first_date).days
            x_values.append(days_diff)
            y_values.append(dp.compliance_rate)

        # Calculate linear regression slope
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values, strict=False))
        sum_x2 = sum(x * x for x in x_values)

        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope

    def _determine_trend_direction(self, trend: float) -> str:
        """Determine trend direction based on slope"""
        threshold = self.trend_params["trend_threshold"]

        if trend > threshold:
            return "improving"
        elif trend < -threshold:
            return "declining"
        else:
            return "stable"

    def _determine_trend_strength(self, trend: float, std_dev: float) -> str:
        """Determine trend strength based on slope and volatility"""
        abs_trend = abs(trend)

        if abs_trend > 0.02:  # 2% per day
            return "strong"
        elif abs_trend > 0.005:  # 0.5% per day
            return "moderate"
        else:
            return "weak"

    def _generate_predictions(
        self, data_points: list[TrendDataPoint]
    ) -> tuple[float, float, float]:
        """Generate compliance predictions using trend analysis"""
        if len(data_points) < 3:
            return 0.0, 0.0, 0.0

        # Use linear trend for prediction
        trend = self._calculate_linear_trend(data_points)
        last_rate = data_points[-1].compliance_rate

        # Predict next week (7 days)
        next_week_prediction = last_rate + (trend * 7)
        next_week_prediction = max(
            0.0, min(1.0, next_week_prediction)
        )  # Clamp to [0, 1]

        # Predict next month (30 days)
        next_month_prediction = last_rate + (trend * 30)
        next_month_prediction = max(
            0.0, min(1.0, next_month_prediction)
        )  # Clamp to [0, 1]

        # Calculate confidence based on data consistency
        compliance_rates = [dp.compliance_rate for dp in data_points]
        std_dev = (
            statistics.stdev(compliance_rates) if len(compliance_rates) > 1 else 0.0
        )
        avg_rate = statistics.mean(compliance_rates)

        # Confidence decreases with higher volatility
        volatility_factor = std_dev / avg_rate if avg_rate > 0 else 1.0
        confidence_level = max(0.0, min(1.0, 1.0 - volatility_factor))

        return next_week_prediction, next_month_prediction, confidence_level

    def generate_trend_report(self, analysis: TrendAnalysis) -> str:
        """Generate comprehensive trend analysis report"""
        report = []
        report.append("=" * 60)
        report.append("COMPLIANCE TREND ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Analysis Period: {analysis.period}")
        report.append(f"Date Range: {analysis.start_date} to {analysis.end_date}")
        report.append(f"Data Points: {len(analysis.data_points)}")
        report.append("")

        # Summary metrics
        report.append("📊 SUMMARY METRICS")
        report.append("-" * 30)
        report.append(f"Average Compliance Rate: {analysis.avg_compliance_rate:.1%}")
        report.append(f"Minimum Compliance Rate: {analysis.min_compliance_rate:.1%}")
        report.append(f"Maximum Compliance Rate: {analysis.max_compliance_rate:.1%}")
        report.append(f"Standard Deviation: {analysis.compliance_std_dev:.3f}")
        report.append("")

        # Trend analysis
        report.append("📈 TREND ANALYSIS")
        report.append("-" * 30)
        report.append(f"Overall Trend: {analysis.overall_trend:+.4f} per day")
        report.append(f"Trend Direction: {analysis.trend_direction.title()}")
        report.append(f"Trend Strength: {analysis.trend_strength.title()}")
        report.append(f"Total Improvement: {analysis.total_improvement:+.1%}")
        report.append(f"Improvement Rate: {analysis.improvement_rate:+.4f} per day")
        report.append(f"Volatility: {analysis.volatility:.1%}")
        report.append("")

        # Predictions
        report.append("🔮 PREDICTIONS")
        report.append("-" * 30)
        report.append(f"Next Week Prediction: {analysis.next_week_prediction:.1%}")
        report.append(f"Next Month Prediction: {analysis.next_month_prediction:.1%}")
        report.append(f"Confidence Level: {analysis.confidence_level:.1%}")

        if analysis.confidence_level < self.trend_params["confidence_threshold"]:
            report.append("⚠️  Low confidence prediction - consider more data points")
        report.append("")

        # Detailed data
        report.append("📋 DETAILED DATA")
        report.append("-" * 30)
        for dp in analysis.data_points[-10:]:  # Show last 10 data points
            report.append(
                f"{dp.date}: {dp.compliance_rate:.1%} ({dp.total_files} files, {dp.total_issues} issues)"
            )
        report.append("")

        # Recommendations
        recommendations = self._generate_trend_recommendations(analysis)
        if recommendations:
            report.append("💡 RECOMMENDATIONS")
            report.append("-" * 30)
            for rec in recommendations:
                report.append(f"• {rec}")
            report.append("")

        return "\n".join(report)

    def _generate_trend_recommendations(self, analysis: TrendAnalysis) -> list[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []

        # Trend-based recommendations
        if analysis.trend_direction == "declining":
            recommendations.append(
                "🚨 Declining trend detected. Investigate recent changes and consider immediate action."
            )
        elif analysis.trend_direction == "improving":
            recommendations.append(
                "📈 Positive trend detected. Continue current practices to maintain improvement."
            )
        elif analysis.trend_direction == "stable":
            recommendations.append(
                "📊 Stable trend detected. Consider optimization to accelerate improvement."
            )

        # Volatility recommendations
        if analysis.volatility > 0.1:  # 10% volatility
            recommendations.append(
                "📊 High volatility detected. Consider stabilizing processes and reducing variability."
            )

        # Prediction recommendations
        if analysis.next_week_prediction < 0.9:
            recommendations.append(
                "⚠️  Predicted compliance below 90% next week. Plan proactive fixes."
            )

        if analysis.next_month_prediction < 0.95:
            recommendations.append(
                "🎯 Predicted compliance below target next month. Implement improvement strategies."
            )

        # Confidence recommendations
        if analysis.confidence_level < 0.7:
            recommendations.append(
                "📊 Low prediction confidence. Collect more data points for better forecasting."
            )

        # Performance recommendations
        if analysis.total_improvement < 0:
            recommendations.append(
                "📉 Overall decline detected. Review and address root causes."
            )
        elif analysis.total_improvement > 0.05:  # 5% improvement
            recommendations.append(
                "🎉 Significant improvement achieved. Document successful practices."
            )

        return recommendations

    def export_trend_data(self, analysis: TrendAnalysis, format: str = "json") -> str:
        """Export trend analysis data in specified format"""
        if format == "json":
            return json.dumps(asdict(analysis), indent=2)
        elif format == "csv":
            return self._export_csv(analysis)
        elif format == "html":
            return self._export_html(analysis)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_csv(self, analysis: TrendAnalysis) -> str:
        """Export trend data as CSV"""
        csv_lines = []
        csv_lines.append(
            "date,compliance_rate,total_files,total_issues,auto_fixes_applied,manual_fixes_needed"
        )

        for dp in analysis.data_points:
            csv_lines.append(
                f"{dp.date},{dp.compliance_rate},{dp.total_files},{dp.total_issues},{dp.auto_fixes_applied},{dp.manual_fixes_needed}"
            )

        return "\n".join(csv_lines)

    def _export_html(self, analysis: TrendAnalysis) -> str:
        """Export trend data as HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Compliance Trend Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 10px 0; }}
        .trend-positive {{ color: green; }}
        .trend-negative {{ color: red; }}
        .trend-stable {{ color: blue; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Compliance Trend Analysis Report</h1>
        <p>Period: {analysis.period} ({analysis.start_date} to {analysis.end_date})</p>
    </div>

    <h2>Summary Metrics</h2>
    <div class="metric">Average Compliance Rate: {analysis.avg_compliance_rate:.1%}</div>
    <div class="metric">Minimum Compliance Rate: {analysis.min_compliance_rate:.1%}</div>
    <div class="metric">Maximum Compliance Rate: {analysis.max_compliance_rate:.1%}</div>
    <div class="metric">Standard Deviation: {analysis.compliance_std_dev:.3f}</div>

    <h2>Trend Analysis</h2>
    <div class="metric">Overall Trend: <span class="trend-{"positive" if analysis.overall_trend > 0 else "negative" if analysis.overall_trend < 0 else "stable"}">{analysis.overall_trend:+.4f} per day</span></div>
    <div class="metric">Trend Direction: {analysis.trend_direction.title()}</div>
    <div class="metric">Trend Strength: {analysis.trend_strength.title()}</div>
    <div class="metric">Total Improvement: {analysis.total_improvement:+.1%}</div>
    <div class="metric">Volatility: {analysis.volatility:.1%}</div>

    <h2>Predictions</h2>
    <div class="metric">Next Week Prediction: {analysis.next_week_prediction:.1%}</div>
    <div class="metric">Next Month Prediction: {analysis.next_month_prediction:.1%}</div>
    <div class="metric">Confidence Level: {analysis.confidence_level:.1%}</div>

    <h2>Historical Data</h2>
    <table>
        <tr>
            <th>Date</th>
            <th>Compliance Rate</th>
            <th>Total Files</th>
            <th>Total Issues</th>
            <th>Auto Fixes Applied</th>
            <th>Manual Fixes Needed</th>
        </tr>
"""

        for dp in analysis.data_points:
            html += f"""
        <tr>
            <td>{dp.date}</td>
            <td>{dp.compliance_rate:.1%}</td>
            <td>{dp.total_files}</td>
            <td>{dp.total_issues}</td>
            <td>{dp.auto_fixes_applied}</td>
            <td>{dp.manual_fixes_needed}</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""
        return html

    def get_performance_metrics(self, days: int = 30) -> dict[str, Any]:
        """Get comprehensive performance metrics"""
        analysis = self.analyze_trends(days)

        # Calculate additional performance metrics
        data_points = analysis.data_points

        # Efficiency metrics
        total_auto_fixes = sum(dp.auto_fixes_applied for dp in data_points)
        total_manual_fixes = sum(dp.manual_fixes_needed for dp in data_points)
        auto_fix_ratio = (
            total_auto_fixes / (total_auto_fixes + total_manual_fixes)
            if (total_auto_fixes + total_manual_fixes) > 0
            else 0.0
        )

        # Consistency metrics
        [dp.compliance_rate for dp in data_points]
        consistency_score = (
            1.0 - (analysis.compliance_std_dev / analysis.avg_compliance_rate)
            if analysis.avg_compliance_rate > 0
            else 0.0
        )

        # Growth metrics
        file_growth_rate = (
            (data_points[-1].total_files - data_points[0].total_files)
            / data_points[0].total_files
            if data_points[0].total_files > 0
            else 0.0
        )

        return {
            "period_days": days,
            "data_points": len(data_points),
            "compliance_metrics": {
                "average_rate": analysis.avg_compliance_rate,
                "min_rate": analysis.min_compliance_rate,
                "max_rate": analysis.max_compliance_rate,
                "std_deviation": analysis.compliance_std_dev,
                "volatility": analysis.volatility,
            },
            "trend_metrics": {
                "direction": analysis.trend_direction,
                "strength": analysis.trend_strength,
                "slope": analysis.overall_trend,
                "total_improvement": analysis.total_improvement,
                "improvement_rate": analysis.improvement_rate,
            },
            "efficiency_metrics": {
                "auto_fix_ratio": auto_fix_ratio,
                "total_auto_fixes": total_auto_fixes,
                "total_manual_fixes": total_manual_fixes,
                "consistency_score": consistency_score,
            },
            "growth_metrics": {
                "file_growth_rate": file_growth_rate,
                "current_files": data_points[-1].total_files,
                "current_issues": data_points[-1].total_issues,
            },
            "predictions": {
                "next_week": analysis.next_week_prediction,
                "next_month": analysis.next_month_prediction,
                "confidence": analysis.confidence_level,
            },
        }


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Compliance trend tracking system")
    parser.add_argument(
        "--analyze-trends", type=int, default=30, help="Analyze trends for N days"
    )
    parser.add_argument(
        "--generate-report", type=int, help="Generate trend report for N days"
    )
    parser.add_argument(
        "--forecast-compliance",
        type=int,
        default=30,
        help="Forecast compliance for N days",
    )
    parser.add_argument("--export-data", help="Export data in format (json/csv/html)")
    parser.add_argument(
        "--performance-metrics",
        type=int,
        default=30,
        help="Get performance metrics for N days",
    )
    parser.add_argument(
        "--db-path", default="compliance_monitoring.db", help="Database file path"
    )

    args = parser.parse_args()

    tracker = ComplianceTrendTracker(args.db_path)

    if args.analyze_trends:
        try:
            analysis = tracker.analyze_trends(args.analyze_trends)
            print(tracker.generate_trend_report(analysis))
        except ValueError as e:
            print(f"Error: {e}")

    elif args.generate_report:
        try:
            analysis = tracker.analyze_trends(args.generate_report)
            report = tracker.generate_trend_report(analysis)
            print(report)
        except ValueError as e:
            print(f"Error: {e}")

    elif args.forecast_compliance:
        try:
            analysis = tracker.analyze_trends(args.forecast_compliance)
            print(
                f"\n🔮 Compliance Forecast ({args.forecast_compliance} days analysis)"
            )
            print("-" * 50)
            print(f"Next Week Prediction: {analysis.next_week_prediction:.1%}")
            print(f"Next Month Prediction: {analysis.next_month_prediction:.1%}")
            print(f"Confidence Level: {analysis.confidence_level:.1%}")
            print(f"Trend Direction: {analysis.trend_direction.title()}")
            print(f"Trend Strength: {analysis.trend_strength.title()}")
        except ValueError as e:
            print(f"Error: {e}")

    elif args.export_data:
        try:
            analysis = tracker.analyze_trends(30)  # Default to 30 days
            exported_data = tracker.export_trend_data(analysis, args.export_data)
            print(exported_data)
        except ValueError as e:
            print(f"Error: {e}")

    elif args.performance_metrics:
        try:
            metrics = tracker.get_performance_metrics(args.performance_metrics)
            print(f"\n📊 Performance Metrics ({args.performance_metrics} days)")
            print("-" * 50)
            print(json.dumps(metrics, indent=2))
        except ValueError as e:
            print(f"Error: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
