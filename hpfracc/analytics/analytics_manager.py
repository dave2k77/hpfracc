"""
Analytics manager for hpfracc library.

Coordinates all analytics components and provides a unified interface
for tracking, analyzing, and reporting on library usage and performance.
"""

import time
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .usage_tracker import UsageTracker
from .performance_monitor import PerformanceMonitor
from .error_analyzer import ErrorAnalyzer
from .workflow_insights import WorkflowInsights

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsConfig:
    """Configuration for analytics system."""
    enable_usage_tracking: bool = True
    enable_performance_monitoring: bool = True
    enable_error_analysis: bool = True
    enable_workflow_insights: bool = True
    data_retention_days: int = 30
    export_format: str = "json"  # json, csv, html
    generate_reports: bool = True
    report_output_dir: str = "analytics_reports"


class AnalyticsManager:
    """Main analytics manager that coordinates all analytics components."""

    def __init__(self, config: AnalyticsConfig = None):
        self.config = config or AnalyticsConfig()
        self.session_id = self._generate_session_id()

        # Initialize analytics components
        self.usage_tracker = UsageTracker(
            enable_tracking=self.config.enable_usage_tracking)
        self.performance_monitor = PerformanceMonitor(
            enable_monitoring=self.config.enable_performance_monitoring)
        self.error_analyzer = ErrorAnalyzer(
            enable_analysis=self.config.enable_error_analysis)
        self.workflow_insights = WorkflowInsights(
            enable_insights=self.config.enable_workflow_insights)

        # Create output directory
        self.output_dir = Path(self.config.report_output_dir)
        self.output_dir.mkdir(exist_ok=True)

        logger.info("Analytics manager initialized successfully")

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())

    def track_method_call(self,
                          method_name: str,
                          estimator_type: str,
                          parameters: Dict[str,
                                           Any],
                          array_size: int,
                          fractional_order: float,
                          execution_success: bool,
                          execution_time: Optional[float] = None,
                          memory_usage: Optional[float] = None,
                          error: Optional[Exception] = None):
        """Track a complete method call across all analytics components."""
        try:
            # Track usage
            self.usage_tracker.track_usage(
                method_name=method_name,
                estimator_type=estimator_type,
                parameters=parameters,
                array_size=array_size,
                fractional_order=fractional_order,
                execution_success=execution_success,
                user_session_id=self.session_id
            )

            # Track workflow
            self.workflow_insights.track_workflow_event(
                session_id=self.session_id,
                method_name=method_name,
                estimator_type=estimator_type,
                parameters=parameters,
                array_size=array_size,
                fractional_order=fractional_order,
                execution_success=execution_success,
                execution_time=execution_time
            )

            # Track errors if any
            if error is not None:
                self.error_analyzer.track_error(
                    method_name=method_name,
                    estimator_type=estimator_type,
                    error=error,
                    parameters=parameters,
                    array_size=array_size,
                    fractional_order=fractional_order,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    user_session_id=self.session_id
                )

            logger.debug(f"Tracked method call: {method_name}")

        except Exception as e:
            logger.error(f"Failed to track method call: {e}")

    @contextmanager
    def monitor_method_performance(self, method_name: str, estimator_type: str,
                                   array_size: int, fractional_order: float,
                                   parameters: Dict[str, Any]):
        """Context manager for monitoring method performance."""
        if not self.config.enable_performance_monitoring:
            yield
            return

        with self.performance_monitor.monitor_performance(
            method_name, estimator_type, array_size, fractional_order, parameters
        ):
            yield

    def get_comprehensive_analytics(
            self, time_window_hours: Optional[int] = None) -> Dict[str, Any]:
        """Get comprehensive analytics from all components."""
        try:
            analytics = {
                'timestamp': time.time(),
                'session_id': self.session_id,
                'time_window_hours': time_window_hours
            }

            # Usage analytics
            if self.config.enable_usage_tracking:
                analytics['usage'] = {
                    'stats': self.usage_tracker.get_usage_stats(time_window_hours),
                    'popular_methods': self.usage_tracker.get_popular_methods(),
                    'method_trends': {}}

                # Get trends for popular methods
                for method_name, _ in analytics['usage']['popular_methods'][:5]:
                    analytics['usage']['method_trends'][method_name] = \
                        self.usage_tracker.get_method_trends(method_name)

            # Performance analytics
            if self.config.enable_performance_monitoring:
                analytics['performance'] = {
                    'stats': self.performance_monitor.get_performance_stats(time_window_hours),
                    'bottlenecks': self.performance_monitor.get_bottleneck_analysis(),
                    'performance_trends': {}}

                # Get trends for methods with performance data
                perf_stats = analytics['performance']['stats']
                for method_name in list(perf_stats.keys())[:5]:
                    analytics['performance']['performance_trends'][method_name] = \
                        self.performance_monitor.get_performance_trends(
                            method_name)

            # Error analytics
            if self.config.enable_error_analysis:
                analytics['errors'] = {
                    'stats': self.error_analyzer.get_error_stats(time_window_hours),
                    'common_patterns': self.error_analyzer.get_common_error_patterns(),
                    'reliability_ranking': self.error_analyzer.get_reliability_ranking(),
                    'error_correlations': self.error_analyzer.get_error_correlation_analysis()}

            # Workflow analytics
            if self.config.enable_workflow_insights:
                analytics['workflow'] = {
                    'patterns': self.workflow_insights.get_workflow_patterns(),
                    'transitions': self.workflow_insights.get_method_transitions(),
                    'session_insights': self.workflow_insights.get_session_insights(),
                    'user_clusters': self.workflow_insights.get_user_behavior_clusters()}

            return analytics

        except Exception as e:
            logger.error(f"Failed to get comprehensive analytics: {e}")
            return {}

    def generate_analytics_report(
            self, time_window_hours: Optional[int] = None) -> str:
        """Generate a comprehensive analytics report."""
        try:
            analytics = self.get_comprehensive_analytics(time_window_hours)

            if self.config.export_format == "json":
                return self._generate_json_report(analytics)
            elif self.config.export_format == "csv":
                return self._generate_csv_report(analytics)
            elif self.config.export_format == "html":
                return self._generate_html_report(analytics)
            else:
                logger.warning(
                    f"Unsupported export format: {self.config.export_format}")
                return self._generate_json_report(analytics)

        except Exception as e:
            logger.error(f"Failed to generate analytics report: {e}")
            return ""

    def _generate_json_report(self, analytics: Dict[str, Any]) -> str:
        """Generate JSON format report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"analytics_report_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2)

        logger.info(f"JSON report generated: {output_path}")
        return str(output_path)

    def _generate_csv_report(self, analytics: Dict[str, Any]) -> str:
        """Generate CSV format report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"analytics_report_{timestamp}.csv"

        # Flatten analytics data for CSV
        csv_data = []

        # Usage data
        if 'usage' in analytics:
            for method_name, stats in analytics['usage']['stats'].items():
                csv_data.append({
                    'category': 'usage',
                    'method_name': method_name,
                    'total_calls': stats.total_calls,
                    'success_rate': stats.success_rate,
                    'avg_array_size': stats.avg_array_size,
                    'user_sessions': stats.user_sessions
                })

        # Performance data
        if 'performance' in analytics:
            for method_name, stats in analytics['performance']['stats'].items(
            ):
                csv_data.append({
                    'category': 'performance',
                    'method_name': method_name,
                    'total_executions': stats.total_executions,
                    'avg_execution_time': stats.avg_execution_time,
                    'avg_memory_usage': stats.avg_memory_usage,
                    'success_rate': stats.success_rate
                })

        # Error data
        if 'errors' in analytics:
            for method_name, stats in analytics['errors']['stats'].items():
                csv_data.append({
                    'category': 'errors',
                    'method_name': method_name,
                    'total_errors': stats.total_errors,
                    'error_rate': stats.error_rate,
                    'reliability_score': stats.reliability_score
                })

        # Convert to DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)

        logger.info(f"CSV report generated: {output_path}")
        return str(output_path)

    def _generate_html_report(self, analytics: Dict[str, Any]) -> str:
        """Generate HTML format report with visualizations."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"analytics_report_{timestamp}.html"

        # Create visualizations
        self._create_analytics_plots(analytics)

        # Generate HTML content
        html_content = self._generate_html_content(analytics)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {output_path}")
        return str(output_path)

    def _create_analytics_plots(self, analytics: Dict[str, Any]):
        """Create visualization plots for analytics data."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('HPFRACC Analytics Dashboard',
                         fontsize=16, fontweight='bold')

            # Handle both numpy array and list indexing for axes
            if hasattr(axes, 'shape'):  # numpy array case
                ax1, ax2, ax3, ax4 = axes[0,
                                          0], axes[0, 1], axes[1, 0], axes[1, 1]
            else:  # list case (for testing)
                ax1, ax2, ax3, ax4 = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

            # 1. Usage patterns
            if 'usage' in analytics and analytics['usage']['stats']:
                methods = list(analytics['usage']['stats'].keys())
                call_counts = [
                    stats.total_calls for stats in analytics['usage']['stats'].values()]

                bars = ax1.bar(methods, call_counts,
                               color='skyblue', alpha=0.7)
                ax1.set_title('Method Usage Counts')
                ax1.set_xlabel('Method')
                ax1.set_ylabel('Total Calls')
                ax1.tick_params(axis='x', rotation=45)

                # Add value labels on bars
                try:
                    for bar, count in zip(bars, call_counts):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                                 f'{count}', ha='center', va='bottom')
                except (TypeError, AttributeError):
                    # Skip adding labels if bars is not iterable (e.g., Mock object)
                    pass

            # 2. Performance comparison
            if 'performance' in analytics and analytics['performance']['stats']:
                methods = list(analytics['performance']['stats'].keys())
                exec_times = [
                    stats.avg_execution_time for stats in analytics['performance']['stats'].values()]

                bars = ax2.bar(methods, exec_times,
                               color='lightcoral', alpha=0.7)
                ax2.set_title('Average Execution Times')
                ax2.set_xlabel('Method')
                ax2.set_ylabel('Time (seconds)')
                ax2.tick_params(axis='x', rotation=45)
                ax2.set_yscale('log')

            # 3. Error rates
            if 'errors' in analytics and analytics['errors']['stats']:
                methods = list(analytics['errors']['stats'].keys())
                error_rates = [
                    stats.error_rate for stats in analytics['errors']['stats'].values()]

                bars = ax3.bar(methods, error_rates, color='gold', alpha=0.7)
                ax3.set_title('Error Rates')
                ax3.set_xlabel('Method')
                ax3.set_ylabel('Error Rate')
                ax3.tick_params(axis='x', rotation=45)

            # 4. Reliability scores
            if 'errors' in analytics and analytics['errors']['stats']:
                methods = list(analytics['errors']['stats'].keys())
                reliability = [
                    stats.reliability_score for stats in analytics['errors']['stats'].values()]

                bars = ax4.bar(methods, reliability,
                               color='lightgreen', alpha=0.7)
                ax4.set_title('Reliability Scores')
                ax4.set_xlabel('Method')
                ax4.set_ylabel('Reliability Score')
                ax4.tick_params(axis='x', rotation=45)
                ax4.set_ylim(0, 1)

            plt.tight_layout()

            # Save plot
            timestamp = int(time.time())
            plot_path = self.output_dir / f"analytics_plots_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Analytics plots generated: {plot_path}")

        except Exception as e:
            logger.error(f"Failed to create analytics plots: {e}")

    def _generate_html_content(self, analytics: Dict[str, Any]) -> str:
        """Generate HTML content for the report."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HPFRACC Analytics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 HPFRACC Analytics Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Session ID:</strong> {analytics.get('session_id', 'N/A')}</p>
            </div>

            <div class="section">
                <h2>📊 Usage Analytics</h2>
                {self._generate_usage_html(analytics.get('usage', {}))}
            </div>

            <div class="section">
                <h2>⚡ Performance Analytics</h2>
                {self._generate_performance_html(analytics.get('performance', {}))}
            </div>

            <div class="section">
                <h2>🚨 Error Analytics</h2>
                {self._generate_error_html(analytics.get('errors', {}))}
            </div>

            <div class="section">
                <h2>🔄 Workflow Analytics</h2>
                {self._generate_workflow_html(analytics.get('workflow', {}))}
            </div>
        </body>
        </html>
        """

        return html

    def _generate_usage_html(self, usage_data: Dict[str, Any]) -> str:
        """Generate HTML for usage section."""
        if not usage_data:
            return "<p>No usage data available</p>"

        html = "<h3>Popular Methods</h3>"
        if 'popular_methods' in usage_data:
            html += "<table><tr><th>Method</th><th>Call Count</th></tr>"
            for method, count in usage_data['popular_methods'][:10]:
                html += f"<tr><td>{method}</td><td>{count}</td></tr>"
            html += "</table>"

        return html

    def _generate_performance_html(self, perf_data: Dict[str, Any]) -> str:
        """Generate HTML for performance section."""
        if not perf_data:
            return "<p>No performance data available</p>"

        html = "<h3>Performance Statistics</h3>"
        if 'stats' in perf_data:
            html += "<table><tr><th>Method</th><th>Avg Time (s)</th><th>Success Rate</th></tr>"
            for method, stats in perf_data['stats'].items():
                success_class = "success" if stats.success_rate > 0.9 else "warning" if stats.success_rate > 0.7 else "error"
                html += f"<tr><td>{method}</td><td>{stats.avg_execution_time:.6f}</td><td class='{success_class}'>{stats.success_rate:.2%}</td></tr>"
            html += "</table>"

        return html

    def _generate_error_html(self, error_data: Dict[str, Any]) -> str:
        """Generate HTML for error section."""
        if not error_data:
            return "<p>No error data available</p>"

        html = "<h3>Reliability Ranking</h3>"
        if 'reliability_ranking' in error_data:
            html += "<table><tr><th>Method</th><th>Reliability Score</th></tr>"
            for method, score in error_data['reliability_ranking'][:10]:
                score_class = "success" if score > 0.9 else "warning" if score > 0.7 else "error"
                html += f"<tr><td>{method}</td><td class='{score_class}'>{score:.3f}</td></tr>"
            html += "</table>"

        return html

    def _generate_workflow_html(self, workflow_data: Dict[str, Any]) -> str:
        """Generate HTML for workflow section."""
        if not workflow_data:
            return "<p>No workflow data available</p>"

        html = "<h3>Common Workflow Patterns</h3>"
        if 'patterns' in workflow_data:
            html += "<table><tr><th>Pattern</th><th>Frequency</th><th>Success Rate</th></tr>"
            for pattern in workflow_data['patterns'][:10]:
                success_class = "success" if pattern.avg_success_rate > 0.9 else "warning" if pattern.avg_success_rate > 0.7 else "error"
                html += f"<tr><td>{' → '.join(pattern.method_sequence)}</td><td>{pattern.frequency}</td><td class='{success_class}'>{pattern.avg_success_rate:.2%}</td></tr>"
            html += "</table>"

        return html

    def export_all_data(self) -> Dict[str, str]:
        """Export data from all analytics components."""
        try:
            export_paths = {}

            # Export usage data
            if self.config.enable_usage_tracking:
                usage_path = self.usage_tracker.export_usage_data()
                if usage_path:
                    export_paths['usage'] = usage_path

            # Export performance data
            if self.config.enable_performance_monitoring:
                perf_path = self.performance_monitor.export_performance_data()
                if perf_path:
                    export_paths['performance'] = perf_path

            # Export error data
            if self.config.enable_error_analysis:
                error_path = self.error_analyzer.export_error_data()
                if error_path:
                    export_paths['errors'] = error_path

            # Export workflow data
            if self.config.enable_workflow_insights:
                workflow_path = self.workflow_insights.export_workflow_data()
                if workflow_path:
                    export_paths['workflow'] = workflow_path

            # Generate comprehensive report
            if self.config.generate_reports:
                report_path = self.generate_analytics_report()
                if report_path:
                    export_paths['comprehensive_report'] = report_path

            logger.info(f"Exported {len(export_paths)} analytics datasets")
            return export_paths

        except Exception as e:
            logger.error(f"Failed to export analytics data: {e}")
            return {}

    def cleanup_old_data(self):
        """Clean up old data from all analytics components."""
        try:
            cleanup_results = {}

            # Cleanup usage data
            if self.config.enable_usage_tracking:
                deleted_count = self.usage_tracker.clear_old_data(
                    self.config.data_retention_days)
                cleanup_results['usage'] = deleted_count

            # Cleanup performance data
            if self.config.enable_performance_monitoring:
                deleted_count = self.performance_monitor.clear_old_data(
                    self.config.data_retention_days)
                cleanup_results['performance'] = deleted_count

            # Cleanup error data
            if self.config.enable_error_analysis:
                deleted_count = self.error_analyzer.clear_old_data(
                    self.config.data_retention_days)
                cleanup_results['errors'] = deleted_count

            # Cleanup workflow data
            if self.config.enable_workflow_insights:
                deleted_count = self.workflow_insights.clear_old_data(
                    self.config.data_retention_days)
                cleanup_results['workflow'] = deleted_count

            total_deleted = sum(cleanup_results.values())
            logger.info(f"Cleaned up {total_deleted} old analytics records")
            return cleanup_results

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            return {}


# Context manager for the performance monitor
