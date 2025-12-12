"""
Expanded comprehensive tests for AnalyticsManager module.
Tests export methods, report generation, data retention, error handling, session management.
"""

import pytest
import tempfile
import json
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import shutil

from hpfracc.analytics.analytics_manager import (
    AnalyticsManager, 
    AnalyticsConfig
)
from hpfracc.analytics.usage_tracker import UsageTracker
from hpfracc.analytics.performance_monitor import PerformanceMonitor
from hpfracc.analytics.error_analyzer import ErrorAnalyzer
from hpfracc.analytics.workflow_insights import WorkflowInsights


class TestAnalyticsConfigExpanded:
    """Test AnalyticsConfig dataclass with all options."""
    
    def test_all_config_options(self):
        """Test all configuration options."""
        config = AnalyticsConfig(
            enable_usage_tracking=False,
            enable_performance_monitoring=False,
            enable_error_analysis=False,
            enable_workflow_insights=False,
            data_retention_days=60,
            export_format="csv",
            generate_reports=False,
            report_output_dir="custom_reports"
        )
        
        assert config.enable_usage_tracking is False
        assert config.enable_performance_monitoring is False
        assert config.enable_error_analysis is False
        assert config.enable_workflow_insights is False
        assert config.data_retention_days == 60
        assert config.export_format == "csv"
        assert config.generate_reports is False
        assert config.report_output_dir == "custom_reports"


class TestAnalyticsManagerExpanded:
    """Expanded tests for AnalyticsManager class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def manager(self, temp_dir):
        """Create AnalyticsManager instance with temp directory."""
        config = AnalyticsConfig(report_output_dir=temp_dir)
        with patch('hpfracc.analytics.analytics_manager.UsageTracker') as mock_usage, \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor') as mock_perf, \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer') as mock_error, \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights') as mock_workflow:
            
            mock_usage.return_value = Mock(spec=UsageTracker)
            mock_perf.return_value = Mock(spec=PerformanceMonitor)
            mock_error.return_value = Mock(spec=ErrorAnalyzer)
            mock_workflow.return_value = Mock(spec=WorkflowInsights)
            
            manager = AnalyticsManager(config)
            yield manager
    
    def test_track_method_call_with_error(self, manager):
        """Test tracking method call with error."""
        error = ValueError("Test error")
        manager.track_method_call(
            method_name="test_method",
            estimator_type="test_estimator",
            parameters={"param1": 1},
            array_size=100,
            fractional_order=0.5,
            execution_success=False,
            execution_time=0.1,
            memory_usage=50.0,
            error=error
        )
        
        manager.error_analyzer.track_error.assert_called_once()
        manager.usage_tracker.track_usage.assert_called_once()
        manager.workflow_insights.track_workflow_event.assert_called_once()
    
    def test_track_method_call_without_error(self, manager):
        """Test tracking method call without error."""
        manager.track_method_call(
            method_name="test_method",
            estimator_type="test_estimator",
            parameters={"param1": 1},
            array_size=100,
            fractional_order=0.5,
            execution_success=True,
            execution_time=0.1,
            memory_usage=50.0,
            error=None
        )
        
        manager.error_analyzer.track_error.assert_not_called()
        manager.usage_tracker.track_usage.assert_called_once()
        manager.workflow_insights.track_workflow_event.assert_called_once()
    
    def test_track_method_call_error_handling(self, manager):
        """Test error handling in track_method_call."""
        manager.usage_tracker.track_usage.side_effect = Exception("Tracking error")
        
        # Should not raise exception
        manager.track_method_call(
            method_name="test_method",
            estimator_type="test_estimator",
            parameters={},
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
    
    def test_monitor_method_performance_enabled(self, manager):
        """Test performance monitoring context manager when enabled."""
        manager.config.enable_performance_monitoring = True
        manager.performance_monitor.monitor_performance.return_value.__enter__ = Mock()
        manager.performance_monitor.monitor_performance.return_value.__exit__ = Mock(return_value=None)
        
        with manager.monitor_method_performance(
            method_name="test",
            estimator_type="test",
            array_size=100,
            fractional_order=0.5,
            parameters={}
        ):
            pass
        
        manager.performance_monitor.monitor_performance.assert_called_once()
    
    def test_monitor_method_performance_disabled(self, manager):
        """Test performance monitoring context manager when disabled."""
        manager.config.enable_performance_monitoring = False
        
        with manager.monitor_method_performance(
            method_name="test",
            estimator_type="test",
            array_size=100,
            fractional_order=0.5,
            parameters={}
        ):
            pass
        
        manager.performance_monitor.monitor_performance.assert_not_called()
    
    def test_get_comprehensive_analytics_all_enabled(self, manager):
        """Test getting comprehensive analytics with all components enabled."""
        manager.usage_tracker.get_usage_stats.return_value = {"method1": Mock()}
        manager.usage_tracker.get_popular_methods.return_value = [("method1", 10)]
        manager.usage_tracker.get_method_trends.return_value = [("2024-01-01", 5)]
        manager.performance_monitor.get_performance_stats.return_value = {"method1": Mock()}
        manager.performance_monitor.get_bottleneck_analysis.return_value = {}
        manager.performance_monitor.get_performance_trends.return_value = [("2024-01-01", 0.1)]
        manager.error_analyzer.get_error_stats.return_value = {"method1": Mock()}
        manager.error_analyzer.get_common_error_patterns.return_value = {}
        manager.error_analyzer.get_reliability_ranking.return_value = [("method1", 0.9)]
        manager.error_analyzer.get_error_correlation_analysis.return_value = {}
        manager.workflow_insights.get_workflow_patterns.return_value = []
        manager.workflow_insights.get_method_transitions.return_value = {}
        manager.workflow_insights.get_session_insights.return_value = {}
        manager.workflow_insights.get_user_behavior_clusters.return_value = {}
        
        analytics = manager.get_comprehensive_analytics(time_window_hours=24)
        
        assert 'timestamp' in analytics
        assert 'session_id' in analytics
        assert 'usage' in analytics
        assert 'performance' in analytics
        assert 'errors' in analytics
        assert 'workflow' in analytics
    
    def test_get_comprehensive_analytics_partial_enabled(self, manager):
        """Test getting comprehensive analytics with some components disabled."""
        manager.config.enable_usage_tracking = False
        manager.config.enable_error_analysis = False
        
        manager.performance_monitor.get_performance_stats.return_value = {}
        manager.performance_monitor.get_bottleneck_analysis.return_value = {}
        manager.workflow_insights.get_workflow_patterns.return_value = []
        manager.workflow_insights.get_method_transitions.return_value = {}
        manager.workflow_insights.get_session_insights.return_value = {}
        manager.workflow_insights.get_user_behavior_clusters.return_value = {}
        
        analytics = manager.get_comprehensive_analytics()
        
        assert 'usage' not in analytics
        assert 'errors' not in analytics
        assert 'performance' in analytics
        assert 'workflow' in analytics
    
    def test_get_comprehensive_analytics_error_handling(self, manager):
        """Test error handling in get_comprehensive_analytics."""
        manager.usage_tracker.get_usage_stats.side_effect = Exception("Error")
        
        analytics = manager.get_comprehensive_analytics()
        
        assert analytics == {}
    
    def test_generate_json_report(self, manager, temp_dir):
        """Test JSON report generation."""
        manager.config.export_format = "json"
        manager.usage_tracker.get_usage_stats.return_value = {}
        manager.usage_tracker.get_popular_methods.return_value = []
        manager.performance_monitor.get_performance_stats.return_value = {}
        manager.performance_monitor.get_bottleneck_analysis.return_value = {}
        manager.error_analyzer.get_error_stats.return_value = {}
        manager.error_analyzer.get_common_error_patterns.return_value = {}
        manager.error_analyzer.get_reliability_ranking.return_value = []
        manager.error_analyzer.get_error_correlation_analysis.return_value = {}
        manager.workflow_insights.get_workflow_patterns.return_value = []
        manager.workflow_insights.get_method_transitions.return_value = {}
        manager.workflow_insights.get_session_insights.return_value = {}
        manager.workflow_insights.get_user_behavior_clusters.return_value = {}
        
        report_path = manager.generate_analytics_report()
        
        assert report_path
        assert os.path.exists(report_path)
        assert report_path.endswith('.json')
        
        # Verify JSON is valid
        with open(report_path, 'r') as f:
            data = json.load(f)
            assert isinstance(data, dict)
    
    def test_generate_csv_report(self, manager, temp_dir):
        """Test CSV report generation."""
        manager.config.export_format = "csv"
        manager.usage_tracker.get_usage_stats.return_value = {
            "method1": Mock(total_calls=10, success_rate=0.9, avg_array_size=100, user_sessions=5)
        }
        manager.usage_tracker.get_popular_methods.return_value = []
        manager.performance_monitor.get_performance_stats.return_value = {
            "method1": Mock(total_executions=10, avg_execution_time=0.1, avg_memory_usage=50, success_rate=0.9)
        }
        manager.performance_monitor.get_bottleneck_analysis.return_value = {}
        manager.error_analyzer.get_error_stats.return_value = {
            "method1": Mock(total_errors=1, error_rate=0.1, reliability_score=0.9)
        }
        manager.error_analyzer.get_common_error_patterns.return_value = {}
        manager.error_analyzer.get_reliability_ranking.return_value = []
        manager.error_analyzer.get_error_correlation_analysis.return_value = {}
        manager.workflow_insights.get_workflow_patterns.return_value = []
        manager.workflow_insights.get_method_transitions.return_value = {}
        manager.workflow_insights.get_session_insights.return_value = {}
        manager.workflow_insights.get_user_behavior_clusters.return_value = {}
        
        report_path = manager.generate_analytics_report()
        
        assert report_path
        assert os.path.exists(report_path)
        assert report_path.endswith('.csv')
    
    @patch('hpfracc.analytics.analytics_manager.plt')
    @patch('hpfracc.analytics.analytics_manager.sns')
    def test_generate_html_report(self, mock_sns, mock_plt, manager, temp_dir):
        """Test HTML report generation."""
        manager.config.export_format = "html"
        manager.usage_tracker.get_usage_stats.return_value = {
            "method1": Mock(total_calls=10)
        }
        manager.usage_tracker.get_popular_methods.return_value = [("method1", 10)]
        manager.performance_monitor.get_performance_stats.return_value = {
            "method1": Mock(avg_execution_time=0.1, success_rate=0.9)
        }
        manager.performance_monitor.get_bottleneck_analysis.return_value = {}
        manager.error_analyzer.get_error_stats.return_value = {
            "method1": Mock(error_rate=0.1, reliability_score=0.9)
        }
        manager.error_analyzer.get_common_error_patterns.return_value = {}
        manager.error_analyzer.get_reliability_ranking.return_value = [("method1", 0.9)]
        manager.error_analyzer.get_error_correlation_analysis.return_value = {}
        manager.workflow_insights.get_workflow_patterns.return_value = [
            Mock(method_sequence=["m1", "m2"], frequency=5, avg_success_rate=0.9)
        ]
        manager.workflow_insights.get_method_transitions.return_value = {}
        manager.workflow_insights.get_session_insights.return_value = {}
        manager.workflow_insights.get_user_behavior_clusters.return_value = {}
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
        mock_fig.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        report_path = manager.generate_analytics_report()
        
        assert report_path
        assert os.path.exists(report_path)
        assert report_path.endswith('.html')
        
        # Verify HTML content
        with open(report_path, 'r') as f:
            content = f.read()
            assert '<html>' in content
            assert 'HPFRACC Analytics Report' in content
    
    def test_generate_report_unsupported_format(self, manager, temp_dir):
        """Test report generation with unsupported format."""
        manager.config.export_format = "unsupported"
        manager.usage_tracker.get_usage_stats.return_value = {}
        manager.usage_tracker.get_popular_methods.return_value = []
        manager.performance_monitor.get_performance_stats.return_value = {}
        manager.performance_monitor.get_bottleneck_analysis.return_value = {}
        manager.error_analyzer.get_error_stats.return_value = {}
        manager.error_analyzer.get_common_error_patterns.return_value = {}
        manager.error_analyzer.get_reliability_ranking.return_value = []
        manager.error_analyzer.get_error_correlation_analysis.return_value = {}
        manager.workflow_insights.get_workflow_patterns.return_value = []
        manager.workflow_insights.get_method_transitions.return_value = {}
        manager.workflow_insights.get_session_insights.return_value = {}
        manager.workflow_insights.get_user_behavior_clusters.return_value = {}
        
        # Should fallback to JSON
        report_path = manager.generate_analytics_report()
        assert report_path.endswith('.json')
    
    def test_generate_report_error_handling(self, manager):
        """Test error handling in report generation."""
        manager.get_comprehensive_analytics = Mock(side_effect=Exception("Error"))
        
        report_path = manager.generate_analytics_report()
        
        assert report_path == ""
    
    def test_export_all_data_all_enabled(self, manager, temp_dir):
        """Test exporting all data with all components enabled."""
        manager.config.generate_reports = True
        manager.usage_tracker.export_usage_data.return_value = "usage.json"
        manager.performance_monitor.export_performance_data.return_value = "perf.json"
        manager.error_analyzer.export_error_data.return_value = "errors.json"
        manager.workflow_insights.export_workflow_data.return_value = "workflow.json"
        manager.generate_analytics_report = Mock(return_value="report.json")
        
        export_paths = manager.export_all_data()
        
        assert 'usage' in export_paths
        assert 'performance' in export_paths
        assert 'errors' in export_paths
        assert 'workflow' in export_paths
        assert 'comprehensive_report' in export_paths
    
    def test_export_all_data_partial_enabled(self, manager):
        """Test exporting data with some components disabled."""
        manager.config.enable_usage_tracking = False
        manager.config.generate_reports = False
        manager.performance_monitor.export_performance_data.return_value = "perf.json"
        manager.error_analyzer.export_error_data.return_value = "errors.json"
        manager.workflow_insights.export_workflow_data.return_value = "workflow.json"
        
        export_paths = manager.export_all_data()
        
        assert 'usage' not in export_paths
        assert 'comprehensive_report' not in export_paths
        assert 'performance' in export_paths
    
    def test_export_all_data_error_handling(self, manager):
        """Test error handling in export_all_data."""
        manager.usage_tracker.export_usage_data.side_effect = Exception("Error")
        
        export_paths = manager.export_all_data()
        
        assert isinstance(export_paths, dict)
    
    def test_cleanup_old_data_all_enabled(self, manager):
        """Test cleanup with all components enabled."""
        manager.config.data_retention_days = 30
        manager.usage_tracker.clear_old_data.return_value = 10
        manager.performance_monitor.clear_old_data.return_value = 5
        manager.error_analyzer.clear_old_data.return_value = 3
        manager.workflow_insights.clear_old_data.return_value = 2
        
        results = manager.cleanup_old_data()
        
        assert results['usage'] == 10
        assert results['performance'] == 5
        assert results['errors'] == 3
        assert results['workflow'] == 2
        
        manager.usage_tracker.clear_old_data.assert_called_once_with(30)
        manager.performance_monitor.clear_old_data.assert_called_once_with(30)
        manager.error_analyzer.clear_old_data.assert_called_once_with(30)
        manager.workflow_insights.clear_old_data.assert_called_once_with(30)
    
    def test_cleanup_old_data_partial_enabled(self, manager):
        """Test cleanup with some components disabled."""
        manager.config.enable_usage_tracking = False
        manager.config.data_retention_days = 60
        manager.performance_monitor.clear_old_data.return_value = 5
        manager.error_analyzer.clear_old_data.return_value = 3
        manager.workflow_insights.clear_old_data.return_value = 2
        
        results = manager.cleanup_old_data()
        
        assert 'usage' not in results
        assert results['performance'] == 5
    
    def test_cleanup_old_data_error_handling(self, manager):
        """Test error handling in cleanup_old_data."""
        manager.usage_tracker.clear_old_data.side_effect = Exception("Error")
        
        results = manager.cleanup_old_data()
        
        assert isinstance(results, dict)
    
    def test_session_id_generation(self, manager):
        """Test session ID generation."""
        session_id1 = manager._generate_session_id()
        session_id2 = manager._generate_session_id()
        
        assert isinstance(session_id1, str)
        assert len(session_id1) > 0
        assert session_id1 != session_id2
    
    def test_output_directory_creation(self, temp_dir):
        """Test that output directory is created."""
        config = AnalyticsConfig(report_output_dir=os.path.join(temp_dir, "new_dir"))
        with patch('hpfracc.analytics.analytics_manager.UsageTracker'), \
             patch('hpfracc.analytics.analytics_manager.PerformanceMonitor'), \
             patch('hpfracc.analytics.analytics_manager.ErrorAnalyzer'), \
             patch('hpfracc.analytics.analytics_manager.WorkflowInsights'):
            
            manager = AnalyticsManager(config)
            
            assert os.path.exists(manager.output_dir)
            assert manager.output_dir == Path(config.report_output_dir)
    
    @patch('hpfracc.analytics.analytics_manager.plt')
    @patch('hpfracc.analytics.analytics_manager.sns')
    def test_create_analytics_plots(self, mock_sns, mock_plt, manager, temp_dir):
        """Test creating analytics plots."""
        mock_fig = Mock()
        mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        analytics = {
            'usage': {
                'stats': {
                    'method1': Mock(total_calls=10),
                    'method2': Mock(total_calls=5)
                }
            },
            'performance': {
                'stats': {
                    'method1': Mock(avg_execution_time=0.1),
                    'method2': Mock(avg_execution_time=0.2)
                }
            },
            'errors': {
                'stats': {
                    'method1': Mock(error_rate=0.1, reliability_score=0.9),
                    'method2': Mock(error_rate=0.2, reliability_score=0.8)
                }
            }
        }
        
        manager._create_analytics_plots(analytics)
        
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()
    
    def test_create_analytics_plots_error_handling(self, manager):
        """Test error handling in plot creation."""
        manager._create_analytics_plots({})
        
        # Should not raise exception
    
    def test_generate_html_content(self, manager):
        """Test HTML content generation."""
        analytics = {
            'session_id': 'test-session',
            'usage': {
                'popular_methods': [('method1', 10), ('method2', 5)]
            },
            'performance': {
                'stats': {
                    'method1': Mock(avg_execution_time=0.1, success_rate=0.9)
                }
            },
            'errors': {
                'reliability_ranking': [('method1', 0.9)]
            },
            'workflow': {
                'patterns': [
                    Mock(method_sequence=['m1', 'm2'], frequency=5, avg_success_rate=0.9)
                ]
            }
        }
        
        html = manager._generate_html_content(analytics)
        
        assert '<html>' in html
        assert 'HPFRACC Analytics Report' in html
        assert 'test-session' in html
    
    def test_generate_usage_html(self, manager):
        """Test usage HTML generation."""
        usage_data = {
            'popular_methods': [('method1', 10), ('method2', 5)]
        }
        
        html = manager._generate_usage_html(usage_data)
        
        assert '<table>' in html
        assert 'method1' in html
        assert '10' in html
    
    def test_generate_usage_html_empty(self, manager):
        """Test usage HTML generation with empty data."""
        html = manager._generate_usage_html({})
        
        assert 'No usage data available' in html
    
    def test_generate_performance_html(self, manager):
        """Test performance HTML generation."""
        perf_data = {
            'stats': {
                'method1': Mock(avg_execution_time=0.1, success_rate=0.9)
            }
        }
        
        html = manager._generate_performance_html(perf_data)
        
        assert '<table>' in html
        assert 'method1' in html
    
    def test_generate_performance_html_empty(self, manager):
        """Test performance HTML generation with empty data."""
        html = manager._generate_performance_html({})
        
        assert 'No performance data available' in html
    
    def test_generate_error_html(self, manager):
        """Test error HTML generation."""
        error_data = {
            'reliability_ranking': [('method1', 0.9), ('method2', 0.8)]
        }
        
        html = manager._generate_error_html(error_data)
        
        assert '<table>' in html
        assert 'method1' in html
    
    def test_generate_error_html_empty(self, manager):
        """Test error HTML generation with empty data."""
        html = manager._generate_error_html({})
        
        assert 'No error data available' in html
    
    def test_generate_workflow_html(self, manager):
        """Test workflow HTML generation."""
        workflow_data = {
            'patterns': [
                Mock(method_sequence=['m1', 'm2'], frequency=5, avg_success_rate=0.9)
            ]
        }
        
        html = manager._generate_workflow_html(workflow_data)
        
        assert '<table>' in html
        assert 'm1' in html
    
    def test_generate_workflow_html_empty(self, manager):
        """Test workflow HTML generation with empty data."""
        html = manager._generate_workflow_html({})
        
        assert 'No workflow data available' in html



