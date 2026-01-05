"""
Expanded comprehensive tests for PerformanceMonitor module.
Tests metric collection, aggregation, performance reports, event tracking.
"""

import pytest
import tempfile
import json
import os
import time
import sqlite3
from unittest.mock import Mock, patch, MagicMock
import shutil

from hpfracc.analytics.performance_monitor import (
    PerformanceMonitor,
    PerformanceEvent,
    PerformanceStats
)


class TestPerformanceMonitorExpanded:
    """Expanded tests for PerformanceMonitor class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_path = tempfile.mkdtemp()
        db_path = os.path.join(temp_path, "test_performance_analytics.db")
        yield db_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def monitor(self, temp_db):
        """Create PerformanceMonitor instance."""
        return PerformanceMonitor(db_path=temp_db, enable_monitoring=True)
    
    def test_initialization_with_custom_db_path(self, temp_db):
        """Test initialization with custom database path."""
        monitor = PerformanceMonitor(db_path=temp_db, enable_monitoring=True)
        assert monitor.db_path == temp_db
        assert monitor.enable_monitoring is True
        assert os.path.exists(temp_db)
    
    def test_initialization_disabled(self):
        """Test initialization with monitoring disabled."""
        monitor = PerformanceMonitor(enable_monitoring=False)
        assert monitor.enable_monitoring is False
    
    def test_monitor_performance_context_manager(self, monitor):
        """Test performance monitoring context manager."""
        monitor.enable_monitoring = True
        
        with monitor.monitor_performance(
            method_name="test_method",
            estimator_type="test_estimator",
            array_size=100,
            fractional_order=0.5,
            parameters={"param": 1}
        ):
            time.sleep(0.01)  # Simulate some work
        
        # Verify event was stored
        stats = monitor.get_performance_stats()
        assert "test_method" in stats
    
    def test_monitor_performance_disabled(self):
        """Test performance monitoring when disabled."""
        monitor = PerformanceMonitor(enable_monitoring=False)
        
        with monitor.monitor_performance(
            method_name="test_method",
            estimator_type="test_estimator",
            array_size=100,
            fractional_order=0.5,
            parameters={}
        ):
            time.sleep(0.01)
        
        # Should not store events
        stats = monitor.get_performance_stats()
        assert len(stats) == 0
    
    def test_monitor_performance_with_exception(self, monitor):
        """Test performance monitoring with exception."""
        monitor.enable_monitoring = True
        
        try:
            with monitor.monitor_performance(
                method_name="test_method",
                estimator_type="test_estimator",
                array_size=100,
                fractional_order=0.5,
                parameters={}
            ):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still record the event with success=False
        stats = monitor.get_performance_stats()
        assert "test_method" in stats
        assert stats["test_method"].success_rate < 1.0
    
    def test_get_performance_stats_empty(self, monitor):
        """Test getting performance stats with no events."""
        stats = monitor.get_performance_stats()
        assert isinstance(stats, dict)
        assert len(stats) == 0
    
    def test_get_performance_stats_with_events(self, monitor):
        """Test getting performance stats with multiple events."""
        # Record multiple performance events
        for i in range(5):
            with monitor.monitor_performance(
                method_name="method1",
                estimator_type="estimator1",
                array_size=100 + i * 10,
                fractional_order=0.5,
                parameters={"iter": i}
            ):
                time.sleep(0.01)
        
        for i in range(3):
            with monitor.monitor_performance(
                method_name="method2",
                estimator_type="estimator2",
                array_size=200,
                fractional_order=0.7,
                parameters={"iter": i}
            ):
                time.sleep(0.02)
        
        stats = monitor.get_performance_stats()
        
        assert "method1" in stats
        assert "method2" in stats
        assert stats["method1"].total_executions == 5
        assert stats["method2"].total_executions == 3
        assert stats["method1"].avg_execution_time > 0
        assert stats["method2"].avg_execution_time > 0
    
    def test_get_performance_stats_with_time_window(self, monitor):
        """Test getting performance stats with time window."""
        # Record old event
        with monitor.monitor_performance(
            method_name="method1",
            estimator_type="estimator1",
            array_size=100,
            fractional_order=0.5,
            parameters={}
        ):
            time.sleep(0.01)
        
        time.sleep(0.1)
        
        # Record new event
        with monitor.monitor_performance(
            method_name="method1",
            estimator_type="estimator1",
            array_size=100,
            fractional_order=0.5,
            parameters={}
        ):
            time.sleep(0.01)
        
        # Get stats with very short time window
        stats = monitor.get_performance_stats(time_window_hours=0.0001)
        
        # Should only include recent events
        assert isinstance(stats, dict)
    
    def test_get_performance_stats_error_handling(self, monitor):
        """Test error handling in get_performance_stats."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            stats = monitor.get_performance_stats()
            assert stats == {}
    
    def test_process_events_to_stats(self, monitor):
        """Test processing events to statistics."""
        # Create mock events
        events = [
            (1, time.time(), "method1", "estimator1", 100, 0.5, 0.1, 50.0, 60.0, 70.0, 10.0, 2, 0.01, 
             json.dumps({"p": 1}), True, None),
            (2, time.time(), "method1", "estimator1", 200, 0.6, 0.2, 60.0, 70.0, 80.0, 15.0, 3, 0.02,
             json.dumps({"p": 2}), True, None),
            (3, time.time(), "method2", "estimator2", 300, 0.7, 0.15, 70.0, 80.0, 90.0, 20.0, 1, 0.01,
             json.dumps({"p": 3}), False, "Error message")
        ]
        
        stats = monitor._process_events_to_stats(events)
        
        assert "method1" in stats
        assert "method2" in stats
        assert stats["method1"].total_executions == 2
        assert stats["method2"].total_executions == 1
        assert stats["method1"].success_rate == 1.0
        assert stats["method2"].success_rate == 0.0
        assert stats["method1"].avg_execution_time > 0
        assert 'p25' in stats["method1"].performance_percentiles
    
    def test_get_performance_trends(self, monitor):
        """Test getting performance trends."""
        # Record some events
        for i in range(3):
            with monitor.monitor_performance(
                method_name="method1",
                estimator_type="estimator1",
                array_size=100,
                fractional_order=0.5,
                parameters={}
            ):
                time.sleep(0.01)
        
        trends = monitor.get_performance_trends("method1", days=7)
        
        assert isinstance(trends, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in trends)
    
    def test_get_performance_trends_nonexistent_method(self, monitor):
        """Test getting performance trends for nonexistent method."""
        trends = monitor.get_performance_trends("nonexistent_method", days=7)
        assert trends == []
    
    def test_get_performance_trends_error_handling(self, monitor):
        """Test error handling in get_performance_trends."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            trends = monitor.get_performance_trends("method1", days=7)
            assert trends == []
    
    def test_get_bottleneck_analysis(self, monitor):
        """Test getting bottleneck analysis."""
        # Record events with different characteristics
        for method_name, exec_time, memory, cpu in [
            ("slow_method", 1.0, 200.0, 50.0),
            ("fast_method", 0.1, 50.0, 10.0),
            ("memory_intensive", 0.5, 500.0, 30.0),
            ("cpu_intensive", 0.3, 100.0, 80.0)
        ]:
            # Manually create and store events
            event = PerformanceEvent(
                timestamp=time.time(),
                method_name=method_name,
                estimator_type="estimator1",
                array_size=100,
                fractional_order=0.5,
                execution_time=exec_time,
                memory_before=100.0,
                memory_after=100.0 + memory,
                memory_peak=100.0 + memory,
                cpu_percent=cpu,
                gc_collections=0,
                gc_time=0.0,
                parameters={},
                success=True
            )
            monitor._store_performance_event(event)
        
        bottlenecks = monitor.get_bottleneck_analysis()
        
        assert isinstance(bottlenecks, dict)
        assert 'slowest_methods' in bottlenecks
        assert 'memory_intensive_methods' in bottlenecks
        assert 'cpu_intensive_methods' in bottlenecks
        assert 'unreliable_methods' in bottlenecks
        assert len(bottlenecks['slowest_methods']) > 0
    
    def test_get_bottleneck_analysis_empty(self, monitor):
        """Test getting bottleneck analysis with no events."""
        bottlenecks = monitor.get_bottleneck_analysis()
        
        assert isinstance(bottlenecks, dict)
        assert len(bottlenecks.get('slowest_methods', [])) == 0
    
    def test_get_bottleneck_analysis_error_handling(self, monitor):
        """Test error handling in get_bottleneck_analysis."""
        with patch.object(monitor, 'get_performance_stats', side_effect=Exception("Error")):
            bottlenecks = monitor.get_bottleneck_analysis()
            assert bottlenecks == {}
    
    def test_export_performance_data(self, monitor, temp_db):
        """Test exporting performance data."""
        # Record some events
        for i in range(3):
            with monitor.monitor_performance(
                method_name="method1",
                estimator_type="estimator1",
                array_size=100,
                fractional_order=0.5,
                parameters={}
            ):
                time.sleep(0.01)
        
        export_path = os.path.join(os.path.dirname(temp_db), "export.json")
        result = monitor.export_performance_data(export_path)
        
        assert result is True
        assert os.path.exists(export_path)
        
        # Verify JSON is valid
        with open(export_path, 'r') as f:
            data = json.load(f)
            assert 'export_timestamp' in data
            assert 'total_methods' in data
            assert 'methods' in data
    
    def test_export_performance_data_empty(self, monitor, temp_db):
        """Test exporting performance data with no events."""
        export_path = os.path.join(os.path.dirname(temp_db), "export.json")
        result = monitor.export_performance_data(export_path)
        
        assert result is True
        assert os.path.exists(export_path)
    
    def test_export_performance_data_error_handling(self, monitor):
        """Test error handling in export_performance_data."""
        with patch.object(monitor, 'get_performance_stats', side_effect=Exception("Error")):
            result = monitor.export_performance_data("export.json")
            assert result is False
    
    def test_clear_old_data(self, monitor):
        """Test clearing old data."""
        # Record some events
        for i in range(5):
            with monitor.monitor_performance(
                method_name="method1",
                estimator_type="estimator1",
                array_size=100,
                fractional_order=0.5,
                parameters={}
            ):
                time.sleep(0.01)
        
        # Clear data older than 0 days (should clear all)
        deleted = monitor.clear_old_data(days_to_keep=0)
        
        assert deleted >= 0
        
        # Verify data was cleared
        stats = monitor.get_performance_stats()
        assert len(stats) == 0 or stats.get("method1", Mock(total_executions=0)).total_executions == 0
    
    def test_clear_old_data_with_retention(self, monitor):
        """Test clearing old data with retention period."""
        # Record some events
        for i in range(3):
            with monitor.monitor_performance(
                method_name="method1",
                estimator_type="estimator1",
                array_size=100,
                fractional_order=0.5,
                parameters={}
            ):
                time.sleep(0.01)
        
        # Clear data older than 30 days (should keep recent data)
        deleted = monitor.clear_old_data(days_to_keep=30)
        
        assert deleted >= 0
        
        # Recent data should still be there
        stats = monitor.get_performance_stats()
        assert len(stats) > 0
    
    def test_clear_old_data_error_handling(self, monitor):
        """Test error handling in clear_old_data."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            deleted = monitor.clear_old_data(days_to_keep=30)
            assert deleted == 0
    
    def test_store_performance_event(self, monitor):
        """Test storing performance event."""
        event = PerformanceEvent(
            timestamp=time.time(),
            method_name="test_method",
            estimator_type="test_estimator",
            array_size=100,
            fractional_order=0.5,
            execution_time=0.1,
            memory_before=50.0,
            memory_after=60.0,
            memory_peak=65.0,
            cpu_percent=10.0,
            gc_collections=2,
            gc_time=0.01,
            parameters={"param": 1},
            success=True
        )
        
        monitor._store_performance_event(event)
        
        # Verify event was stored
        stats = monitor.get_performance_stats()
        assert "test_method" in stats
    
    def test_store_performance_event_error_handling(self, monitor):
        """Test error handling in store_performance_event."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            event = PerformanceEvent(
                timestamp=time.time(),
                method_name="test_method",
                estimator_type="test_estimator",
                array_size=100,
                fractional_order=0.5,
                execution_time=0.1,
                memory_before=50.0,
                memory_after=60.0,
                memory_peak=65.0,
                cpu_percent=10.0,
                gc_collections=0,
                gc_time=0.0,
                parameters={},
                success=True
            )
            
            # Should not raise exception
            monitor._store_performance_event(event)
    
    def test_performance_stats_calculation(self, monitor):
        """Test performance statistics calculation."""
        # Record events with known values
        execution_times = [0.1, 0.2, 0.3, 0.4, 0.5]
        for exec_time in execution_times:
            event = PerformanceEvent(
                timestamp=time.time(),
                method_name="method1",
                estimator_type="estimator1",
                array_size=100,
                fractional_order=0.5,
                execution_time=exec_time,
                memory_before=50.0,
                memory_after=60.0,
                memory_peak=65.0,
                cpu_percent=10.0,
                gc_collections=0,
                gc_time=0.0,
                parameters={},
                success=True
            )
            monitor._store_performance_event(event)
        
        stats = monitor.get_performance_stats()
        
        assert stats["method1"].total_executions == 5
        assert abs(stats["method1"].avg_execution_time - 0.3) < 0.01
        assert stats["method1"].min_execution_time == 0.1
        assert stats["method1"].max_execution_time == 0.5
        assert stats["method1"].success_rate == 1.0





