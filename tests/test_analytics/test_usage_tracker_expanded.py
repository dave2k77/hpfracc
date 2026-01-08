"""
Expanded comprehensive tests for UsageTracker module.
Tests usage statistics, method tracking, session management, data export.
"""

import pytest
import tempfile
import json
import os
import time
import sqlite3
from unittest.mock import Mock, patch
import shutil

from hpfracc.analytics.usage_tracker import (
    UsageTracker,
    UsageEvent,
    UsageStats
)


class TestUsageTrackerExpanded:
    """Expanded tests for UsageTracker class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_path = tempfile.mkdtemp()
        db_path = os.path.join(temp_path, "test_usage_analytics.db")
        yield db_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def tracker(self, temp_db):
        """Create UsageTracker instance."""
        return UsageTracker(db_path=temp_db, enable_tracking=True)
    
    def test_initialization_with_custom_db_path(self, temp_db):
        """Test initialization with custom database path."""
        tracker = UsageTracker(db_path=temp_db, enable_tracking=True)
        assert tracker.db_path == temp_db
        assert tracker.enable_tracking is True
        assert os.path.exists(temp_db)
    
    def test_initialization_disabled(self):
        """Test initialization with tracking disabled."""
        tracker = UsageTracker(enable_tracking=False)
        assert tracker.enable_tracking is False
    
    def test_track_usage_basic(self, tracker):
        """Test basic usage tracking."""
        tracker.track_usage(
            method_name="test_method",
            estimator_type="test_estimator",
            parameters={"param1": 1},
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        # Verify usage was tracked
        stats = tracker.get_usage_stats()
        assert "test_method" in stats
        assert stats["test_method"].total_calls == 1
    
    def test_track_usage_with_all_parameters(self, tracker):
        """Test usage tracking with all parameters."""
        tracker.track_usage(
            method_name="test_method",
            estimator_type="test_estimator",
            parameters={"param1": 1, "param2": "value"},
            array_size=200,
            fractional_order=0.7,
            execution_success=True,
            user_session_id="session123",
            ip_address="127.0.0.1"
        )
        
        stats = tracker.get_usage_stats()
        assert stats["test_method"].total_calls == 1
        assert stats["test_method"].user_sessions == 1
    
    def test_track_usage_disabled(self):
        """Test usage tracking when disabled."""
        tracker = UsageTracker(enable_tracking=False)
        
        tracker.track_usage(
            method_name="test_method",
            estimator_type="test_estimator",
            parameters={},
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        # Should not track
        stats = tracker.get_usage_stats()
        assert len(stats) == 0
    
    def test_track_usage_error_handling(self, tracker):
        """Test error handling in track_usage."""
        with patch.object(tracker, '_store_event', side_effect=Exception("DB Error")):
            tracker.track_usage(
                method_name="test_method",
                estimator_type="test_estimator",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            # Should not raise exception
    
    def test_get_usage_stats_empty(self, tracker):
        """Test getting usage stats with no usage."""
        stats = tracker.get_usage_stats()
        assert isinstance(stats, dict)
        assert len(stats) == 0
    
    def test_get_usage_stats_with_usage(self, tracker):
        """Test getting usage stats with multiple usage events."""
        # Track multiple usage events
        for i in range(5):
            tracker.track_usage(
                method_name="method1",
                estimator_type="estimator1",
                parameters={"iter": i},
                array_size=100 + i * 10,
                fractional_order=0.5,
                execution_success=(i % 2 == 0)
            )
        
        for i in range(3):
            tracker.track_usage(
                method_name="method2",
                estimator_type="estimator2",
                parameters={"iter": i},
                array_size=200,
                fractional_order=0.7,
                execution_success=True
            )
        
        stats = tracker.get_usage_stats()
        
        assert "method1" in stats
        assert "method2" in stats
        assert stats["method1"].total_calls == 5
        assert stats["method2"].total_calls == 3
        assert stats["method1"].success_rate < 1.0
        assert stats["method2"].success_rate == 1.0
    
    def test_get_usage_stats_with_time_window(self, tracker):
        """Test getting usage stats with time window."""
        # Track old usage
        tracker.track_usage(
            method_name="method1",
            estimator_type="estimator1",
            parameters={},
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        time.sleep(0.1)
        
        # Track new usage
        tracker.track_usage(
            method_name="method1",
            estimator_type="estimator1",
            parameters={},
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        # Get stats with very short time window
        stats = tracker.get_usage_stats(time_window_hours=0.0001)
        
        # Should only include recent usage
        assert isinstance(stats, dict)
    
    def test_get_usage_stats_error_handling(self, tracker):
        """Test error handling in get_usage_stats."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            stats = tracker.get_usage_stats()
            assert stats == {}
    
    def test_get_popular_methods(self, tracker):
        """Test getting popular methods."""
        # Track usage for different methods
        for method_name, count in [("method1", 10), ("method2", 5), ("method3", 15)]:
            for i in range(count):
                tracker.track_usage(
                    method_name=method_name,
                    estimator_type="estimator1",
                    parameters={},
                    array_size=100,
                    fractional_order=0.5,
                    execution_success=True
                )
        
        popular = tracker.get_popular_methods(limit=5)
        
        assert isinstance(popular, list)
        assert len(popular) == 3
        # Should be sorted by count (descending)
        counts = [p[1] for p in popular]
        assert counts == sorted(counts, reverse=True)
        assert popular[0][0] == "method3"  # Most popular
    
    def test_get_popular_methods_empty(self, tracker):
        """Test getting popular methods with no usage."""
        popular = tracker.get_popular_methods()
        assert popular == []
    
    def test_get_method_trends(self, tracker):
        """Test getting method trends."""
        # Track some usage
        for i in range(3):
            tracker.track_usage(
                method_name="method1",
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
        
        trends = tracker.get_method_trends("method1", days=7)
        
        assert isinstance(trends, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in trends)
    
    def test_get_method_trends_nonexistent_method(self, tracker):
        """Test getting method trends for nonexistent method."""
        trends = tracker.get_method_trends("nonexistent_method", days=7)
        assert trends == []
    
    def test_get_method_trends_error_handling(self, tracker):
        """Test error handling in get_method_trends."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            trends = tracker.get_method_trends("method1", days=7)
            assert trends == []
    
    def test_export_usage_data(self, tracker, temp_db):
        """Test exporting usage data."""
        # Track some usage
        for i in range(3):
            tracker.track_usage(
                method_name="method1",
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
        
        export_path = os.path.join(os.path.dirname(temp_db), "export.json")
        result = tracker.export_usage_data(export_path)
        
        assert result is True
        assert os.path.exists(export_path)
        
        # Verify JSON is valid
        with open(export_path, 'r') as f:
            data = json.load(f)
            assert 'export_timestamp' in data
            assert 'total_methods' in data
            assert 'methods' in data
    
    def test_export_usage_data_empty(self, tracker, temp_db):
        """Test exporting usage data with no usage."""
        export_path = os.path.join(os.path.dirname(temp_db), "export.json")
        result = tracker.export_usage_data(export_path)
        
        assert result is True
        assert os.path.exists(export_path)
    
    def test_export_usage_data_error_handling(self, tracker):
        """Test error handling in export_usage_data."""
        with patch.object(tracker, 'get_usage_stats', side_effect=Exception("Error")):
            result = tracker.export_usage_data("export.json")
            assert result is False
    
    def test_clear_old_data(self, tracker):
        """Test clearing old data."""
        # Track some usage
        for i in range(5):
            tracker.track_usage(
                method_name="method1",
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
        
        # Clear data older than 0 days (should clear all)
        deleted = tracker.clear_old_data(days_to_keep=0)
        
        assert deleted >= 0
        
        # Verify data was cleared
        stats = tracker.get_usage_stats()
        assert len(stats) == 0 or stats.get("method1", Mock(total_calls=0)).total_calls == 0
    
    def test_clear_old_data_with_retention(self, tracker):
        """Test clearing old data with retention period."""
        # Track some usage
        for i in range(3):
            tracker.track_usage(
                method_name="method1",
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
        
        # Clear data older than 30 days (should keep recent data)
        deleted = tracker.clear_old_data(days_to_keep=30)
        
        assert deleted >= 0
        
        # Recent data should still be there
        stats = tracker.get_usage_stats()
        assert len(stats) > 0
    
    def test_clear_old_data_error_handling(self, tracker):
        """Test error handling in clear_old_data."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            deleted = tracker.clear_old_data(days_to_keep=30)
            assert deleted == 0
    
    def test_store_event(self, tracker):
        """Test storing usage event."""
        event = UsageEvent(
            timestamp=time.time(),
            method_name="test_method",
            estimator_type="test_estimator",
            parameters={"param": 1},
            array_size=100,
            fractional_order=0.5,
            execution_success=True,
            user_session_id="session1",
            ip_address="127.0.0.1"
        )
        
        tracker._store_event(event)
        
        # Verify event was stored
        stats = tracker.get_usage_stats()
        assert "test_method" in stats
    
    def test_store_event_error_handling(self, tracker):
        """Test error handling in store_event."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            event = UsageEvent(
                timestamp=time.time(),
                method_name="test_method",
                estimator_type="test_estimator",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            
            # Should not raise exception
            tracker._store_event(event)
    
    def test_session_id_generation(self, tracker):
        """Test session ID generation."""
        session_id1 = tracker._generate_session_id()
        session_id2 = tracker._generate_session_id()
        
        assert isinstance(session_id1, str)
        assert len(session_id1) > 0
        assert session_id1 != session_id2
    
    def test_usage_stats_calculation(self, tracker):
        """Test usage statistics calculation."""
        # Track usage with known values
        for i, success in enumerate([True, True, False, True, False]):
            tracker.track_usage(
                method_name="method1",
                estimator_type="estimator1",
                parameters={},
                array_size=100 + i * 10,
                fractional_order=0.5,
                execution_success=success,
                user_session_id=f"session{i % 2}"  # 2 unique sessions
            )
        
        stats = tracker.get_usage_stats()
        
        assert stats["method1"].total_calls == 5
        assert stats["method1"].success_rate == 0.6  # 3 out of 5
        assert stats["method1"].avg_array_size == 120.0  # (100+110+120+130+140)/5
        assert stats["method1"].user_sessions == 2
    
    def test_process_events_to_stats(self, tracker):
        """Test processing events to statistics."""
        # Create mock events
        base_time = time.time()
        events = [
            (1, base_time, "method1", "estimator1", json.dumps({"p": 1}), 100, 0.5, True, "session1", None),
            (2, base_time + 1, "method1", "estimator1", json.dumps({"p": 2}), 200, 0.6, True, "session1", None),
            (3, base_time + 2, "method1", "estimator1", json.dumps({"p": 3}), 300, 0.5, False, "session2", None),
            (4, base_time + 3, "method2", "estimator2", json.dumps({"p": 1}), 150, 0.7, True, "session2", None)
        ]
        
        stats = tracker._process_events_to_stats(events)
        
        assert "method1" in stats
        assert "method2" in stats
        assert stats["method1"].total_calls == 3
        assert stats["method2"].total_calls == 1
        assert stats["method1"].success_rate == 2/3
        assert stats["method1"].user_sessions == 2






