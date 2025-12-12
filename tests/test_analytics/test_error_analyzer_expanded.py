"""
Expanded comprehensive tests for ErrorAnalyzer module.
Tests error categorization, statistics computation, error patterns, aggregation.
"""

import pytest
import tempfile
import json
import os
import time
import sqlite3
from unittest.mock import Mock, patch, MagicMock
import shutil

from hpfracc.analytics.error_analyzer import (
    ErrorAnalyzer,
    ErrorEvent,
    ErrorStats
)


class TestErrorAnalyzerExpanded:
    """Expanded tests for ErrorAnalyzer class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_path = tempfile.mkdtemp()
        db_path = os.path.join(temp_path, "test_error_analytics.db")
        yield db_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def analyzer(self, temp_db):
        """Create ErrorAnalyzer instance."""
        return ErrorAnalyzer(db_path=temp_db, enable_analysis=True)
    
    def test_initialization_with_custom_db_path(self, temp_db):
        """Test initialization with custom database path."""
        analyzer = ErrorAnalyzer(db_path=temp_db, enable_analysis=True)
        assert analyzer.db_path == temp_db
        assert analyzer.enable_analysis is True
        assert os.path.exists(temp_db)
    
    def test_initialization_disabled(self):
        """Test initialization with analysis disabled."""
        analyzer = ErrorAnalyzer(enable_analysis=False)
        assert analyzer.enable_analysis is False
    
    def test_track_error_basic(self, analyzer):
        """Test basic error tracking."""
        error = ValueError("Test error message")
        analyzer.track_error(
            method_name="test_method",
            estimator_type="test_estimator",
            error=error,
            parameters={"param1": 1},
            array_size=100,
            fractional_order=0.5
        )
        
        # Verify error was stored
        stats = analyzer.get_error_stats()
        assert "test_method" in stats
        assert stats["test_method"].total_errors == 1
    
    def test_track_error_with_all_parameters(self, analyzer):
        """Test error tracking with all parameters."""
        error = RuntimeError("Runtime error")
        analyzer.track_error(
            method_name="test_method",
            estimator_type="test_estimator",
            error=error,
            parameters={"param1": 1, "param2": "value"},
            array_size=200,
            fractional_order=0.7,
            execution_time=0.5,
            memory_usage=100.0,
            user_session_id="session123"
        )
        
        stats = analyzer.get_error_stats()
        assert stats["test_method"].total_errors == 1
        assert stats["test_method"].avg_execution_time_before_error == 0.5
    
    def test_track_error_disabled(self):
        """Test error tracking when disabled."""
        analyzer = ErrorAnalyzer(enable_analysis=False)
        error = ValueError("Test error")
        
        analyzer.track_error(
            method_name="test_method",
            estimator_type="test_estimator",
            error=error,
            parameters={},
            array_size=100,
            fractional_order=0.5
        )
        
        # Should not raise exception
        stats = analyzer.get_error_stats()
        assert len(stats) == 0
    
    def test_track_error_error_handling(self, analyzer):
        """Test error handling in track_error."""
        # Force database error
        with patch.object(analyzer, '_store_error_event', side_effect=Exception("DB Error")):
            error = ValueError("Test error")
            analyzer.track_error(
                method_name="test_method",
                estimator_type="test_estimator",
                error=error,
                parameters={},
                array_size=100,
                fractional_order=0.5
            )
            # Should not raise exception
    
    def test_get_error_stats_empty(self, analyzer):
        """Test getting error stats with no errors."""
        stats = analyzer.get_error_stats()
        assert isinstance(stats, dict)
        assert len(stats) == 0
    
    def test_get_error_stats_with_errors(self, analyzer):
        """Test getting error stats with multiple errors."""
        # Track multiple errors
        for i in range(5):
            error = ValueError(f"Error {i}")
            analyzer.track_error(
                method_name="method1",
                estimator_type="estimator1",
                error=error,
                parameters={"param": i},
                array_size=100,
                fractional_order=0.5
            )
        
        for i in range(3):
            error = TypeError(f"Type error {i}")
            analyzer.track_error(
                method_name="method2",
                estimator_type="estimator2",
                error=error,
                parameters={"param": i},
                array_size=200,
                fractional_order=0.7
            )
        
        stats = analyzer.get_error_stats()
        
        assert "method1" in stats
        assert "method2" in stats
        assert stats["method1"].total_errors == 5
        assert stats["method2"].total_errors == 3
        assert len(stats["method1"].common_error_types) > 0
    
    def test_get_error_stats_with_time_window(self, analyzer):
        """Test getting error stats with time window."""
        # Track old error
        error1 = ValueError("Old error")
        analyzer.track_error(
            method_name="method1",
            estimator_type="estimator1",
            error=error1,
            parameters={},
            array_size=100,
            fractional_order=0.5
        )
        
        # Wait a bit and track new error
        time.sleep(0.1)
        error2 = ValueError("New error")
        analyzer.track_error(
            method_name="method1",
            estimator_type="estimator1",
            error=error2,
            parameters={},
            array_size=100,
            fractional_order=0.5
        )
        
        # Get stats with very short time window
        stats = analyzer.get_error_stats(time_window_hours=0.0001)
        
        # Should only include recent errors (may be 0 or 1 depending on timing)
        assert isinstance(stats, dict)
    
    def test_get_error_stats_error_handling(self, analyzer):
        """Test error handling in get_error_stats."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            stats = analyzer.get_error_stats()
            assert stats == {}
    
    def test_process_events_to_stats(self, analyzer):
        """Test processing events to statistics."""
        # Create mock events
        events = [
            (1, time.time(), "method1", "estimator1", "ValueError", "Error 1", "traceback", "hash1", 
             json.dumps({"p": 1}), 100, 0.5, 0.1, 50.0, "session1"),
            (2, time.time(), "method1", "estimator1", "ValueError", "Error 2", "traceback", "hash2",
             json.dumps({"p": 2}), 200, 0.6, 0.2, 60.0, "session2"),
            (3, time.time(), "method2", "estimator2", "TypeError", "Error 3", "traceback", "hash3",
             json.dumps({"p": 3}), 300, 0.7, None, None, "session3")
        ]
        
        stats = analyzer._process_events_to_stats(events)
        
        assert "method1" in stats
        assert "method2" in stats
        assert stats["method1"].total_errors == 2
        assert stats["method2"].total_errors == 1
        assert stats["method1"].avg_execution_time_before_error > 0
        assert stats["method2"].avg_execution_time_before_error == 0.0
    
    def test_get_error_trends(self, analyzer):
        """Test getting error trends."""
        # Track some errors
        for i in range(3):
            error = ValueError(f"Error {i}")
            analyzer.track_error(
                method_name="method1",
                estimator_type="estimator1",
                error=error,
                parameters={},
                array_size=100,
                fractional_order=0.5
            )
        
        trends = analyzer.get_error_trends("method1", days=7)
        
        assert isinstance(trends, list)
        assert len(trends) > 0
        assert all(isinstance(t, tuple) and len(t) == 2 for t in trends)
    
    def test_get_error_trends_nonexistent_method(self, analyzer):
        """Test getting error trends for nonexistent method."""
        trends = analyzer.get_error_trends("nonexistent_method", days=7)
        assert trends == []
    
    def test_get_error_trends_error_handling(self, analyzer):
        """Test error handling in get_error_trends."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            trends = analyzer.get_error_trends("method1", days=7)
            assert trends == []
    
    def test_get_common_error_patterns(self, analyzer):
        """Test getting common error patterns."""
        # Track various errors
        errors = [
            (ValueError, "Value error 1"),
            (ValueError, "Value error 2"),
            (TypeError, "Type error 1"),
            (RuntimeError, "Runtime error 1")
        ]
        
        for error_class, error_msg in errors:
            error = error_class(error_msg)
            analyzer.track_error(
                method_name="method1",
                estimator_type="estimator1",
                error=error,
                parameters={},
                array_size=100,
                fractional_order=0.5
            )
        
        patterns = analyzer.get_common_error_patterns()
        
        assert isinstance(patterns, dict)
        assert 'common_error_types' in patterns
        assert 'common_error_messages' in patterns
        assert 'error_prone_parameters' in patterns
        assert len(patterns['common_error_types']) > 0
    
    def test_get_common_error_patterns_empty(self, analyzer):
        """Test getting common error patterns with no errors."""
        patterns = analyzer.get_common_error_patterns()
        
        assert isinstance(patterns, dict)
        assert 'common_error_types' in patterns
        assert len(patterns['common_error_types']) == 0
    
    def test_get_common_error_patterns_error_handling(self, analyzer):
        """Test error handling in get_common_error_patterns."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            patterns = analyzer.get_common_error_patterns()
            assert patterns == {}
    
    def test_get_reliability_ranking(self, analyzer):
        """Test getting reliability ranking."""
        # Track errors for different methods
        for method_name, error_count in [("method1", 2), ("method2", 5), ("method3", 1)]:
            for i in range(error_count):
                error = ValueError(f"Error {i}")
                analyzer.track_error(
                    method_name=method_name,
                    estimator_type="estimator1",
                    error=error,
                    parameters={},
                    array_size=100,
                    fractional_order=0.5
                )
        
        ranking = analyzer.get_reliability_ranking()
        
        assert isinstance(ranking, list)
        assert len(ranking) == 3
        # Should be sorted by reliability score (descending)
        scores = [r[1] for r in ranking]
        assert scores == sorted(scores, reverse=True)
    
    def test_get_reliability_ranking_empty(self, analyzer):
        """Test getting reliability ranking with no errors."""
        ranking = analyzer.get_reliability_ranking()
        assert ranking == []
    
    def test_get_reliability_ranking_error_handling(self, analyzer):
        """Test error handling in get_reliability_ranking."""
        with patch.object(analyzer, 'get_error_stats', side_effect=Exception("Error")):
            ranking = analyzer.get_reliability_ranking()
            assert ranking == []
    
    def test_get_error_correlation_analysis(self, analyzer):
        """Test getting error correlation analysis."""
        # Track errors with different characteristics
        for array_size, fractional_order, exec_time in [
            (100, 0.5, 0.1),
            (200, 0.6, 0.2),
            (100, 0.5, 0.15),
            (300, 0.7, 1.5)
        ]:
            error = ValueError("Error")
            analyzer.track_error(
                method_name="method1",
                estimator_type="estimator1",
                error=error,
                parameters={},
                array_size=array_size,
                fractional_order=fractional_order,
                execution_time=exec_time
            )
        
        correlations = analyzer.get_error_correlation_analysis()
        
        assert isinstance(correlations, dict)
        assert 'array_size_correlation' in correlations
        assert 'fractional_order_correlation' in correlations
        assert 'execution_time_correlation' in correlations
        assert len(correlations['array_size_correlation']) > 0
    
    def test_get_error_correlation_analysis_empty(self, analyzer):
        """Test getting error correlation analysis with no errors."""
        correlations = analyzer.get_error_correlation_analysis()
        
        assert isinstance(correlations, dict)
        assert 'array_size_correlation' in correlations
        assert len(correlations['array_size_correlation']) == 0
    
    def test_get_error_correlation_analysis_error_handling(self, analyzer):
        """Test error handling in get_error_correlation_analysis."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            correlations = analyzer.get_error_correlation_analysis()
            assert correlations == {}
    
    def test_export_error_data(self, analyzer, temp_db):
        """Test exporting error data."""
        # Track some errors
        for i in range(3):
            error = ValueError(f"Error {i}")
            analyzer.track_error(
                method_name="method1",
                estimator_type="estimator1",
                error=error,
                parameters={},
                array_size=100,
                fractional_order=0.5
            )
        
        export_path = os.path.join(os.path.dirname(temp_db), "export.json")
        result = analyzer.export_error_data(export_path)
        
        assert result is True
        assert os.path.exists(export_path)
        
        # Verify JSON is valid
        with open(export_path, 'r') as f:
            data = json.load(f)
            assert 'export_timestamp' in data
            assert 'total_methods_with_errors' in data
            assert 'methods' in data
    
    def test_export_error_data_empty(self, analyzer, temp_db):
        """Test exporting error data with no errors."""
        export_path = os.path.join(os.path.dirname(temp_db), "export.json")
        result = analyzer.export_error_data(export_path)
        
        assert result is True
        assert os.path.exists(export_path)
    
    def test_export_error_data_error_handling(self, analyzer):
        """Test error handling in export_error_data."""
        with patch.object(analyzer, 'get_error_stats', side_effect=Exception("Error")):
            result = analyzer.export_error_data("export.json")
            assert result is False
    
    def test_clear_old_data(self, analyzer):
        """Test clearing old data."""
        # Track some errors
        for i in range(5):
            error = ValueError(f"Error {i}")
            analyzer.track_error(
                method_name="method1",
                estimator_type="estimator1",
                error=error,
                parameters={},
                array_size=100,
                fractional_order=0.5
            )
        
        # Clear data older than 0 days (should clear all)
        deleted = analyzer.clear_old_data(days_to_keep=0)
        
        assert deleted >= 0
        
        # Verify data was cleared
        stats = analyzer.get_error_stats()
        assert len(stats) == 0 or stats["method1"].total_errors == 0
    
    def test_clear_old_data_with_retention(self, analyzer):
        """Test clearing old data with retention period."""
        # Track some errors
        for i in range(3):
            error = ValueError(f"Error {i}")
            analyzer.track_error(
                method_name="method1",
                estimator_type="estimator1",
                error=error,
                parameters={},
                array_size=100,
                fractional_order=0.5
            )
        
        # Clear data older than 30 days (should keep recent data)
        deleted = analyzer.clear_old_data(days_to_keep=30)
        
        assert deleted >= 0
        
        # Recent data should still be there
        stats = analyzer.get_error_stats()
        assert len(stats) > 0
    
    def test_clear_old_data_error_handling(self, analyzer):
        """Test error handling in clear_old_data."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            deleted = analyzer.clear_old_data(days_to_keep=30)
            assert deleted == 0
    
    def test_error_hash_generation(self, analyzer):
        """Test error hash generation."""
        error1 = ValueError("Test error")
        error2 = ValueError("Test error")
        error3 = TypeError("Different error")
        
        hash1 = analyzer._generate_error_hash(error1, "method1", {"p": 1})
        hash2 = analyzer._generate_error_hash(error2, "method1", {"p": 1})
        hash3 = analyzer._generate_error_hash(error3, "method1", {"p": 1})
        
        # Same error should produce same hash
        assert hash1 == hash2
        # Different error should produce different hash
        assert hash1 != hash3
    
    def test_error_hash_with_different_parameters(self, analyzer):
        """Test error hash with different parameters."""
        error = ValueError("Test error")
        
        hash1 = analyzer._generate_error_hash(error, "method1", {"p": 1})
        hash2 = analyzer._generate_error_hash(error, "method1", {"p": 2})
        
        # Different parameters should produce different hash
        assert hash1 != hash2
    
    def test_store_error_event(self, analyzer):
        """Test storing error event."""
        event = ErrorEvent(
            timestamp=time.time(),
            method_name="test_method",
            estimator_type="test_estimator",
            error_type="ValueError",
            error_message="Test error",
            error_traceback="Traceback...",
            error_hash="hash123",
            parameters={"param": 1},
            array_size=100,
            fractional_order=0.5,
            execution_time=0.1,
            memory_usage=50.0,
            user_session_id="session1"
        )
        
        analyzer._store_error_event(event)
        
        # Verify event was stored
        stats = analyzer.get_error_stats()
        assert "test_method" in stats
    
    def test_store_error_event_error_handling(self, analyzer):
        """Test error handling in store_error_event."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            event = ErrorEvent(
                timestamp=time.time(),
                method_name="test_method",
                estimator_type="test_estimator",
                error_type="ValueError",
                error_message="Test error",
                error_traceback="Traceback...",
                error_hash="hash123",
                parameters={},
                array_size=100,
                fractional_order=0.5
            )
            
            # Should not raise exception
            analyzer._store_error_event(event)
    
    def test_database_fallback_on_error(self):
        """Test database fallback when primary path fails."""
        # Use invalid path that will fail
        invalid_path = "/invalid/path/that/does/not/exist/error_analytics.db"
        
        # Should fallback to temp directory
        analyzer = ErrorAnalyzer(db_path=invalid_path, enable_analysis=True)
        
        # Should still be able to track errors (using fallback path)
        error = ValueError("Test error")
        analyzer.track_error(
            method_name="test_method",
            estimator_type="test_estimator",
            error=error,
            parameters={},
            array_size=100,
            fractional_order=0.5
        )
        
        # Should work with fallback
        stats = analyzer.get_error_stats()
        assert isinstance(stats, dict)



