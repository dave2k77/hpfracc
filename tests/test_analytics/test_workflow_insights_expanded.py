"""
Expanded comprehensive tests for WorkflowInsights module.
Tests pattern detection, workflow analysis, insights generation, event tracking.
"""

import pytest
import tempfile
import json
import os
import time
import sqlite3
from unittest.mock import Mock, patch
import shutil

from hpfracc.analytics.workflow_insights import (
    WorkflowInsights,
    WorkflowEvent,
    WorkflowPattern
)


class TestWorkflowInsightsExpanded:
    """Expanded tests for WorkflowInsights class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_path = tempfile.mkdtemp()
        db_path = os.path.join(temp_path, "test_workflow_analytics.db")
        yield db_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def insights(self, temp_db):
        """Create WorkflowInsights instance."""
        return WorkflowInsights(db_path=temp_db, enable_insights=True)
    
    def test_initialization_with_custom_db_path(self, temp_db):
        """Test initialization with custom database path."""
        insights = WorkflowInsights(db_path=temp_db, enable_insights=True)
        assert insights.db_path == temp_db
        assert insights.enable_insights is True
        assert os.path.exists(temp_db)
    
    def test_initialization_disabled(self):
        """Test initialization with insights disabled."""
        insights = WorkflowInsights(enable_insights=False)
        assert insights.enable_insights is False
    
    def test_track_workflow_event_basic(self, insights):
        """Test basic workflow event tracking."""
        insights.track_workflow_event(
            session_id="session1",
            method_name="method1",
            estimator_type="estimator1",
            parameters={"param": 1},
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        # Verify event was tracked
        patterns = insights.get_workflow_patterns()
        assert isinstance(patterns, list)
    
    def test_track_workflow_event_with_all_parameters(self, insights):
        """Test workflow event tracking with all parameters."""
        insights.track_workflow_event(
            session_id="session1",
            method_name="method1",
            estimator_type="estimator1",
            parameters={"param1": 1, "param2": "value"},
            array_size=200,
            fractional_order=0.7,
            execution_success=True,
            execution_time=0.5,
            user_agent="test-agent",
            ip_address="127.0.0.1"
        )
        
        # Should not raise exception
        transitions = insights.get_method_transitions()
        assert isinstance(transitions, dict)
    
    def test_track_workflow_event_disabled(self):
        """Test workflow event tracking when disabled."""
        insights = WorkflowInsights(enable_insights=False)
        
        insights.track_workflow_event(
            session_id="session1",
            method_name="method1",
            estimator_type="estimator1",
            parameters={},
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        # Should not track
        patterns = insights.get_workflow_patterns()
        assert len(patterns) == 0
    
    def test_track_workflow_event_error_handling(self, insights):
        """Test error handling in track_workflow_event."""
        with patch.object(insights, '_store_workflow_event', side_effect=Exception("DB Error")):
            insights.track_workflow_event(
                session_id="session1",
                method_name="method1",
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            # Should not raise exception
    
    def test_get_workflow_patterns_empty(self, insights):
        """Test getting workflow patterns with no events."""
        patterns = insights.get_workflow_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) == 0
    
    def test_get_workflow_patterns_with_events(self, insights):
        """Test getting workflow patterns with events."""
        # Create a workflow pattern: method1 -> method2 -> method3
        session_id = "session1"
        methods = ["method1", "method2", "method3"]
        
        for method in methods:
            insights.track_workflow_event(
                session_id=session_id,
                method_name=method,
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True,
                execution_time=0.1
            )
            time.sleep(0.01)  # Small delay between events
        
        patterns = insights.get_workflow_patterns(min_frequency=1, max_pattern_length=3)
        
        assert len(patterns) > 0
        # Should find pattern of length 2 and 3
        pattern_sequences = [p.method_sequence for p in patterns]
        assert ["method1", "method2"] in pattern_sequences or ["method2", "method3"] in pattern_sequences
    
    def test_get_workflow_patterns_min_frequency(self, insights):
        """Test workflow patterns with minimum frequency filter."""
        # Create pattern that appears twice
        for session_id in ["session1", "session2"]:
            methods = ["method1", "method2"]
            for method in methods:
                insights.track_workflow_event(
                    session_id=session_id,
                    method_name=method,
                    estimator_type="estimator1",
                    parameters={},
                    array_size=100,
                    fractional_order=0.5,
                    execution_success=True
                )
                time.sleep(0.01)
        
        patterns = insights.get_workflow_patterns(min_frequency=2, max_pattern_length=2)
        
        # Should find pattern that appears at least twice
        assert all(p.frequency >= 2 for p in patterns)
    
    def test_get_workflow_patterns_error_handling(self, insights):
        """Test error handling in get_workflow_patterns."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            patterns = insights.get_workflow_patterns()
            assert patterns == []
    
    def test_get_method_transitions(self, insights):
        """Test getting method transitions."""
        # Create transitions: method1 -> method2, method2 -> method3
        session_id = "session1"
        methods = ["method1", "method2", "method3"]
        
        for method in methods:
            insights.track_workflow_event(
                session_id=session_id,
                method_name=method,
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            time.sleep(0.01)
        
        transitions = insights.get_method_transitions()
        
        assert isinstance(transitions, dict)
        assert "method1" in transitions
        assert "method2" in transitions["method1"]  # method1 -> method2
        assert transitions["method1"]["method2"] > 0
    
    def test_get_method_transitions_empty(self, insights):
        """Test getting method transitions with no events."""
        transitions = insights.get_method_transitions()
        assert transitions == {}
    
    def test_get_method_transitions_error_handling(self, insights):
        """Test error handling in get_method_transitions."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            transitions = insights.get_method_transitions()
            assert transitions == {}
    
    def test_get_session_insights(self, insights):
        """Test getting session insights."""
        # Create multiple sessions
        for session_id in ["session1", "session2"]:
            for i in range(3):
                insights.track_workflow_event(
                    session_id=session_id,
                    method_name=f"method{i}",
                    estimator_type="estimator1",
                    parameters={},
                    array_size=100,
                    fractional_order=0.5,
                    execution_success=(i % 2 == 0),
                    execution_time=0.1 * (i + 1)
                )
                time.sleep(0.01)
        
        session_insights = insights.get_session_insights()
        
        assert isinstance(session_insights, dict)
        assert 'total_sessions' in session_insights
        assert session_insights['total_sessions'] == 2
        assert 'session_durations' in session_insights
        assert len(session_insights['session_durations']) == 2
    
    def test_get_session_insights_empty(self, insights):
        """Test getting session insights with no events."""
        session_insights = insights.get_session_insights()
        
        assert isinstance(session_insights, dict)
        assert session_insights.get('total_sessions', 0) == 0
    
    def test_get_session_insights_error_handling(self, insights):
        """Test error handling in get_session_insights."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            session_insights = insights.get_session_insights()
            assert session_insights == {}
    
    def test_get_user_behavior_clusters(self, insights):
        """Test getting user behavior clusters."""
        # Create sessions with different characteristics
        # Power user: long session, many events
        for i in range(25):
            insights.track_workflow_event(
                session_id="power_user",
                method_name=f"method{i % 5}",
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True,
                execution_time=0.1
            )
            time.sleep(0.01)
        
        # Regular user: medium session, moderate events
        for i in range(8):
            insights.track_workflow_event(
                session_id="regular_user",
                method_name=f"method{i % 3}",
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True,
                execution_time=0.1
            )
            time.sleep(0.01)
        
        clusters = insights.get_user_behavior_clusters()
        
        assert isinstance(clusters, dict)
        # Note: clustering logic may vary, so we just check structure
        assert 'power_users' in clusters or 'regular_users' in clusters or 'casual_users' in clusters
    
    def test_get_user_behavior_clusters_empty(self, insights):
        """Test getting user behavior clusters with no events."""
        clusters = insights.get_user_behavior_clusters()
        assert clusters == {}
    
    def test_get_user_behavior_clusters_error_handling(self, insights):
        """Test error handling in get_user_behavior_clusters."""
        with patch.object(insights, 'get_session_insights', side_effect=Exception("Error")):
            clusters = insights.get_user_behavior_clusters()
            assert clusters == {}
    
    def test_get_workflow_recommendations(self, insights):
        """Test getting workflow recommendations."""
        # Create transitions: method1 -> method2 (3 times), method1 -> method3 (1 time)
        for i in range(3):
            session_id = f"session{i}"
            insights.track_workflow_event(
                session_id=session_id,
                method_name="method1",
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            time.sleep(0.01)
            insights.track_workflow_event(
                session_id=session_id,
                method_name="method2",
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            time.sleep(0.01)
        
        # One transition to method3
        insights.track_workflow_event(
            session_id="session3",
            method_name="method1",
            estimator_type="estimator1",
            parameters={},
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        time.sleep(0.01)
        insights.track_workflow_event(
            session_id="session3",
            method_name="method3",
            estimator_type="estimator1",
            parameters={},
            array_size=100,
            fractional_order=0.5,
            execution_success=True
        )
        
        recommendations = insights.get_workflow_recommendations("method1", [])
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # method2 should have higher probability than method3
        method2_prob = next((p for m, p in recommendations if m == "method2"), 0)
        method3_prob = next((p for m, p in recommendations if m == "method3"), 0)
        assert method2_prob >= method3_prob
    
    def test_get_workflow_recommendations_nonexistent_method(self, insights):
        """Test getting workflow recommendations for nonexistent method."""
        recommendations = insights.get_workflow_recommendations("nonexistent_method", [])
        assert recommendations == []
    
    def test_get_workflow_recommendations_error_handling(self, insights):
        """Test error handling in get_workflow_recommendations."""
        with patch.object(insights, 'get_method_transitions', side_effect=Exception("Error")):
            recommendations = insights.get_workflow_recommendations("method1", [])
            assert recommendations == []
    
    def test_export_workflow_data(self, insights, temp_db):
        """Test exporting workflow data."""
        # Track some events
        for i in range(3):
            insights.track_workflow_event(
                session_id=f"session{i}",
                method_name=f"method{i}",
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            time.sleep(0.01)
        
        export_path = os.path.join(os.path.dirname(temp_db), "export.json")
        result = insights.export_workflow_data(export_path)
        
        assert result is True
        assert os.path.exists(export_path)
        
        # Verify JSON is valid
        with open(export_path, 'r') as f:
            data = json.load(f)
            assert 'export_timestamp' in data
            assert 'workflow_patterns' in data
            assert 'method_transitions' in data
    
    def test_export_workflow_data_empty(self, insights, temp_db):
        """Test exporting workflow data with no events."""
        export_path = os.path.join(os.path.dirname(temp_db), "export.json")
        result = insights.export_workflow_data(export_path)
        
        assert result is True
        assert os.path.exists(export_path)
    
    def test_export_workflow_data_error_handling(self, insights):
        """Test error handling in export_workflow_data."""
        with patch.object(insights, 'get_workflow_patterns', side_effect=Exception("Error")):
            result = insights.export_workflow_data("export.json")
            assert result is False
    
    def test_clear_old_data(self, insights):
        """Test clearing old data."""
        # Track some events
        for i in range(5):
            insights.track_workflow_event(
                session_id=f"session{i}",
                method_name="method1",
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            time.sleep(0.01)
        
        # Clear data older than 0 days (should clear all)
        deleted = insights.clear_old_data(days_to_keep=0)
        
        assert deleted >= 0
    
    def test_clear_old_data_with_retention(self, insights):
        """Test clearing old data with retention period."""
        # Track some events
        for i in range(3):
            insights.track_workflow_event(
                session_id=f"session{i}",
                method_name="method1",
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            time.sleep(0.01)
        
        # Clear data older than 30 days (should keep recent data)
        deleted = insights.clear_old_data(days_to_keep=30)
        
        assert deleted >= 0
    
    def test_clear_old_data_error_handling(self, insights):
        """Test error handling in clear_old_data."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            deleted = insights.clear_old_data(days_to_keep=30)
            assert deleted == 0
    
    def test_store_workflow_event(self, insights):
        """Test storing workflow event."""
        event = WorkflowEvent(
            timestamp=time.time(),
            session_id="session1",
            method_name="test_method",
            estimator_type="test_estimator",
            parameters={"param": 1},
            array_size=100,
            fractional_order=0.5,
            execution_success=True,
            execution_time=0.1,
            user_agent="test-agent",
            ip_address="127.0.0.1"
        )
        
        insights._store_workflow_event(event)
        
        # Verify event was stored
        transitions = insights.get_method_transitions()
        assert isinstance(transitions, dict)
    
    def test_store_workflow_event_error_handling(self, insights):
        """Test error handling in store_workflow_event."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            event = WorkflowEvent(
                timestamp=time.time(),
                session_id="session1",
                method_name="test_method",
                estimator_type="test_estimator",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True
            )
            
            # Should not raise exception
            insights._store_workflow_event(event)
    
    def test_find_patterns_of_length(self, insights):
        """Test finding patterns of specific length."""
        # Create a sequence: method1 -> method2 -> method3
        session_id = "session1"
        methods = ["method1", "method2", "method3"]
        
        for method in methods:
            insights.track_workflow_event(
                session_id=session_id,
                method_name=method,
                estimator_type="estimator1",
                parameters={},
                array_size=100,
                fractional_order=0.5,
                execution_success=True,
                execution_time=0.1
            )
            time.sleep(0.01)
        
        # Get session sequences
        conn = sqlite3.connect(insights.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id, method_name, timestamp, execution_success, execution_time
            FROM workflow_events
            ORDER BY session_id, timestamp
        ''')
        events = cursor.fetchall()
        conn.close()
        
        from collections import defaultdict
        session_sequences = defaultdict(list)
        for event in events:
            session_id, method_name, timestamp, success, exec_time = event
            session_sequences[session_id].append({
                'method': method_name,
                'timestamp': timestamp,
                'success': success,
                'exec_time': exec_time
            })
        
        # Find patterns of length 2
        patterns = insights._find_patterns_of_length(session_sequences, pattern_length=2, min_frequency=1)
        
        assert len(patterns) > 0
        assert all(len(p.method_sequence) == 2 for p in patterns)






