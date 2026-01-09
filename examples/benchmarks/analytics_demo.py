#!/usr/bin/env python3
"""
Demo script for the hpfracc analytics system.

This script demonstrates how to use the comprehensive analytics system
to track usage patterns, performance metrics, error analysis, and workflow insights.
"""

from hpfracc.analytics import AnalyticsManager, AnalyticsConfig
import sys
import time
import numpy as np
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def demo_basic_analytics():
    """Demonstrate basic analytics functionality."""
    print("🚀 HPFRACC Analytics System Demo")
    print("=" * 50)

    # Configure analytics system
    config = AnalyticsConfig(
        enable_usage_tracking=True,
        enable_performance_monitoring=True,
        enable_error_analysis=True,
        enable_workflow_insights=True,
        export_format="html",
        generate_reports=True
    )

    # Initialize analytics manager
    analytics = AnalyticsManager(config)
    print("✅ Analytics manager initialized")

    # Simulate some method calls
    print("\n📊 Simulating method calls for analytics...")

    methods = [
        ("optimized_riemann_liouville", "RL", {"method": "fft"}),
        ("optimized_caputo", "Caputo", {"method": "l1"}),
        ("optimized_grunwald_letnikov", "GL", {"method": "direct"}),
        ("optimized_weyl_derivative", "Weyl", {}),
        ("optimized_marchaud_derivative", "Marchaud", {})
    ]

    array_sizes = [100, 500, 1000, 2000]
    fractional_orders = [0.25, 0.5, 0.75]

    for i in range(20):  # Simulate 20 method calls
        method_name, estimator_type, parameters = methods[i % len(methods)]
        array_size = array_sizes[i % len(array_sizes)]
        alpha = fractional_orders[i % len(fractional_orders)]

        # Simulate execution time and success
        execution_time = np.random.exponential(0.1) + 0.01
        success = np.random.random() > 0.05  # 95% success rate

        # Track the method call
        analytics.track_method_call(
            method_name=method_name,
            estimator_type=estimator_type,
            parameters=parameters,
            array_size=array_size,
            fractional_order=alpha,
            execution_success=success,
            execution_time=execution_time,
            memory_usage=np.random.exponential(10) + 5
        )

        # Simulate some errors
        if not success:
            try:
                raise ValueError(f"Simulated error in {method_name}")
            except Exception as e:
                analytics.track_method_call(
                    method_name=method_name,
                    estimator_type=estimator_type,
                    parameters=parameters,
                    array_size=array_size,
                    fractional_order=alpha,
                    execution_success=False,
                    execution_time=execution_time,
                    error=e
                )

        time.sleep(0.1)  # Small delay to simulate real usage

    print("✅ Simulated method calls completed")

    # Generate analytics report
    print("\n📈 Generating analytics report...")
    report_path = analytics.generate_analytics_report()
    print(f"✅ Analytics report generated: {report_path}")

    # Get comprehensive analytics
    print("\n🔍 Getting comprehensive analytics...")
    analytics_data = analytics.get_comprehensive_analytics()

    # Display key insights
    print("\n📊 Key Analytics Insights:")
    print("-" * 30)

    if 'usage' in analytics_data:
        usage = analytics_data['usage']
        print(f"📈 Total methods tracked: {len(usage['stats'])}")
        if usage['popular_methods']:
            top_method = usage['popular_methods'][0]
            print(
                f"🏆 Most popular method: {top_method[0]} ({top_method[1]} calls)")

    if 'performance' in analytics_data:
        perf = analytics_data['performance']
        print(f"⚡ Performance data for {len(perf['stats'])} methods")
        if perf['bottlenecks']:
            bottlenecks = perf['bottlenecks']
            if bottlenecks['slowest_methods']:
                slowest = bottlenecks['slowest_methods'][0]
                print(f"🐌 Slowest method: {slowest[0]} ({slowest[1]:.6f}s)")

    if 'errors' in analytics_data:
        errors = analytics_data['errors']
        print(f"🚨 Error data for {len(errors['stats'])} methods")
        if errors['reliability_ranking']:
            most_reliable = errors['reliability_ranking'][0]
            print(
                f"🛡️ Most reliable method: {most_reliable[0]} (score: {most_reliable[1]:.3f})")

    if 'workflow' in analytics_data:
        workflow = analytics_data['workflow']
        print(f"🔄 Workflow patterns discovered: {len(workflow['patterns'])}")
        if workflow['patterns']:
            top_pattern = workflow['patterns'][0]
            print(
                f"🔄 Most common pattern: {' → '.join(top_pattern.method_sequence)} ({top_pattern.frequency} times)")

    # Export all data
    print("\n💾 Exporting analytics data...")
    export_paths = analytics.export_all_data()
    print(f"✅ Exported {len(export_paths)} datasets:")
    for category, path in export_paths.items():
        print(f"   {category}: {path}")

    return analytics


def demo_advanced_features(analytics):
    """Demonstrate advanced analytics features."""
    print("\n🔬 Advanced Analytics Features Demo")
    print("=" * 40)

    # Performance monitoring with context manager
    print("\n⚡ Performance monitoring with context manager...")

    with analytics.monitor_method_performance(
        method_name="test_method",
        estimator_type="test",
        array_size=1000,
        fractional_order=0.5,
        parameters={"test": True}
    ):
        # Simulate some computation
        time.sleep(0.5)
        result = np.random.random(1000)
        _ = np.sum(result)

    print("✅ Performance monitoring completed")

    # Get specific analytics
    print("\n📊 Getting specific analytics...")

    # Usage trends for a specific method
    usage_trends = analytics.usage_tracker.get_method_trends(
        "optimized_riemann_liouville", days=7)
    print(f"📈 Usage trends for RL method: {len(usage_trends)} data points")

    # Performance bottlenecks
    bottlenecks = analytics.performance_monitor.get_bottleneck_analysis()
    print(f"🔍 Performance bottlenecks analyzed: {len(bottlenecks)} categories")

    # Error patterns
    error_patterns = analytics.error_analyzer.get_common_error_patterns()
    print(f"🚨 Common error patterns: {len(error_patterns)} categories")

    # Workflow recommendations
    recommendations = analytics.workflow_insights.get_workflow_recommendations(
        current_method="optimized_riemann_liouville",
        user_history=["optimized_caputo", "optimized_riemann_liouville"]
    )
    print(f"💡 Workflow recommendations: {len(recommendations)} suggestions")

    # Cleanup old data
    print("\n🧹 Cleaning up old analytics data...")
    cleanup_results = analytics.cleanup_old_data()
    print(f"✅ Cleanup completed: {cleanup_results}")


def demo_analytics_configuration():
    """Demonstrate different analytics configurations."""
    print("\n⚙️ Analytics Configuration Demo")
    print("=" * 35)

    # Minimal configuration (usage tracking only)
    print("\n📊 Minimal configuration (usage tracking only)...")
    minimal_config = AnalyticsConfig(
        enable_usage_tracking=True,
        enable_performance_monitoring=False,
        enable_error_analysis=False,
        enable_workflow_insights=False
    )

    minimal_analytics = AnalyticsManager(minimal_config)
    print("✅ Minimal analytics manager initialized")

    # Track a simple method call
    minimal_analytics.track_method_call(
        method_name="test_minimal",
        estimator_type="test",
        parameters={},
        array_size=100,
        fractional_order=0.5,
        execution_success=True
    )

    # Get analytics (should only have usage data)
    minimal_data = minimal_analytics.get_comprehensive_analytics()
    print(f"📈 Minimal analytics data keys: {list(minimal_data.keys())}")

    # Performance-focused configuration
    print("\n⚡ Performance-focused configuration...")
    perf_config = AnalyticsConfig(
        enable_usage_tracking=False,
        enable_performance_monitoring=True,
        enable_error_analysis=False,
        enable_workflow_insights=False,
        export_format="csv"
    )

    perf_analytics = AnalyticsManager(perf_config)
    print("✅ Performance-focused analytics manager initialized")

    # Monitor performance
    with perf_analytics.monitor_method_performance(
        method_name="test_performance",
        estimator_type="test",
        array_size=2000,
        fractional_order=0.75,
        parameters={"method": "test"}
    ):
        time.sleep(0.3)
        _ = np.random.random(2000)

    # Get performance analytics
    perf_data = perf_analytics.get_comprehensive_analytics()
    print(f"⚡ Performance analytics data keys: {list(perf_data.keys())}")


def main():
    """Main demo function."""
    try:
        # Basic analytics demo
        analytics = demo_basic_analytics()

        # Advanced features demo
        demo_advanced_features(analytics)

        # Configuration demo
        demo_analytics_configuration()

        print("\n🎉 Analytics demo completed successfully!")
        print("\n📁 Check the following directories for generated files:")
        print("   - analytics_reports/ (HTML reports and plots)")
        print("   - *.db files (SQLite databases)")
        print("   - *.json files (exported data)")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
