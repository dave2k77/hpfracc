"""
HPFRACC MLOps Integration Demo
==============================

This script demonstrates the MLOps capabilities of the HPFRACC library, including:
1. Model Registry: Versioning, metadata, and artifact storage
2. Development Workflow: Experiment tracking and validation
3. Production Workflow: Quality gates and promotion management

"""

import torch
import numpy as np
import sys
import os
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from hpfracc.ml import (
    FractionalNeuralNetwork,
    DevelopmentWorkflow,
    ProductionWorkflow,
    DeploymentStatus,
    QualityGate,
    QualityMetric,
    QualityThreshold,
    BackendManager
)

def clean_previous_run():
    """Clean up registry and models directory from previous runs"""
    if os.path.exists("models"):
        shutil.rmtree("models", ignore_errors=True)
    if os.path.exists("registry.db"):
        try:
            os.remove("registry.db")
        except OSError:
            pass
    print("🧹 Cleaned up previous run data")

def main():
    print("🚀 Starting MLOps Workflow Demo")
    print("================================")
    
    # Ensure clean slate
    clean_previous_run()

    # 1. Initialize Workflows
    # -----------------------
    print("\n📦 Initializing Workflows...")
    
    # Development workflow manages experiments and model registration
    dev_workflow = DevelopmentWorkflow()
    
    # Production workflow manages deployment and promotion
    prod_workflow = ProductionWorkflow(registry=dev_workflow.registry)
    
    print("✅ Workflows initialized")

    # 2. Experiment & Model Creation
    # ------------------------------
    print("\n🧪 Starting Experiment: 'Fractional Optimization'")
    experiment = dev_workflow.create_experiment(
        name="Fractional Optimization",
        description="Optimizing alpha for time series forecasting",
        tags=["fractional", "time-series", "forecasting"]
    )
    
    # Create a simple Fractional Neural Network
    print("🧠 Creating Fractional Neural Network (alpha=0.6)...")
    model = FractionalNeuralNetwork(
        input_size=10,
        hidden_sizes=[32, 16],
        output_size=1,
        fractional_order=0.6
    )
    
    # 3. Model Registration (Development)
    # -----------------------------------
    print("\n📝 Registering model in Development...")
    model_id = dev_workflow.register_development_model(
        model=model,
        name="alpha_forecaster",
        version="0.1.0",
        description="Initial prototype with alpha=0.6",
        author="DataScientist_Using_Hpfracc",
        tags=["prototype", "v0.1"],
        fractional_order=0.6,
        hyperparameters={
            "input_size": 10,
            "hidden_sizes": [32, 16],
            "output_size": 1,
            "fractional_order": 0.6,
            "learning_rate": 0.001
        },
        performance_metrics={},  # Not yet evaluated
        dataset_info={"name": "synthetic_lorenz", "size": 1000},
        dependencies={"torch": torch.__version__}
    )
    print(f"✅ Model registered with ID: {model_id}")

    # 4. Validation & Quality Gates
    # -----------------------------
    print("\n🛡️ Running Validation (Quality Gates)...")
    
    # Create dummy validation data
    val_data = torch.randn(100, 10)
    val_labels = torch.randn(100, 1) # Dummy targets
    
    # In a real scenario, this would compute actual loss/accuracy
    # For demo, we inject custom metrics to simulate a good model
    mock_metrics = {
        "accuracy": 0.85,
        "loss": 0.25,
        "inference_time": 45.0  # ms
    }
    
    validation_results = dev_workflow.validate_development_model(
        model_id=model_id,
        test_data=val_data,
        test_labels=val_labels,
        custom_metrics=mock_metrics
    )
    
    if validation_results['validation_passed']:
        print(f"✅ Validation PASSED (Score: {validation_results['final_score']:.2f})")
    else:
        print("❌ Validation FAILED")
        exit(1)

    # 5. Production Promotion
    # -----------------------
    print("\n🚀 Promoting to Production...")
    
    # Attempt promotion
    promote_result = prod_workflow.promote_to_production(
        model_id=model_id,
        version="0.1.0",
        test_data=val_data,
        test_labels=val_labels,
        custom_metrics=mock_metrics
    )
    
    if promote_result['promoted']:
        print(f"✅ Model promoted successfully at {promote_result['promoted_at']}")
    else:
        print(f"❌ Promotion failed: {promote_result.get('reason')}")
        exit(1)

    # 6. Verify Production State
    # --------------------------
    print("\n🔍 Verifying Registry State...")
    prod_models = prod_workflow.registry.get_production_models()
    
    print(f"Found {len(prod_models)} production model(s):")
    for pm in prod_models:
        print(f" - {pm.metadata.name} v{pm.version} (ID: {pm.model_id})")
        print(f"   Status: {pm.metadata.deployment_status}")

    # 7. Model Loading from Registry
    # ------------------------------
    print("\n📥 Loading model from registry...")
    loaded_model = prod_workflow.registry.reconstruct_model(model_id, version="0.1.0")
    
    if loaded_model:
        print(f"✅ Model loaded successfully: {type(loaded_model).__name__}")
        # Verify it works
        with torch.no_grad():
            output = loaded_model(val_data[:1])
        print(f"   Inference check: Output shape {output.shape}")
    else:
        print("❌ Failed to load model")

    print("\n🎉 MLOps Workflow Demo Complete!")

if __name__ == "__main__":
    main()
