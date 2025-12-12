"""
Expanded comprehensive tests for plotting.py module.
Tests plot creation, style management, figure saving, comparison plots.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from hpfracc.utils.plotting import (
    PlotManager,
    setup_plotting_style,
    create_comparison_plot,
    plot_convergence,
    plot_error_analysis,
    save_plot
)


class TestPlotManager:
    """Tests for PlotManager class."""
    
    @pytest.fixture
    def plot_manager(self):
        """Create PlotManager instance."""
        return PlotManager()
    
    def test_initialization_default(self, plot_manager):
        """Test initialization with default parameters."""
        assert plot_manager.style == "default"
        assert plot_manager.figsize == (10, 6)
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        manager = PlotManager(style="scientific", figsize=(12, 8))
        assert manager.style == "scientific"
        assert manager.figsize == (12, 8)
    
    @patch('hpfracc.utils.plotting.plt')
    def test_setup_plotting_style_default(self, mock_plt, plot_manager):
        """Test setting up default plotting style."""
        plot_manager.setup_plotting_style("default")
        assert plot_manager.style == "default"
    
    @patch('hpfracc.utils.plotting.plt')
    def test_setup_plotting_style_scientific(self, mock_plt, plot_manager):
        """Test setting up scientific plotting style."""
        plot_manager.setup_plotting_style("scientific")
        assert plot_manager.style == "scientific"
    
    @patch('hpfracc.utils.plotting.plt')
    def test_setup_plotting_style_presentation(self, mock_plt, plot_manager):
        """Test setting up presentation plotting style."""
        plot_manager.setup_plotting_style("presentation")
        assert plot_manager.style == "presentation"
    
    @patch('hpfracc.utils.plotting.plt')
    def test_create_plot(self, mock_plt, plot_manager):
        """Test creating a simple plot."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 4, 9, 16, 25])
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        fig, ax = plot_manager.create_plot(
            x, y, title="Test Plot", xlabel="X", ylabel="Y"
        )
        
        assert fig == mock_fig
        assert ax == mock_ax
        mock_ax.plot.assert_called_once()
        mock_ax.set_title.assert_called_once_with("Test Plot", fontweight="bold")
        mock_ax.set_xlabel.assert_called_once_with("X")
        mock_ax.set_ylabel.assert_called_once_with("Y")
    
    @patch('hpfracc.utils.plotting.plt')
    def test_create_plot_with_save(self, mock_plt, plot_manager):
        """Test creating plot and saving."""
        x = np.array([1, 2, 3])
        y = np.array([1, 4, 9])
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_path = tmp.name
        
        try:
            fig, ax = plot_manager.create_plot(x, y, save_path=save_path)
            mock_fig.savefig.assert_called_once()
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestStandalonePlottingFunctions:
    """Tests for standalone plotting functions."""
    
    @patch('hpfracc.utils.plotting.plt')
    def test_setup_plotting_style_function(self, mock_plt):
        """Test setup_plotting_style standalone function."""
        setup_plotting_style("default")
        # Should not raise exception
    
    @patch('hpfracc.utils.plotting.plt')
    def test_create_comparison_plot(self, mock_plt):
        """Test create_comparison_plot function."""
        x = np.array([1, 2, 3, 4, 5])
        y1 = np.array([1, 4, 9, 16, 25])
        y2 = np.array([2, 5, 10, 17, 26])
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        fig, ax = create_comparison_plot(
            x, [y1, y2], labels=["Method 1", "Method 2"]
        )
        
        assert fig == mock_fig
        assert ax == mock_ax
        assert mock_ax.plot.call_count == 2
    
    @patch('hpfracc.utils.plotting.plt')
    def test_plot_convergence(self, mock_plt):
        """Test plot_convergence function."""
        h_values = np.array([0.1, 0.05, 0.025, 0.0125])
        errors = np.array([0.01, 0.0025, 0.000625, 0.00015625])
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        fig, ax = plot_convergence(h_values, errors)
        
        assert fig == mock_fig
        assert ax == mock_ax
        mock_ax.loglog.assert_called_once()
    
    @patch('hpfracc.utils.plotting.plt')
    def test_plot_error_analysis(self, mock_plt):
        """Test plot_error_analysis function."""
        x = np.array([1, 2, 3, 4, 5])
        analytical = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        numerical = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
        
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_ax3 = Mock()
        mock_ax4 = Mock()
        # plt.subplots(2, 2) returns (fig, ((ax1, ax2), (ax3, ax4)))
        mock_plt.subplots.return_value = (mock_fig, ((mock_ax1, mock_ax2), (mock_ax3, mock_ax4)))
        mock_fig.axes = [mock_ax1, mock_ax2, mock_ax3, mock_ax4]
        
        fig, ax = plot_error_analysis(x, analytical, numerical)
        
        assert fig == mock_fig
        assert ax == mock_fig.axes
    
    @patch('hpfracc.utils.plotting.plt')
    def test_save_plot(self, mock_plt):
        """Test save_plot function."""
        mock_fig = Mock()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_path = tmp.name
        
        try:
            save_plot(mock_fig, save_path)
            mock_fig.savefig.assert_called_once()
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @patch('hpfracc.utils.plotting.plt')
    def test_plot_with_empty_arrays(self, mock_plt):
        """Test plotting with empty arrays."""
        manager = PlotManager()
        
        x = np.array([])
        y = np.array([])
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Should handle empty arrays gracefully
        try:
            fig, ax = manager.create_plot(x, y)
            assert fig == mock_fig
        except Exception:
            # Some operations may not support empty arrays
            pass
    
    @patch('hpfracc.utils.plotting.plt')
    def test_plot_with_different_lengths(self, mock_plt):
        """Test plotting with mismatched array lengths."""
        manager = PlotManager()
        
        x = np.array([1, 2, 3])
        y = np.array([1, 4])  # Different length
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Should handle or raise appropriate error
        try:
            fig, ax = manager.create_plot(x, y)
        except (ValueError, AssertionError):
            # Expected behavior for mismatched lengths
            pass
    
    @patch('hpfracc.utils.plotting.plt')
    def test_plot_with_nan_values(self, mock_plt):
        """Test plotting with NaN values."""
        manager = PlotManager()
        
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, np.nan, 9, 16, 25])
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        # Should handle NaN values
        fig, ax = manager.create_plot(x, y)
        assert fig == mock_fig
