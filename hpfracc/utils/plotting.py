"""
Plotting and visualization tools for fractional calculus computations.

This module provides tools for:
- Creating comparison plots between different methods
- Plotting convergence analysis
- Visualizing error analysis
- Saving and managing plots
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Dict, List, Optional, Tuple
import os
import warnings


class PlotManager:
    """Manager for creating and managing plots."""

    def __init__(self, style: str = "default",
                 figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the plot manager.

        Args:
            style: Plotting style ('default', 'scientific', 'presentation')
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.setup_plotting_style(style)

    def setup_plotting_style(self, style: str = "default") -> None:
        """
        Setup plotting style.

        Args:
            style: Style to use ('default', 'scientific', 'presentation')
        """
        self.style = style
        if style == "scientific":
            plt.style.use("seaborn-v0_8-whitegrid")
            rcParams.update(
                {
                    "font.family": "serif",
                    "font.serif": ["Times New Roman", "DejaVu Serif"],
                    "font.size": 12,
                    "axes.titlesize": 14,
                    "axes.labelsize": 12,
                    "xtick.labelsize": 10,
                    "ytick.labelsize": 10,
                    "legend.fontsize": 10,
                    "figure.titlesize": 16,
                    "lines.linewidth": 2,
                    "lines.markersize": 6,
                    "grid.alpha": 0.3,
                    "axes.grid": True,
                }
            )
        elif style == "presentation":
            plt.style.use("seaborn-v0_8-darkgrid")
            rcParams.update(
                {
                    "font.family": "sans-serif",
                    "font.sans-serif": ["Arial", "DejaVu Sans"],
                    "font.size": 14,
                    "axes.titlesize": 18,
                    "axes.labelsize": 16,
                    "xtick.labelsize": 14,
                    "ytick.labelsize": 14,
                    "legend.fontsize": 14,
                    "figure.titlesize": 20,
                    "lines.linewidth": 3,
                    "lines.markersize": 8,
                    "grid.alpha": 0.5,
                    "axes.grid": True,
                }
            )
        else:  # default
            plt.style.use("default")
            rcParams.update(
                {
                    "font.size": 10,
                    "axes.titlesize": 12,
                    "axes.labelsize": 10,
                    "xtick.labelsize": 8,
                    "ytick.labelsize": 8,
                    "legend.fontsize": 8,
                    "figure.titlesize": 14,
                    "lines.linewidth": 1.5,
                    "lines.markersize": 4,
                }
            )

    def create_plot(
        self,
        x: np.ndarray,
        y: np.ndarray,
        title: str = "Plot",
        xlabel: str = "x",
        ylabel: str = "y",
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a simple plot.

        Args:
            x: x-axis data
            y: y-axis data
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            save_path: Path to save the plot (optional)

        Returns:
            Tuple of (figure, axes) objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(x, y, linewidth=2)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if save_path:
            self.save_plot(fig, save_path)

        return fig, ax

    def create_comparison_plot(
        self,
        data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
        title: str = "Comparison Plot",
        xlabel: str = "x",
        ylabel: str = "y",
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a comparison plot of multiple datasets.

        Args:
            data_dict: Dictionary of {label: (x_data, y_data)} pairs
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            save_path: Path to save the plot (optional)

        Returns:
            Tuple of (figure, axes) objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))

        for i, (label, (x_data, y_data)) in enumerate(data_dict.items()):
            ax.plot(x_data, y_data, color=colors[i], label=label, linewidth=2)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            self.save_plot(fig, save_path)

        return fig, ax

    def plot_convergence(
        self,
        grid_sizes: List[int],
        errors: Dict[str, List[float]],
        title: str = "Convergence Analysis",
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot convergence analysis.

        Args:
            grid_sizes: List of grid sizes
            errors: Dictionary of {metric: error_list} pairs
            title: Plot title
            save_path: Path to save the plot (optional)

        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot errors vs grid size
        colors = plt.cm.tab10(np.linspace(0, 1, len(errors)))

        for i, (metric, error_list) in enumerate(errors.items()):
            ax1.loglog(
                grid_sizes,
                error_list,
                "o-",
                color=colors[i],
                label=metric.upper(),
                linewidth=2,
                markersize=6,
            )

        ax1.set_title("Error vs Grid Size", fontweight="bold")
        ax1.set_xlabel("Grid Size (N)")
        ax1.set_ylabel("Error")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot convergence rates
        convergence_rates = {}
        for metric, error_list in errors.items():
            try:
                # Compute convergence rate using linear regression
                log_n = np.log(np.array(grid_sizes))
                log_error = np.log(np.array(error_list))
                coeffs = np.polyfit(log_n, log_error, 1)
                convergence_rate = -coeffs[0]
                convergence_rates[metric] = convergence_rate
            except Exception:
                convergence_rates[metric] = np.nan

        # Plot convergence rates
        metrics = list(convergence_rates.keys())
        rates = list(convergence_rates.values())

        bars = ax2.bar(metrics, rates, color=colors[: len(metrics)], alpha=0.7)
        ax2.set_title("Convergence Rates", fontweight="bold")
        ax2.set_ylabel("Convergence Rate (order)")
        ax2.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            if not np.isnan(rate):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{rate:.2f}",
                    ha="center",
                    va="bottom",
                )

        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            self.save_plot(fig, save_path)

        return fig

    def plot_error_analysis(
        self,
        x: np.ndarray,
        numerical: np.ndarray,
        analytical: np.ndarray,
        title: str = "Error Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot error analysis between numerical and analytical solutions.

        Args:
            x: x-axis data
            numerical: Numerical solution
            analytical: Analytical solution
            title: Plot title
            save_path: Path to save the plot (optional)

        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot solutions
        ax1.plot(x, numerical, "b-", label="Numerical", linewidth=2)
        ax1.plot(x, analytical, "r--", label="Analytical", linewidth=2)
        ax1.set_title("Solutions Comparison", fontweight="bold")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot absolute error
        abs_error = np.abs(numerical - analytical)
        ax2.plot(x, abs_error, "g-", linewidth=2)
        ax2.set_title("Absolute Error", fontweight="bold")
        ax2.set_xlabel("x")
        ax2.set_ylabel("|Error|")
        ax2.grid(True, alpha=0.3)

        # Plot relative error
        rel_error = np.abs(numerical - analytical) / \
            (np.abs(analytical) + 1e-12)
        ax3.plot(x, rel_error, "m-", linewidth=2)
        ax3.set_title("Relative Error", fontweight="bold")
        ax3.set_xlabel("x")
        ax3.set_ylabel("|Error|/|Analytical|")
        ax3.grid(True, alpha=0.3)

        # Error statistics
        error_stats = {
            "L1 Error": np.mean(abs_error),
            "L2 Error": np.sqrt(np.mean(abs_error**2)),
            "L∞ Error": np.max(abs_error),
            "Mean Rel. Error": np.mean(rel_error),
        }

        # Create text box with statistics
        stats_text = "\n".join(
            [f"{k}: {v:.2e}" for k, v in error_stats.items()])
        ax4.text(
            0.1,
            0.5,
            stats_text,
            transform=ax4.transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax4.set_title("Error Statistics", fontweight="bold")
        ax4.axis("off")

        plt.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if save_path:
            self.save_plot(fig, save_path)

        return fig

    def save_plot(
            self,
            fig: plt.Figure,
            path: str,
            dpi: int = 300,
            bbox_inches: str = "tight") -> None:
        """
        Save a plot to file.

        Args:
            fig: Matplotlib figure object
            path: File path to save to
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box setting
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
            print(f"Plot saved to: {path}")
        except Exception as e:
            warnings.warn(f"Failed to save plot to {path}: {e}")


# Convenience functions
def setup_plotting_style(style: str = "default") -> None:
    """Setup plotting style."""
    manager = PlotManager(style=style)
    manager.setup_plotting_style(style)


def create_comparison_plot(
    data_dict_or_x, data=None, title: str = "Comparison Plot",
    xlabel: str = "x", ylabel: str = "y", save_path: Optional[str] = None,
    labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a comparison plot of multiple datasets.

    Args:
        data_dict_or_x: Either a dict of {label: (x_data, y_data)} or x-axis data
        data: If first arg is x-axis data, this should be y-axis data (list of arrays or single array)
        title: Plot title
        xlabel: x-axis label
        ylabel: y-axis label
        save_path: Path to save the plot
        labels: Labels for datasets (when data is a list of arrays)
    """
    manager = PlotManager()

    if isinstance(data_dict_or_x, dict):
        # Called with data_dict
        data_dict = data_dict_or_x
        return manager.create_comparison_plot(
            data_dict, title, xlabel, ylabel, save_path
        )
    else:
        # Called with x, data format
        x_data = data_dict_or_x
        y_data = data
        if y_data is None:
            raise ValueError(
                "When x data is provided, y data must be provided as second argument")

        # Handle different formats for y_data
        if isinstance(y_data, list):
            # List of y arrays - use labels if provided
            if labels is None:
                labels = [f"Method {i+1}" for i in range(len(y_data))]
            elif len(labels) != len(y_data):
                raise ValueError("Number of labels must match number of datasets")
            
            data_dict = {label: (x_data, y_values)
                         for label, y_values in zip(labels, y_data)}
        elif isinstance(y_data, dict):
            # Convert dict of {label: y_values} to dict of {label: (x, y)}
            data_dict = {label: (x_data, y_values)
                         for label, y_values in y_data.items()}
        else:
            # Single y array
            label = labels[0] if labels and len(labels) > 0 else "data"
            data_dict = {label: (x_data, y_data)}
        return manager.create_comparison_plot(
            data_dict, title, xlabel, ylabel, save_path
        )


def plot_convergence(
    methods_or_grid_sizes, h_values_or_errors=None, errors=None,
    title: str = "Convergence Analysis", save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot convergence analysis.

    Args:
        methods_or_grid_sizes: Either list of method names or grid sizes (h_values)
        h_values_or_errors: Either h_values array or errors dict
        errors: Errors dict (only if first arg is methods) or errors array if first arg is h_values
        title: Plot title
        save_path: Path to save the plot
    """
    manager = PlotManager()

    # Handle different calling patterns
    if h_values_or_errors is None:
        # Called with just grid_sizes, errors
        if isinstance(methods_or_grid_sizes, (list, tuple, np.ndarray)) and len(methods_or_grid_sizes) > 0:
            if isinstance(methods_or_grid_sizes[0], (int, float, np.integer, np.floating)):
                # First arg is grid sizes, second should be errors dict
                grid_sizes = list(methods_or_grid_sizes)
                if errors is not None:
                    errors_dict = {method: list(
                        errors[method]) for method in errors.keys()}
                    fig = manager.plot_convergence(
                        grid_sizes, errors_dict, title, save_path)
                    return fig, fig.axes if fig.axes else None
                else:
                    raise ValueError(
                        "When grid_sizes is provided, errors dict must be provided as second argument")
            else:
                raise ValueError("Invalid grid_sizes format")
        else:
            raise ValueError("Invalid arguments provided")
    else:
        if errors is None:
            # Check if called with (h_values, errors) pattern where both are arrays
            if isinstance(methods_or_grid_sizes, (list, tuple, np.ndarray)) and isinstance(h_values_or_errors, (list, tuple, np.ndarray)):
                # Both are arrays - this is the (h_values, errors) pattern
                h_values = np.array(methods_or_grid_sizes)
                errors_array = np.array(h_values_or_errors)
                
                # Create a simple convergence plot
                fig, ax = plt.subplots(figsize=manager.figsize)
                ax.loglog(h_values, errors_array, 'o-', linewidth=2, markersize=6)
                ax.set_xlabel("Step Size (h)")
                ax.set_ylabel("Error")
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                
                if save_path:
                    manager.save_plot(fig, save_path)
                
                return fig, ax
            # Called with methods, h_values, errors
            elif isinstance(methods_or_grid_sizes, (list, tuple)) and all(isinstance(x, str) for x in methods_or_grid_sizes):
                methods = methods_or_grid_sizes
                h_values = np.array(h_values_or_errors)
                # errors should be the third argument but it's None, this is an error
                raise ValueError(
                    "When methods and h_values provided, errors dict must be provided as third argument")
            else:
                # Called with grid_sizes, errors
                grid_sizes = list(methods_or_grid_sizes)
                errors_dict = h_values_or_errors
                if isinstance(errors_dict, dict):
                    errors_dict = {method: list(
                        errors_dict[method]) for method in errors_dict.keys()}
                    fig = manager.plot_convergence(
                        grid_sizes, errors_dict, title, save_path)
                    return fig, fig.axes if fig.axes else None
                else:
                    raise ValueError("When grid_sizes provided, errors must be a dict")
        else:
            # Called with methods, h_values, errors
            methods = methods_or_grid_sizes
            h_values = np.array(h_values_or_errors)
            grid_sizes = h_values.tolist()
            errors_dict = {method: list(errors[method])
                           for method in methods if method in errors}
            fig = manager.plot_convergence(
                grid_sizes, errors_dict, title, save_path)
            return fig, fig.axes if fig.axes else None


def plot_error_analysis(
    *args, title: str = "Error Analysis", save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot error analysis between numerical and analytical solutions.
    
    Args:
        *args: Either (numerical, analytical) or (x, analytical, numerical)
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Tuple of (figure, axes)
    """
    manager = PlotManager()
    
    if len(args) == 2:
        # Called with (numerical, analytical)
        numerical, analytical = args
        x = np.arange(len(numerical))
    elif len(args) == 3:
        # Called with (x, analytical, numerical)
        x, analytical, numerical = args
    else:
        raise ValueError(
            "plot_error_analysis expects either 2 arguments (numerical, analytical) "
            "or 3 arguments (x, analytical, numerical)"
        )
    
    fig = manager.plot_error_analysis(
        x, numerical, analytical, title, save_path)
    axes = fig.axes if hasattr(fig, 'axes') and fig.axes else None
    return fig, axes


def save_plot(
    fig: plt.Figure, path: str, dpi: int = 300, bbox_inches: str = "tight"
) -> None:
    """Save a plot to file."""
    manager = PlotManager()
    manager.save_plot(fig, path, dpi, bbox_inches)
