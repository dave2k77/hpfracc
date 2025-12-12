"""
Comprehensive tests for hpfracc.ml.data module

This module tests all data loading utilities and dataset classes
for fractional calculus machine learning applications.
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Tuple

from hpfracc.ml.data import (
    FractionalDataset,
    FractionalTensorDataset,
    FractionalTimeSeriesDataset,
    FractionalGraphDataset,
    FractionalDataLoader,
    FractionalDataProcessor,
    FractionalBatchSampler,
    FractionalCollateFunction,
    FractionalDataModule,
    create_fractional_dataset,
    create_fractional_dataloader,
    create_fractional_datamodule
)
from hpfracc.ml.backends import BackendType
from hpfracc.core.definitions import FractionalOrder


class TestFractionalDataset:
    """Test the base FractionalDataset class"""

    def test_initialization_default(self):
        """Test dataset initialization with default parameters"""
        dataset = FractionalDataset()
        
        assert dataset.fractional_order.alpha == 0.5
        assert dataset.method == "RL"
        assert dataset.backend == BackendType.TORCH
        assert dataset.apply_fractional == True

    def test_initialization_custom(self):
        """Test dataset initialization with custom parameters"""
        dataset = FractionalDataset(
            fractional_order=0.7,
            method="Caputo",
            backend=BackendType.NUMBA,
            apply_fractional=False
        )
        
        assert dataset.fractional_order.alpha == 0.7
        assert dataset.method == "Caputo"
        assert dataset.backend == BackendType.NUMBA
        assert dataset.apply_fractional == False

    def test_fractional_transform_torch_backend(self):
        """Test fractional transform for PyTorch backend"""
        dataset = FractionalDataset(backend=BackendType.TORCH)
        
        # Test with tensor input
        data = torch.tensor([1.0, 2.0, 3.0])
        transformed = dataset.fractional_transform(data)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == data.shape

    def test_fractional_transform_non_torch_backend(self):
        """Test fractional transform for non-PyTorch backend"""
        dataset = FractionalDataset(backend=BackendType.NUMBA)
        
        # Should return input unchanged
        data = np.array([1.0, 2.0, 3.0])
        transformed = dataset.fractional_transform(data)
        
        assert np.array_equal(transformed, data)

    def test_fractional_transform_disabled(self):
        """Test fractional transform when disabled"""
        dataset = FractionalDataset(apply_fractional=False)
        
        # Should return input unchanged
        data = torch.tensor([1.0, 2.0, 3.0])
        transformed = dataset.fractional_transform(data)
        
        assert torch.equal(transformed, data)

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError"""
        dataset = FractionalDataset()
        
        with pytest.raises(NotImplementedError):
            len(dataset)
        
        with pytest.raises(NotImplementedError):
            dataset[0]


class TestFractionalTensorDataset:
    """Test the FractionalTensorDataset class"""

    def test_initialization_valid(self):
        """Test initialization with valid tensors"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        
        dataset = FractionalTensorDataset(tensors)
        
        assert len(dataset) == 3
        assert len(dataset.tensors) == 2

    def test_initialization_empty_tensors(self):
        """Test initialization with empty tensors list"""
        with pytest.raises(ValueError, match="Tensors list cannot be empty"):
            FractionalTensorDataset([])

    def test_initialization_mismatched_dimensions(self):
        """Test initialization with mismatched tensor dimensions"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0])  # Different length
        ]
        
        with pytest.raises(ValueError, match="All tensors must have the same first dimension"):
            FractionalTensorDataset(tensors)

    def test_getitem_valid_index(self):
        """Test getting item with valid index"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        
        dataset = FractionalTensorDataset(tensors)
        
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert torch.equal(item[0], torch.tensor(1.0))
        assert torch.equal(item[1], torch.tensor(4.0))

    def test_getitem_invalid_index(self):
        """Test getting item with invalid index"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        
        dataset = FractionalTensorDataset(tensors)
        
        with pytest.raises(IndexError):
            dataset[5]  # Index out of range

    def test_getitem_negative_index(self):
        """Test getting item with negative index"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        
        dataset = FractionalTensorDataset(tensors)
        
        item = dataset[-1]
        assert torch.equal(item[0], torch.tensor(3.0))
        assert torch.equal(item[1], torch.tensor(6.0))

    def test_fractional_transform_applied(self):
        """Test that fractional transform is applied when enabled"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        
        dataset = FractionalTensorDataset(tensors, apply_fractional=True)
        
        item = dataset[0]
        # Should have fractional transform applied
        assert isinstance(item[0], torch.Tensor)
        assert isinstance(item[1], torch.Tensor)

    def test_fractional_transform_disabled(self):
        """Test that fractional transform is not applied when disabled"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        
        dataset = FractionalTensorDataset(tensors, apply_fractional=False)
        
        item = dataset[0]
        # Should return original tensors
        assert torch.equal(item[0], torch.tensor(1.0))
        assert torch.equal(item[1], torch.tensor(4.0))


class TestFractionalDataLoader:
    """Test the FractionalDataLoader class"""

    def test_initialization_default(self):
        """Test dataloader initialization with default parameters"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        dataset = FractionalTensorDataset(tensors)
        
        dataloader = FractionalDataLoader(dataset)
        
        assert dataloader.dataset == dataset
        assert dataloader.batch_size == 1
        assert dataloader.shuffle == False
        assert dataloader.num_workers == 0
        assert dataloader.pin_memory == False
        assert dataloader.drop_last == False

    def test_initialization_custom(self):
        """Test dataloader initialization with custom parameters"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        dataset = FractionalTensorDataset(tensors)
        
        dataloader = FractionalDataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        assert dataloader.batch_size == 2
        assert dataloader.shuffle == True
        assert dataloader.num_workers == 2
        assert dataloader.pin_memory == True
        assert dataloader.drop_last == True

    def test_iteration(self):
        """Test dataloader iteration"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        dataset = FractionalTensorDataset(tensors)
        dataloader = FractionalDataLoader(dataset, batch_size=2)
        
        batches = list(dataloader)
        
        assert len(batches) == 2  # 3 samples with batch_size=2
        assert len(batches[0]) == 2  # First batch has 2 samples
        assert len(batches[1]) == 1  # Second batch has 1 sample

    def test_batch_sampling(self):
        """Test batch sampling"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([5.0, 6.0, 7.0, 8.0])
        ]
        dataset = FractionalTensorDataset(tensors)
        dataloader = FractionalDataLoader(dataset, batch_size=2)
        
        batches = list(dataloader)
        
        # Check batch structure
        for batch in batches:
            assert isinstance(batch, tuple)
            assert len(batch) == 2  # X and y
            assert isinstance(batch[0], torch.Tensor)
            assert isinstance(batch[1], torch.Tensor)

    def test_shuffle(self):
        """Test shuffling functionality"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([5.0, 6.0, 7.0, 8.0])
        ]
        dataset = FractionalTensorDataset(tensors)
        
        # Test with shuffle=True
        dataloader_shuffled = FractionalDataLoader(dataset, batch_size=1, shuffle=True)
        batches_shuffled = list(dataloader_shuffled)
        
        # Test with shuffle=False
        dataloader_unshuffled = FractionalDataLoader(dataset, batch_size=1, shuffle=False)
        batches_unshuffled = list(dataloader_unshuffled)
        
        # Shuffled and unshuffled should be different (with high probability)
        assert len(batches_shuffled) == len(batches_unshuffled)
        assert len(batches_shuffled) == 4

    def test_drop_last(self):
        """Test drop_last functionality"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([5.0, 6.0, 7.0, 8.0])
        ]
        dataset = FractionalTensorDataset(tensors)
        
        # Test with drop_last=True
        dataloader_drop = FractionalDataLoader(dataset, batch_size=3, drop_last=True)
        batches_drop = list(dataloader_drop)
        
        # Test with drop_last=False
        dataloader_no_drop = FractionalDataLoader(dataset, batch_size=3, drop_last=False)
        batches_no_drop = list(dataloader_no_drop)
        
        assert len(batches_drop) == 1  # Only complete batch
        assert len(batches_no_drop) == 2  # Complete batch + incomplete batch


class TestFractionalBatchSampler:
    """Test the FractionalBatchSampler class"""

    def test_initialization_default(self):
        """Test batch sampler initialization with default parameters"""
        sampler = FractionalBatchSampler(10)
        
        assert sampler.dataset_size == 10
        assert sampler.batch_size == 1
        assert sampler.shuffle == False
        assert sampler.drop_last == False

    def test_initialization_custom(self):
        """Test batch sampler initialization with custom parameters"""
        sampler = FractionalBatchSampler(
            dataset_size=10,
            batch_size=3,
            shuffle=True,
            drop_last=True
        )
        
        assert sampler.dataset_size == 10
        assert sampler.batch_size == 3
        assert sampler.shuffle == True
        assert sampler.drop_last == True

    def test_iteration_no_shuffle(self):
        """Test batch sampler iteration without shuffle"""
        sampler = FractionalBatchSampler(dataset_size=5, batch_size=2, shuffle=False)
        
        batches = list(sampler)
        
        assert len(batches) == 3  # 5 samples with batch_size=2
        assert batches[0] == [0, 1]
        assert batches[1] == [2, 3]
        assert batches[2] == [4]

    def test_iteration_with_shuffle(self):
        """Test batch sampler iteration with shuffle"""
        sampler = FractionalBatchSampler(dataset_size=5, batch_size=2, shuffle=True)
        
        batches = list(sampler)
        
        assert len(batches) == 3  # 5 samples with batch_size=2
        # First batch should have 2 elements
        assert len(batches[0]) == 2
        # All indices should be unique and in range
        all_indices = []
        for batch in batches:
            all_indices.extend(batch)
        
        assert len(set(all_indices)) == len(all_indices)  # All unique
        assert all(0 <= idx < 5 for idx in all_indices)  # All in range

    def test_drop_last(self):
        """Test drop_last functionality"""
        sampler_drop = FractionalBatchSampler(dataset_size=5, batch_size=3, drop_last=True)
        batches_drop = list(sampler_drop)
        
        sampler_no_drop = FractionalBatchSampler(dataset_size=5, batch_size=3, drop_last=False)
        batches_no_drop = list(sampler_no_drop)
        
        assert len(batches_drop) == 1  # Only complete batch
        assert len(batches_no_drop) == 2  # Complete batch + incomplete batch

    def test_length(self):
        """Test batch sampler length"""
        sampler = FractionalBatchSampler(dataset_size=10, batch_size=3)
        
        assert len(sampler) == 4  # 10 samples with batch_size=3


class TestFractionalCollateFunction:
    """Test the FractionalCollateFunction class"""

    def test_initialization_default(self):
        """Test collate function initialization with default parameters"""
        collate_fn = FractionalCollateFunction()
        
        assert collate_fn.pad_value == 0.0
        assert collate_fn.pad_length == None
        assert collate_fn.fractional_order.alpha == 0.5
        assert collate_fn.method == "RL"
        assert collate_fn.backend == BackendType.TORCH

    def test_initialization_custom(self):
        """Test collate function initialization with custom parameters"""
        collate_fn = FractionalCollateFunction(
            pad_value=1.0,
            pad_length=10,
            fractional_order=0.7,
            method="Caputo",
            backend=BackendType.NUMBA
        )
        
        assert collate_fn.pad_value == 1.0
        assert collate_fn.pad_length == 10
        assert collate_fn.fractional_order.alpha == 0.7
        assert collate_fn.method == "Caputo"
        assert collate_fn.backend == BackendType.NUMBA

    def test_collate_tensors(self):
        """Test collating tensors"""
        collate_fn = FractionalCollateFunction()
        
        batch = [
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0])),
            (torch.tensor([4.0, 5.0]), torch.tensor([6.0])),
            (torch.tensor([7.0, 8.0]), torch.tensor([9.0]))
        ]
        
        collated = collate_fn(batch)
        
        assert isinstance(collated, tuple)
        assert len(collated) == 2  # X and y
        
        # Check X batch
        x_batch = collated[0]
        assert isinstance(x_batch, torch.Tensor)
        assert x_batch.shape == (3, 2)  # 3 samples, 2 features
        
        # Check y batch
        y_batch = collated[1]
        assert isinstance(y_batch, torch.Tensor)
        assert y_batch.shape == (3, 1)  # 3 samples, 1 target

    def test_collate_with_padding(self):
        """Test collating with padding"""
        collate_fn = FractionalCollateFunction(pad_length=5)
        
        batch = [
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0])),
            (torch.tensor([4.0]), torch.tensor([5.0])),  # Shorter
            (torch.tensor([6.0, 7.0, 8.0]), torch.tensor([9.0]))  # Longer
        ]
        
        collated = collate_fn(batch)
        
        # Check X batch (should be padded to length 5)
        x_batch = collated[0]
        assert x_batch.shape == (3, 5)  # 3 samples, padded to 5 features
        
        # Check that padding was applied correctly
        assert torch.equal(x_batch[0], torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0]))
        assert torch.equal(x_batch[1], torch.tensor([4.0, 0.0, 0.0, 0.0, 0.0]))
        assert torch.equal(x_batch[2], torch.tensor([6.0, 7.0, 8.0, 0.0, 0.0]))

    def test_collate_empty_batch(self):
        """Test collating empty batch"""
        collate_fn = FractionalCollateFunction()
        
        batch = []
        
        with pytest.raises(ValueError, match="Batch cannot be empty"):
            collate_fn(batch)

    def test_collate_single_sample(self):
        """Test collating single sample"""
        collate_fn = FractionalCollateFunction()
        
        batch = [(torch.tensor([1.0, 2.0]), torch.tensor([3.0]))]
        
        collated = collate_fn(batch)
        
        assert isinstance(collated, tuple)
        assert len(collated) == 2
        
        x_batch = collated[0]
        y_batch = collated[1]
        
        assert x_batch.shape == (1, 2)
        assert y_batch.shape == (1, 1)


class TestFractionalDataModule:
    """Test the FractionalDataModule class"""

    def test_initialization_default(self):
        """Test data module initialization with default parameters"""
        datamodule = FractionalDataModule()
        
        assert datamodule.batch_size == 32
        assert datamodule.num_workers == 0
        assert datamodule.pin_memory == False
        assert datamodule.shuffle_train == True
        assert datamodule.shuffle_val == False
        assert datamodule.fractional_order.alpha == 0.5
        assert datamodule.method == "RL"
        assert datamodule.backend == BackendType.TORCH

    def test_initialization_custom(self):
        """Test data module initialization with custom parameters"""
        datamodule = FractionalDataModule(
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            shuffle_train=False,
            shuffle_val=True,
            fractional_order=0.7,
            method="Caputo",
            backend=BackendType.NUMBA
        )
        
        assert datamodule.batch_size == 64
        assert datamodule.num_workers == 4
        assert datamodule.pin_memory == True
        assert datamodule.shuffle_train == False
        assert datamodule.shuffle_val == True
        assert datamodule.fractional_order.alpha == 0.7
        assert datamodule.method == "Caputo"
        assert datamodule.backend == BackendType.NUMBA

    def test_setup_datasets(self):
        """Test setting up datasets"""
        datamodule = FractionalDataModule()
        
        # Create dummy data
        x_train = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y_train = torch.tensor([[3.0], [7.0]])
        x_val = torch.tensor([[5.0, 6.0]])
        y_val = torch.tensor([[11.0]])
        
        datamodule.setup_datasets(x_train, y_train, x_val, y_val)
        
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        assert len(datamodule.train_dataset) == 2
        assert len(datamodule.val_dataset) == 1

    def test_train_dataloader(self):
        """Test training dataloader creation"""
        datamodule = FractionalDataModule(batch_size=2)
        
        # Setup datasets
        x_train = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_train = torch.tensor([[3.0], [7.0], [11.0]])
        x_val = torch.tensor([[7.0, 8.0]])
        y_val = torch.tensor([[15.0]])
        
        datamodule.setup_datasets(x_train, y_train, x_val, y_val)
        
        train_loader = datamodule.train_dataloader()
        
        assert isinstance(train_loader, FractionalDataLoader)
        assert train_loader.batch_size == 2
        assert train_loader.shuffle == True
        
        # Test iteration
        batches = list(train_loader)
        assert len(batches) == 2  # 3 samples with batch_size=2

    def test_val_dataloader(self):
        """Test validation dataloader creation"""
        datamodule = FractionalDataModule(batch_size=2)
        
        # Setup datasets
        x_train = torch.tensor([[1.0, 2.0]])
        y_train = torch.tensor([[3.0]])
        x_val = torch.tensor([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
        y_val = torch.tensor([[9.0], [13.0], [17.0]])
        
        datamodule.setup_datasets(x_train, y_train, x_val, y_val)
        
        val_loader = datamodule.val_dataloader()
        
        assert isinstance(val_loader, FractionalDataLoader)
        assert val_loader.batch_size == 2
        assert val_loader.shuffle == False
        
        # Test iteration
        batches = list(val_loader)
        assert len(batches) == 2  # 3 samples with batch_size=2

    def test_test_dataloader(self):
        """Test test dataloader creation"""
        datamodule = FractionalDataModule(batch_size=2)
        
        # Setup datasets
        x_train = torch.tensor([[1.0, 2.0]])
        y_train = torch.tensor([[3.0]])
        x_val = torch.tensor([[4.0, 5.0]])
        y_val = torch.tensor([[9.0]])
        
        datamodule.setup_datasets(x_train, y_train, x_val, y_val)
        
        test_loader = datamodule.test_dataloader()
        
        assert isinstance(test_loader, FractionalDataLoader)
        assert test_loader.batch_size == 2
        assert test_loader.shuffle == False

    def test_setup_without_datasets(self):
        """Test setup without calling setup_datasets"""
        datamodule = FractionalDataModule()
        
        with pytest.raises(RuntimeError, match="Datasets not set up"):
            datamodule.train_dataloader()


class TestCreateFractionalDataset:
    """Test the create_fractional_dataset function"""

    def test_create_tensor_dataset(self):
        """Test creating FractionalTensorDataset"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        
        dataset = create_fractional_dataset(
            dataset_type='tensor',
            data=tensors,
            fractional_order=0.7
        )
        
        assert isinstance(dataset, FractionalTensorDataset)
        assert dataset.fractional_order.alpha == 0.7
        assert len(dataset) == 3

    def test_create_custom_dataset(self):
        """Test creating custom dataset"""
        class CustomDataset(FractionalDataset):
            def __init__(self, data, **kwargs):
                super().__init__(**kwargs)
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, index):
                return self.data[index]
        
        data = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        
        dataset = create_fractional_dataset(
            dataset_type='custom',
            data=data,
            dataset_class=CustomDataset,
            fractional_order=0.6
        )
        
        assert isinstance(dataset, CustomDataset)
        assert dataset.fractional_order.alpha == 0.6
        assert len(dataset) == 3

    def test_invalid_dataset_type(self):
        """Test creating dataset with invalid type"""
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            create_fractional_dataset(dataset_type='invalid')


class TestCreateFractionalDataLoader:
    """Test the create_fractional_dataloader function"""

    def test_create_dataloader_default(self):
        """Test creating dataloader with default parameters"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        dataset = FractionalTensorDataset(tensors)
        
        dataloader = create_fractional_dataloader(dataset)
        
        assert isinstance(dataloader, FractionalDataLoader)
        assert dataloader.dataset == dataset
        assert dataloader.batch_size == 1

    def test_create_dataloader_custom(self):
        """Test creating dataloader with custom parameters"""
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        dataset = FractionalTensorDataset(tensors)
        
        dataloader = create_fractional_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=2
        )
        
        assert isinstance(dataloader, FractionalDataLoader)
        assert dataloader.batch_size == 2
        assert dataloader.shuffle == True
        assert dataloader.num_workers == 2


class TestCreateFractionalDataModule:
    """Test the create_fractional_datamodule function"""

    def test_create_datamodule_default(self):
        """Test creating datamodule with default parameters"""
        datamodule = create_fractional_datamodule()
        
        assert isinstance(datamodule, FractionalDataModule)
        assert datamodule.batch_size == 32

    def test_create_datamodule_custom(self):
        """Test creating datamodule with custom parameters"""
        datamodule = create_fractional_datamodule(
            batch_size=64,
            num_workers=4,
            fractional_order=0.8
        )
        
        assert isinstance(datamodule, FractionalDataModule)
        assert datamodule.batch_size == 64
        assert datamodule.num_workers == 4
        assert datamodule.fractional_order.alpha == 0.8


# Integration tests
class TestDataIntegration:
    """Integration tests for data module"""

    def test_full_data_pipeline(self):
        """Test complete data loading pipeline"""
        # Create data
        x_train = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_train = torch.tensor([[3.0], [7.0], [11.0]])
        x_val = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
        y_val = torch.tensor([[15.0], [19.0]])
        
        # Create datasets
        train_dataset = FractionalTensorDataset([x_train, y_train])
        val_dataset = FractionalTensorDataset([x_val, y_val])
        
        # Create dataloaders
        train_loader = FractionalDataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = FractionalDataLoader(val_dataset, batch_size=2, shuffle=False)
        
        # Test training loader
        train_batches = list(train_loader)
        assert len(train_batches) == 2  # 3 samples with batch_size=2
        
        # Test validation loader
        val_batches = list(val_loader)
        assert len(val_batches) == 1  # 2 samples with batch_size=2
        
        # Verify batch structure
        for batch in train_batches + val_batches:
            assert isinstance(batch, tuple)
            assert len(batch) == 2
            assert isinstance(batch[0], torch.Tensor)
            assert isinstance(batch[1], torch.Tensor)

    def test_datamodule_full_workflow(self):
        """Test complete datamodule workflow"""
        # Create data
        x_train = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_train = torch.tensor([[3.0], [7.0], [11.0]])
        x_val = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
        y_val = torch.tensor([[15.0], [19.0]])
        
        # Create datamodule
        datamodule = FractionalDataModule(batch_size=2)
        datamodule.setup_datasets(x_train, y_train, x_val, y_val)
        
        # Get dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()
        
        # Test all loaders
        assert len(list(train_loader)) == 2
        assert len(list(val_loader)) == 1
        assert len(list(test_loader)) == 1

    def test_fractional_transform_integration(self):
        """Test fractional transform integration"""
        # Create dataset with fractional transform enabled
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0])
        ]
        dataset = FractionalTensorDataset(
            tensors, 
            apply_fractional=True,
            fractional_order=0.7
        )
        
        # Create dataloader
        dataloader = FractionalDataLoader(dataset, batch_size=2)
        
        # Test that fractional transform is applied
        batch = next(iter(dataloader))
        assert isinstance(batch[0], torch.Tensor)
        assert isinstance(batch[1], torch.Tensor)
        assert batch[0].shape == (2, 3)
        assert batch[1].shape == (2, 3)

    def test_collate_function_integration(self):
        """Test collate function integration"""
        # Create dataset
        tensors = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0])
        ]
        dataset = FractionalTensorDataset(tensors)
        
        # Create custom collate function
        collate_fn = FractionalCollateFunction(pad_length=5)
        
        # Create dataloader with custom collate function
        dataloader = FractionalDataLoader(
            dataset, 
            batch_size=2,
            collate_fn=collate_fn
        )
        
        # Test collation
        batch = next(iter(dataloader))
        assert batch[0].shape == (2, 5)  # Padded to length 5
        assert batch[1].shape == (2, 5)  # Padded to length 5
