from __future__ import annotations

import sys
import logging
from typing import (
    Any,
    Optional,
    Tuple,
    Union,
    List,
    Dict,
    Final,
    TypeAlias,
    Generator,
    Iterator,
    Sequence,
    no_type_check,
)
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
from enum import Enum, auto
from contextlib import contextmanager
from abc import ABC, abstractmethod
import math

import numpy as np
from numpy.typing import NDArray

# ============================================================================
# 1. EXCEPTION HIERARCHY (Standard 5)
# ============================================================================

class NumPyTutorialError(Exception):
    """Base exception for all tutorial-related errors."""
    pass


class ValidationError(NumPyTutorialError):
    """Raised when input validation fails."""
    pass


class ConfigurationError(NumPyTutorialError):
    """Raised when configuration is invalid."""
    pass


class ResourceError(NumPyTutorialError):
    """Raised when resource management fails."""
    pass


# ============================================================================
# 2. IMMUTABLE CONFIGURATION (Standard 6)
# ============================================================================

@dataclass(frozen=True)
class TutorialConfig:
    """Immutable configuration for the NumPy tutorial.
    
    Attributes:
        seed: Deterministic seed for random operations
        float_precision: Floating point precision for comparisons
        log_level: Structured logging level
        output_dir: Directory for generated artifacts
        array_shapes: Standard array shapes for examples
    """
    seed: int = 42
    float_precision: float = 1e-12
    log_level: int = logging.INFO
    output_dir: Path = Path("./tutorial_output")
    array_shapes: Tuple[Tuple[int, ...], ...] = (
        (5,),           # 1D vector
        (2, 3),         # 2D matrix
        (2, 5, 3),      # 3D tensor
        (0,),           # Empty vector
        (1,),           # Single element
    )
    
    def __post_init__(self) -> None:
        """Validate configuration upon initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate all configuration parameters."""
        # Standard 3: Comprehensive precondition validation
        assert isinstance(self.seed, int), f"seed must be int, got {type(self.seed)}"
        assert -2**63 <= self.seed <= 2**63 - 1, f"seed out of 64-bit range: {self.seed}"
        
        assert isinstance(self.float_precision, float), \
            f"float_precision must be float, got {type(self.float_precision)}"
        assert self.float_precision > 0, f"float_precision must be positive: {self.float_precision}"
        assert self.float_precision < 1.0, f"float_precision must be < 1.0: {self.float_precision}"
        
        assert isinstance(self.log_level, int), f"log_level must be int, got {type(self.log_level)}"
        assert self.log_level in {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL}, \
            f"Invalid log_level: {self.log_level}"
        
        assert isinstance(self.output_dir, Path), \
            f"output_dir must be Path, got {type(self.output_dir)}"
        
        assert isinstance(self.array_shapes, tuple), \
            f"array_shapes must be tuple, got {type(self.array_shapes)}"
        
        for shape in self.array_shapes:
            assert isinstance(shape, tuple), f"Each shape must be tuple, got {type(shape)}"
            for dim in shape:
                assert isinstance(dim, int), f"Dimension must be int, got {type(dim)}"
                assert dim >= 0, f"Dimension must be non-negative: {dim}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        
        Raises:
            ConfigurationError: If serialization fails
        """
        try:
            config_dict = asdict(self)
            config_dict["output_dir"] = str(config_dict["output_dir"])
            config_dict["array_shapes"] = list(config_dict["array_shapes"])
            return config_dict
        except Exception as e:
            raise ConfigurationError(f"Failed to serialize config: {e}") from e


# ============================================================================
# 3. STRUCTURED LOGGING (Standard 7)
# ============================================================================

class StructuredLogger:
    """Structured logger for tutorial events."""
    
    def __init__(self, config: TutorialConfig) -> None:
        """Initialize structured logger.
        
        Args:
            config: Tutorial configuration
        
        Raises:
            ResourceError: If logger setup fails
        """
        self.config = config
        self._logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure and return structured logger.
        
        Returns:
            Configured logger instance
        
        Raises:
            ResourceError: If log directory creation fails
        """
        try:
            # Create output directory if it doesn't exist
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Configure logging format
            log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            date_format = "%Y-%m-%dT%H:%M:%SZ"
            
            # Create logger
            logger = logging.getLogger("NumPyTutorial")
            logger.setLevel(self.config.log_level)
            
            # Remove existing handlers
            logger.handlers.clear()
            
            # Add file handler
            log_file = self.config.output_dir / "tutorial.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter(log_format, datefmt=date_format)
            )
            logger.addHandler(file_handler)
            
            # Add console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter(log_format, datefmt=date_format)
            )
            logger.addHandler(console_handler)
            
            return logger
            
        except (OSError, PermissionError) as e:
            raise ResourceError(f"Failed to setup logging: {e}") from e
    
    def log_event(
        self,
        level: int,
        event: str,
        array_shape: Optional[Tuple[int, ...]] = None,
        array_size: Optional[int] = None,
        **context: Any,
    ) -> None:
        """Log structured event.
        
        Args:
            level: Logging level
            event: Event identifier
            array_shape: Optional array shape for context
            array_size: Optional array size for context
            **context: Additional context fields
        
        Raises:
            ValidationError: If event is empty
        """
        # Standard 3: Input validation
        if not event or not isinstance(event, str):
            raise ValidationError(f"Event must be non-empty string, got: {event}")
        
        # Build structured context
        structured_context = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),  # Standard 11
            "config_seed": self.config.seed,
            **context,
        }
        
        if array_shape is not None:
            structured_context["array_shape"] = str(array_shape)
            structured_context["array_ndim"] = len(array_shape)
        
        if array_size is not None:
            structured_context["array_size"] = array_size
        
        # Log with appropriate level
        message = f"{event} | {json.dumps(structured_context, default=str)}"
        self._logger.log(level, message)


# ============================================================================
# 4. PURE DATA TRANSFORMATION FUNCTIONS (Standard 8)
# ============================================================================

def create_example_array(shape: Tuple[int, ...], seed: int = 42) -> NDArray[np.float64]:
    """Create deterministic example array with given shape.
    
    Standard 2: Deterministic execution via explicit seed
    Standard 8: Pure function with no side effects
    
    Args:
        shape: Desired array shape
        seed: Random seed for reproducibility
    
    Returns:
        Deterministic NumPy array
    
    Raises:
        ValidationError: If shape is invalid
        ValueError: If array creation fails
    
    Complexity: O(∏shape) time, O(∏shape) space (Standard 14)
    """
    # Standard 3: Comprehensive precondition validation
    if not isinstance(shape, tuple):
        raise ValidationError(f"shape must be tuple, got {type(shape)}")
    
    for dim in shape:
        if not isinstance(dim, int):
            raise ValidationError(f"Dimension must be int, got {type(dim)}")
        if dim < 0:
            raise ValidationError(f"Dimension must be non-negative: {dim}")
    
    # Standard 2: Set explicit random seed
    rng = np.random.default_rng(seed)
    
    try:
        # Create array with deterministic values
        size = np.prod(shape, dtype=np.int64)
        if size == 0:
            return np.array([], dtype=np.float64).reshape(shape)
        
        # Generate values and reshape
        values = rng.random(size=int(size), dtype=np.float64) * 100
        array = values.reshape(shape)
        
        return array
        
    except (MemoryError, ValueError) as e:
        raise ValueError(f"Failed to create array with shape {shape}: {e}") from e


def compute_array_properties(array: NDArray[Any]) -> Dict[str, Any]:
    """Compute all properties of a NumPy array.
    
    Standard 8: Pure function with no side effects
    
    Args:
        array: Input NumPy array
    
    Returns:
        Dictionary containing array properties
    
    Raises:
        ValidationError: If input is not a NumPy array
    
    Complexity: O(1) time, O(1) space (Standard 14)
    """
    # Standard 3: Input validation
    if not isinstance(array, np.ndarray):
        raise ValidationError(f"array must be np.ndarray, got {type(array)}")
    
    return {
        "size": int(array.size),
        "shape": array.shape,
        "ndim": array.ndim,
        "dtype": str(array.dtype),
        "itemsize": array.itemsize,
        "nbytes": array.nbytes,
        "len": len(array),
    }


def verify_size_shape_relationship(array: NDArray[Any]) -> bool:
    """Verify that size = product of shape dimensions.
    
    Standard 10: Deterministic numerical precision control
    
    Args:
        array: Input NumPy array
    
    Returns:
        True if relationship holds, False otherwise
    
    Raises:
        ValidationError: If input is invalid
    """
    properties = compute_array_properties(array)
    
    # Compute product of shape dimensions
    shape_product = 1
    for dim in properties["shape"]:
        shape_product *= dim
    
    # Standard 10: Explicit tolerance for integer comparison
    return bool(properties["size"] == shape_product)


# ============================================================================
# 5. RESOURCE MANAGEMENT (Standard 4)
# ============================================================================

@contextmanager
def managed_array_generation(
    config: TutorialConfig,
) -> Generator[Dict[str, NDArray[np.float64]], None, None]:
    """Context manager for deterministic array generation.
    
    Standard 4: Explicit resource lifecycle management
    
    Args:
        config: Tutorial configuration
    
    Yields:
        Dictionary of example arrays
    
    Raises:
        ResourceError: If array generation fails
    """
    arrays: Dict[str, NDArray[np.float64]] = {}
    
    try:
        # Standard 2: Set global random state
        np.random.seed(config.seed)
        
        # Generate all example arrays
        for i, shape in enumerate(config.array_shapes):
            try:
                array_name = f"array_{i}_{'x'.join(map(str, shape))}"
                arrays[array_name] = create_example_array(shape, config.seed)
            except Exception as e:
                raise ResourceError(f"Failed to create array with shape {shape}: {e}") from e
        
        yield arrays
        
    finally:
        # Cleanup: Explicitly delete large arrays
        for array_name in list(arrays.keys()):
            del arrays[array_name]
        arrays.clear()


# ============================================================================
# 6. MAIN TUTORIAL CLASS (Standards 9, 12, 15, 19)
# ============================================================================

class NumPyArrayTutorial:
    """Tutorial for NumPy array attributes with full standards compliance.
    
    Standard 9: Complete public API documentation
    Standard 12: Unit test isolation via dependency injection
    Standard 15: Cyclomatic complexity control
    Standard 19: No implicit global state
    
    Attributes:
        config: Immutable tutorial configuration
        logger: Structured logger for events
        examples: Generated example arrays
    """
    
    def __init__(
        self,
        config: Optional[TutorialConfig] = None,
        logger: Optional[StructuredLogger] = None,
    ) -> None:
        """Initialize tutorial with configuration and dependencies.
        
        Args:
            config: Tutorial configuration (creates default if None)
            logger: Structured logger (creates default if None)
        
        Raises:
            ConfigurationError: If initialization fails
        """
        self.config = config or TutorialConfig()
        self.logger = logger or StructuredLogger(self.config)
        self.examples: Dict[str, NDArray[np.float64]] = {}
        
        self.logger.log_event(
            logging.INFO,
            "tutorial_initialized",
            config_seed=self.config.seed,
        )
    
    def demonstrate_size_attribute(self) -> None:
        """Demonstrate .size attribute (total element count).
        
        Complexity: O(n) where n is number of examples (Standard 14)
        """
        self.logger.log_event(logging.INFO, "demonstrate_size_start")
        
        print("\n" + "=" * 60)
        print("PART 1: .size Attribute (Total Element Count)")
        print("=" * 60)
        
        for name, array in self.examples.items():
            props = compute_array_properties(array)
            
            print(f"\n{name}: shape={array.shape}")
            print(f"  array.size = {props['size']:,}")
            print(f"  Interpretation: {props['size']:,} total elements")
            
            self.logger.log_event(
                logging.DEBUG,
                "size_demonstration",
                array_shape=array.shape,
                array_size=props["size"],
                array_name=name,
            )
        
        self.logger.log_event(logging.INFO, "demonstrate_size_complete")
    
    def demonstrate_shape_attribute(self) -> None:
        """Demonstrate .shape attribute (dimensional structure).
        
        Complexity: O(n) where n is number of examples (Standard 14)
        """
        self.logger.log_event(logging.INFO, "demonstrate_shape_start")
        
        print("\n" + "=" * 60)
        print("PART 2: .shape Attribute (Dimensional Structure)")
        print("=" * 60)
        
        for name, array in self.examples.items():
            props = compute_array_properties(array)
            
            print(f"\n{name}:")
            print(f"  array.shape = {array.shape}")
            print(f"  Type: {type(array.shape).__name__}")
            print(f"  Dimensionality: {props['ndim']}D array")
            
            if props['ndim'] == 1:
                print(f"  Interpretation: Vector with {array.shape[0]} elements")
            elif props['ndim'] == 2:
                rows, cols = array.shape
                print(f"  Interpretation: Matrix with {rows} rows, {cols} columns")
            else:
                print(f"  Interpretation: {props['ndim']}D tensor")
            
            self.logger.log_event(
                logging.DEBUG,
                "shape_demonstration",
                array_shape=array.shape,
                array_ndim=props['ndim'],
                array_name=name,
            )
        
        self.logger.log_event(logging.INFO, "demonstrate_shape_complete")
    
    def demonstrate_len_function(self) -> None:
        """Demonstrate len() vs NumPy attributes.
        
        Complexity: O(n) where n is number of examples (Standard 14)
        """
        self.logger.log_event(logging.INFO, "demonstrate_len_start")
        
        print("\n" + "=" * 60)
        print("PART 3: len() vs NumPy Attributes")
        print("=" * 60)
        
        print("\nKey Insight: len(array) returns first dimension only!")
        print("-" * 40)
        
        for name, array in self.examples.items():
            props = compute_array_properties(array)
            
            print(f"\n{name}: shape={array.shape}")
            print(f"  len(array) = {props['len']}")
            print(f"  array.shape[0] = {array.shape[0]}")
            print(f"  array.size = {props['size']:,}")
            
            if props['ndim'] > 1:
                print(f"  ⚠️  Warning: len() ≠ size for {props['ndim']}D arrays!")
            
            self.logger.log_event(
                logging.DEBUG,
                "len_demonstration",
                array_shape=array.shape,
                array_len=props['len'],
                array_size=props['size'],
                array_name=name,
            )
        
        self.logger.log_event(logging.INFO, "demonstrate_len_complete")
    
    def demonstrate_relationships(self) -> None:
        """Demonstrate mathematical relationships between attributes.
        
        Complexity: O(n) where n is number of examples (Standard 14)
        """
        self.logger.log_event(logging.INFO, "demonstrate_relationships_start")
        
        print("\n" + "=" * 60)
        print("PART 4: Mathematical Relationships")
        print("=" * 60)
        
        print("\nFundamental Identity: size = ∏(shape)")
        print("-" * 40)
        
        all_verified = True
        
        for name, array in self.examples.items():
            props = compute_array_properties(array)
            verified = verify_size_shape_relationship(array)
            
            print(f"\n{name}: shape={array.shape}")
            print(f"  array.size = {props['size']:,}")
            
            # Compute product explicitly
            product = 1
            product_str_parts = []
            for dim in array.shape:
                product *= dim
                product_str_parts.append(str(dim))
            
            product_str = " × ".join(product_str_parts)
            print(f"  ∏(shape) = {product_str} = {product:,}")
            print(f"  Verification: {props['size']:,} == {product:,} ? {verified}")
            
            if not verified:
                all_verified = False
                self.logger.log_event(
                    logging.ERROR,
                    "relationship_verification_failed",
                    array_shape=array.shape,
                    computed_product=product,
                    actual_size=props['size'],
                    array_name=name,
                )
        
        if all_verified:
            print(f"\n✅ All arrays satisfy size = ∏(shape)")
            self.logger.log_event(logging.INFO, "all_relationships_verified")
        else:
            print(f"\n❌ Some arrays failed verification!")
            self.logger.log_event(logging.ERROR, "some_relationships_failed")
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive tutorial summary.
        
        Returns:
            Dictionary with tutorial summary
        
        Complexity: O(n) where n is number of examples (Standard 14)
        """
        self.logger.log_event(logging.INFO, "generate_summary_start")
        
        summary: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "examples": {},
            "statistics": {
                "total_examples": 0,
                "total_elements": 0,
                "max_dimensions": 0,
                "shapes_covered": [],
            },
        }
        
        total_elements = 0
        max_dimensions = 0
        
        for name, array in self.examples.items():
            props = compute_array_properties(array)
            
            summary["examples"][name] = {
                "shape": array.shape,
                "size": props["size"],
                "ndim": props["ndim"],
                "dtype": props["dtype"],
                "len": props["len"],
                "nbytes": props["nbytes"],
            }
            
            total_elements += props["size"]
            max_dimensions = max(max_dimensions, props["ndim"])
        
        summary["statistics"].update({
            "total_examples": len(self.examples),
            "total_elements": total_elements,
            "max_dimensions": max_dimensions,
            "shapes_covered": [str(array.shape) for array in self.examples.values()],
        })
        
        self.logger.log_event(
            logging.INFO,
            "generate_summary_complete",
            total_examples=len(self.examples),
            total_elements=total_elements,
        )
        
        return summary
    
    def run_tutorial(self) -> Dict[str, Any]:
        """Execute complete tutorial with all demonstrations.
        
        Returns:
            Tutorial summary
        
        Raises:
            ResourceError: If tutorial execution fails
        
        Complexity: O(n) where n is total elements across examples (Standard 14)
        """
        self.logger.log_event(logging.INFO, "tutorial_start")
        
        try:
            # Standard 4: Resource management
            with managed_array_generation(self.config) as arrays:
                self.examples = arrays
                
                # Execute all demonstration sections
                self.demonstrate_size_attribute()
                self.demonstrate_shape_attribute()
                self.demonstrate_len_function()
                self.demonstrate_relationships()
                
                # Generate and display summary
                summary = self.generate_summary()
                
                print("\n" + "=" * 60)
                print("SUMMARY")
                print("=" * 60)
                
                print(f"\nGenerated {summary['statistics']['total_examples']} examples")
                print(f"Total elements: {summary['statistics']['total_elements']:,}")
                print(f"Maximum dimensions: {summary['statistics']['max_dimensions']}D")
                
                # Save summary to file
                output_file = self.config.output_dir / "tutorial_summary.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, default=str)
                
                print(f"\nSummary saved to: {output_file}")
                
                self.logger.log_event(
                    logging.INFO,
                    "tutorial_complete",
                    output_file=str(output_file),
                    **summary["statistics"],
                )
                
                return summary
                
        except Exception as e:
            self.logger.log_event(
                logging.ERROR,
                "tutorial_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise ResourceError(f"Tutorial execution failed: {e}") from e


# ============================================================================
# 7. MAIN EXECUTION (Standards 1, 2, 3, 17, 20)
# ============================================================================

def main() -> int:
    """Main entry point for NumPy tutorial.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    
    Standards Compliance:
        1. Complete type annotations
        2. Deterministic execution
        3. Precondition validation
        17. Explicit dependencies (enforced via pyproject.toml)
        20. Linter conformance (enforced via pre-commit)
    """
    try:
        # Standard 2: Set global deterministic state
        np.random.seed(42)
        
        # Standard 6: Immutable configuration
        config = TutorialConfig(
            seed=42,
            float_precision=1e-12,
            log_level=logging.INFO,
            output_dir=Path("./tutorial_output"),
        )
        
        # Initialize and run tutorial
        tutorial = NumPyArrayTutorial(config)
        tutorial.run_tutorial()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nTutorial interrupted by user")
        return 130  # Standard SIGINT exit code
    except Exception as e:
        print(f"\n❌ Tutorial failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())