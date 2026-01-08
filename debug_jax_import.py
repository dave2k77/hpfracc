import traceback
import sys

print(f"Python version: {sys.version}")

try:
    import jax
    print(f"JAX version: {jax.__version__}")
except ImportError:
    print("JAX not found")

try:
    import jaxtyping
    print(f"jaxtyping version: {jaxtyping.__version__}")
except ImportError:
    print("jaxtyping not found")
except Exception:
    traceback.print_exc()

try:
    import optax
    print(f"optax version: {optax.__version__}")
except ImportError:
    print("optax not found")
except Exception:
    traceback.print_exc()

print("Attempting to import xla_pmap_p...")
try:
    from jax.interpreters.pxla import xla_pmap_p
    print("Imported xla_pmap_p successfully (unexpectedly!)")
except Exception:
    traceback.print_exc()
