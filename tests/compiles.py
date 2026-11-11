"""The compiles a block of a test runs, for the compile-once tests."""
import contextlib
import logging

import jax


@contextlib.contextmanager
def compiles(counted=lambda: True):
    """Yields the list of the compiles logged inside the block (each message's first 120 characters), those for which
    ``counted()`` holds when they happen. The persistent compilation cache is off inside the block, so a program it
    holds from an earlier session is compiled, and counted, as in a fresh one."""
    seen = []

    class Catch(logging.Handler):
        def emit(self, record):
            if counted() and record.getMessage().startswith("Compiling"):
                seen.append(record.getMessage()[:120])

    handler, logger = Catch(), logging.getLogger("jax")
    cache = jax.config.jax_enable_compilation_cache
    logger.addHandler(handler)
    jax.config.update("jax_log_compiles", True)
    jax.config.update("jax_enable_compilation_cache", False)
    try:
        yield seen
    finally:
        jax.config.update("jax_enable_compilation_cache", cache)
        jax.config.update("jax_log_compiles", False)
        logger.removeHandler(handler)
