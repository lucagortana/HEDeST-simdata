"""Regenerates every simulated dataset HEDeST is validated on, from the raw downloads.

Three families, one command::

    python make_simdata.py /path/to/data

See the repository README for what gets downloaded, what gets built and what it costs.
"""
__all__ = ["config", "manifest", "pipeline", "hedest_bridge"]
