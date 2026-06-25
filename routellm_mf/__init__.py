"""
routellm-mf: Local-embedding-based LLM prompt difficulty router.

Quick start::

    from routellm_mf import Router, RouterConfig

    router = Router(RouterConfig.from_yaml("config.yaml"))
    response = router.chat("What is 2+2?")
"""

from routellm_mf.config import RouterConfig
from routellm_mf.router import Router

__all__ = ["Router", "RouterConfig"]
__version__ = "0.1.0"
