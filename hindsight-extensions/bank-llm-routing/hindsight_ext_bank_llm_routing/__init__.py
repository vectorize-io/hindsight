"""Bank LLM routing extension for the Hindsight API server.

Configure the server to load it with::

    HINDSIGHT_API_LLM_REQUEST_EXTENSION=hindsight_ext_bank_llm_routing:BankLLMRoutingExtension
"""

from hindsight_ext_bank_llm_routing.extension import BankLLMRoutingExtension

__all__ = ["BankLLMRoutingExtension"]
