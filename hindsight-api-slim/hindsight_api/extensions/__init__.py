"""
Hindsight Extensions System.

Extensions allow customizing and extending Hindsight behavior without modifying core code.
Extensions are loaded via environment variables pointing to implementation classes.

Example:
    HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION=mypackage.validators:MyValidator
    HINDSIGHT_API_OPERATION_VALIDATOR_MAX_RETRIES=3

    HINDSIGHT_API_HTTP_EXTENSION=mypackage.http:MyHttpExtension
    HINDSIGHT_API_HTTP_SOME_CONFIG=value

Extensions receive an ExtensionContext that provides a controlled API for interacting
with the system (e.g., running migrations for tenant schemas).
"""

from hindsight_api.extensions.base import Extension
from hindsight_api.extensions.builtin import ApiKeyTenantExtension, SupabaseTenantExtension
from hindsight_api.extensions.context import DefaultExtensionContext, ExtensionContext
from hindsight_api.extensions.http import HttpExtension
from hindsight_api.extensions.loader import load_extension
from hindsight_api.extensions.mcp import MCPExtension
from hindsight_api.extensions.operation_validator import (
    # Bank Management operations
    BankListContext,
    BankListResult,
    BankReadContext,
    BankWriteContext,
    # Consolidation operation
    ConsolidateContext,
    ConsolidateResult,
    # File Conversion
    FileConvertResult,
    # Mental Model operations
    MentalModelGetContext,
    MentalModelGetResult,
    MentalModelRefreshContext,
    MentalModelRefreshResult,
    # Core operations
    OperationValidationError,
    OperationValidatorExtension,
    RecallContext,
    RecallResult,
    ReflectContext,
    ReflectResultContext,
    RetainContext,
    RetainResult,
    ValidationResult,
)
from hindsight_api.extensions.tenant import (
    AuthenticationError,
    Tenant,
    TenantContext,
    TenantExtension,
)
from hindsight_api.models import RequestContext
from hindsight_api.worker.exceptions import DeferOperation

__all__ = [
    # Base
    "Extension",
    "load_extension",
    # Context
    "ExtensionContext",
    "DefaultExtensionContext",
    # HTTP Extension
    "HttpExtension",
    # MCP Extension
    "MCPExtension",
    # Operation Validator - Core
    "DeferOperation",
    "OperationValidationError",
    "OperationValidatorExtension",
    "RecallContext",
    "RecallResult",
    "ReflectContext",
    "ReflectResultContext",
    "RetainContext",
    "RetainResult",
    "ValidationResult",
    # Operation Validator - Bank Management
    "BankListContext",
    "BankListResult",
    "BankReadContext",
    "BankWriteContext",
    # Operation Validator - Consolidation
    "ConsolidateContext",
    "ConsolidateResult",
    # Operation Validator - File Conversion
    "FileConvertResult",
    # Operation Validator - Mental Model
    "MentalModelGetContext",
    "MentalModelGetResult",
    "MentalModelRefreshContext",
    "MentalModelRefreshResult",
    # Tenant/Auth
    "ApiKeyTenantExtension",
    "SupabaseTenantExtension",
    "AuthenticationError",
    "RequestContext",
    "Tenant",
    "TenantContext",
    "TenantExtension",
]
