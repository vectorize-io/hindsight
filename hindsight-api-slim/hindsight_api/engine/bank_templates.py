"""Bank templates: a manifest of bank config, mental models and directives, and applying it to a bank.

Shared by the HTTP import/export endpoints and by ``MemoryEngine``, which applies the
server's default template (``HINDSIGHT_API_DEFAULT_BANK_TEMPLATE``) to every new bank.
"""

import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from hindsight_api.config import RETAIN_EXTRACTION_MODES, get_config
from hindsight_api.engine.interface import BankTemplateImportWrite
from hindsight_api.engine.response_models import ConsolidationStrategySpec, FieldWithDefault, MentalModelTrigger
from hindsight_api.engine.retain.entity_labels import LabelGroup, migrate_entity_labels_input
from hindsight_api.extensions.operation_validator import BankWriteOperation
from hindsight_api.models import RequestContext

if TYPE_CHECKING:
    from hindsight_api.engine.memory_engine import MemoryEngine

# Current manifest schema version. Bump when making breaking changes.
BANK_TEMPLATE_CURRENT_VERSION = "1"


class BankTemplateMentalModel(BaseModel):
    """A mental model definition within a bank template manifest."""

    id: str = Field(description="Unique ID for the mental model (alphanumeric lowercase with hyphens)")
    name: str = Field(description="Human-readable name for the mental model")
    source_query: str = Field(description="The query to run to generate content")
    tags: list[str] = FieldWithDefault(list, description="Tags for scoped visibility")
    max_tokens: int = Field(default=2048, ge=256, le=8192, description="Maximum tokens for generated content")
    trigger: MentalModelTrigger = FieldWithDefault(MentalModelTrigger, description="Trigger settings")

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9][a-z0-9-]*$", v):
            raise ValueError(
                f"Mental model id '{v}' must be alphanumeric lowercase with hyphens, starting with a letter or digit."
            )
        return v


class BankTemplateConfig(BaseModel):
    """Bank configuration fields within a template manifest.

    Only includes configurable (per-bank) fields. Credential fields
    (API keys, base URLs) are intentionally excluded for security.
    """

    reflect_mission: str | None = Field(default=None, description="Mission/context for Reflect operations")
    retain_mission: str | None = Field(default=None, description="Steers what gets extracted during retain")
    retain_extraction_mode: str | None = Field(
        default=None,
        description="Fact extraction mode: 'concise' (default), 'verbose', 'custom', 'verbatim', or 'chunks'",
    )
    retain_custom_instructions: str | None = Field(
        default=None, description="Custom extraction prompt (when mode='custom')"
    )
    retain_chunk_size: int | None = Field(default=None, description="Target max characters for each content chunk")
    retain_structured_chunk_size: int | None = Field(
        default=None,
        description=(
            "Max characters for a single JSONL line or conversation turn to keep whole; "
            "defaults to retain_chunk_size when unset"
        ),
    )
    enable_observations: bool | None = Field(default=None, description="Toggle observation consolidation")
    observations_mission: str | None = Field(default=None, description="Controls what gets synthesised")
    enable_text_search: bool | None = Field(
        default=None, description="Toggle the keyword (BM25) arm during recall, leaving pure vector search"
    )
    enable_temporal_retrieval: bool | None = Field(
        default=None, description="Toggle the temporal arm (and its date-aware query analysis) during recall"
    )
    enable_graph_retrieval: bool | None = Field(
        default=None, description="Toggle the entity/link graph arm during recall"
    )
    enable_reranking: bool | None = Field(default=None, description="Toggle cross-encoder reranking during recall")
    reflect_use_decision_model: bool | None = Field(
        default=None,
        description="When the server has a decision model, reflect lets it prune evidence and decide when to stop",
    )
    mental_model_refresh_use_decision_model: bool | None = Field(
        default=None, description="The same, for knowledge-page refreshes"
    )
    mental_model_delta_use_decision_model: bool | None = Field(
        default=None,
        description="When the server has a decision model, a delta knowledge-page refresh asks it which new "
        "memories could change the page instead of running reflect",
    )
    disposition_skepticism: int | None = Field(default=None, ge=1, le=5, description="Skepticism trait (1-5)")
    disposition_literalism: int | None = Field(default=None, ge=1, le=5, description="Literalism trait (1-5)")
    disposition_empathy: int | None = Field(default=None, ge=1, le=5, description="Empathy trait (1-5)")
    entity_labels: list[LabelGroup] | None = Field(default=None, description="Controlled vocabulary for entity labels")
    _migrate_entity_labels = field_validator("entity_labels", mode="before")(migrate_entity_labels_input)
    entities_allow_free_form: bool | None = Field(
        default=None, description="Allow entities outside the label vocabulary"
    )
    retain_default_strategy: str | None = Field(
        default=None, description="Name of the default retain strategy (key into retain_strategies map)"
    )
    retain_strategies: dict | None = Field(
        default=None, description="Map of retain strategy name to per-strategy config dict"
    )
    retain_chunk_batch_size: int | None = Field(
        default=None, description="Max chunks per streaming batch (0 disables batching)"
    )
    retain_max_attachments_per_chunk: int | None = Field(
        default=None, description="Hard cap on inline images in a single extraction chunk"
    )
    mcp_enabled_tools: list[str] | None = Field(
        default=None, description="MCP tool allowlist for this bank (None = all tools)"
    )
    consolidation_llm_batch_size: int | None = Field(
        default=None, description="LLM batch size for observation consolidation"
    )
    consolidation_source_facts_max_tokens: int | None = Field(
        default=None, description="Max tokens of source facts per consolidation batch"
    )
    consolidation_source_facts_max_tokens_per_observation: int | None = Field(
        default=None, description="Max tokens of source facts per observation"
    )
    max_observations_per_scope: int | None = Field(
        default=None, description="Max observations to retain per consolidation scope"
    )
    observation_scope_limits: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "DEPRECATED — use consolidation_strategies, which carries the mission too. "
            "Still honoured, but consulted only after consolidation_strategies. "
            "Per-scope overrides of max_observations_per_scope: "
            '[{"scope": ["run_*", "shared"], "limit": 1}]. Each scope is a list of '
            "fnmatch tag-globs; a consolidation scope matches under exact cover "
            "(every tag matched by a glob and every glob matched by a tag). The first "
            "matching rule wins; unmatched scopes fall back to max_observations_per_scope."
        ),
    )
    default_observation_scopes: (
        Literal["per_tag", "combined", "all_combinations", "shared"] | list[list[str]] | None
    ) = Field(
        default=None,
        description=(
            "observation_scopes applied at retain to items that send none ('combined' when unset). "
            "A list of tag lists may use globs, e.g. [['topic:*']] scopes each memory by its own topic tags."
        ),
    )
    consolidation_strategies: list[ConsolidationStrategySpec] | None = Field(
        default=None,
        description=(
            "Per-scope consolidation settings: "
            '[{"scopes": [{"tags": ["company:*"]}], "observations_mission": "Record only generalized '
            'trends.", "max_observations_per_scope": 20}]. Each strategy lists the rules it claims '
            "scopes with — a rule's tags are fnmatch globs that must all be on the scope, and its "
            '"tags_match" decides whether the scope may carry others ("all", the default) or not '
            '("exact"). The rules are alternatives: any one matching claims the scope. A strategy may '
            "set any of observations_mission, max_observations_per_scope, "
            "consolidation_source_facts_max_tokens and "
            "consolidation_source_facts_max_tokens_per_observation; each is optional. "
            "Exactly one strategy applies to a scope: the first in the list that claims it. "
            "Whatever that strategy leaves unset — and every scope no strategy claims — "
            "uses the bank-wide value; a later strategy never fills the gaps. "
            "Supersedes observation_scope_limits. Lets one bank be federated across "
            "user/team/company tag scopes, each consolidating under its own brief."
        ),
    )
    reflect_source_facts_max_tokens: int | None = Field(
        default=None, description="Max tokens of source facts per reflect call"
    )
    knowledge_page_default_trigger: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Trigger fields merged over the built-in knowledge-page default when a page is created "
            '(e.g. {"refresh_cron": "0 * * * *"}). A trigger sent with the create request still wins.'
        ),
    )
    reflect_default_options: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Default reflect options for this bank "
            '(e.g. {"reflect_search_observations_max_tokens": 3000, "reflect_search_observations_include_entities": false}). '
            "Applied to every reflect in the bank -- API, MCP and mental-model refresh -- whenever "
            "the request (or the model's trigger) leaves the option unset."
        ),
    )
    mental_model_min_refresh_interval_seconds: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Minimum seconds between two automatic refreshes of the same mental model in this "
            "bank. 0 (the default) means no floor. Overridable per model via the trigger's "
            "min_refresh_interval_seconds."
        ),
    )
    llm_gemini_safety_settings: list | None = Field(
        default=None, description="Per-bank Gemini/VertexAI safety filter settings"
    )
    recall_budget_function: str | None = Field(
        default=None, description="Recall budget mapping function: 'fixed' or 'adaptive'"
    )
    recall_budget_fixed_low: int | None = Field(
        default=None, description="Fixed thinking_budget for budget=low (function='fixed')"
    )
    recall_budget_fixed_mid: int | None = Field(
        default=None, description="Fixed thinking_budget for budget=mid (function='fixed')"
    )
    recall_budget_fixed_high: int | None = Field(
        default=None, description="Fixed thinking_budget for budget=high (function='fixed')"
    )
    recall_budget_adaptive_low: float | None = Field(
        default=None, description="Ratio of max_tokens for budget=low (function='adaptive')"
    )
    recall_budget_adaptive_mid: float | None = Field(
        default=None, description="Ratio of max_tokens for budget=mid (function='adaptive')"
    )
    recall_budget_adaptive_high: float | None = Field(
        default=None, description="Ratio of max_tokens for budget=high (function='adaptive')"
    )
    recall_budget_min: int | None = Field(default=None, description="Floor for the adaptive function (after clamping)")
    recall_budget_max: int | None = Field(
        default=None, description="Ceiling for the adaptive function (after clamping)"
    )
    audit_log_enabled: bool | None = Field(
        default=None, description="Enable audit logging for this bank (overrides the server default)"
    )
    store_document_text: bool | None = Field(
        default=None,
        description="Persist raw source text (documents.original_text / chunks.chunk_text). "
        "Set false to keep only derived facts.",
    )
    enable_auto_consolidation: bool | None = Field(
        default=None, description="Automatically consolidate observations after retain"
    )
    consolidation_use_decision_model: bool | None = Field(
        default=None,
        description="When the server has a decision model, consolidation lets it settle restatements, new "
        "facets and chatter before the LLM call",
    )
    consolidation_max_memories_per_round: int | None = Field(
        default=None, description="Max memory units fed into a single consolidation round"
    )
    consolidation_llm_parallelism: int | None = Field(
        default=None, description="Number of consolidation LLM batches processed concurrently"
    )
    recall_include_chunks: bool | None = Field(default=None, description="Include raw chunks in recall results")
    recall_prefer_observations: bool | None = Field(
        default=None, description="Default for recall prefer_observations when the request omits it"
    )
    recall_max_tokens: int | None = Field(default=None, description="Max tokens of results returned by recall")
    recall_chunks_max_tokens: int | None = Field(
        default=None, description="Max tokens of raw chunks returned by recall (when recall_include_chunks is set)"
    )
    memory_defense: dict | None = Field(
        default=None,
        description="Memory Defense policy for this bank (validated against the DefensePolicy schema on write)",
    )

    def get_config_updates(self) -> dict[str, Any]:
        """Return only the fields that were explicitly set (non-None)."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class BankTemplateDirective(BaseModel):
    """A directive definition within a bank template manifest.

    Directives are matched by name on re-import: existing directives
    with the same name are updated, new ones are created.
    """

    name: str = Field(description="Human-readable name for the directive (used as match key on re-import)")
    content: str = Field(description="The directive text to inject into prompts")
    priority: int = Field(default=0, description="Higher priority directives are injected first")
    is_active: bool = Field(default=True, description="Whether this directive is active")
    tags: list[str] = FieldWithDefault(list, description="Tags for filtering")


class BankTemplateManifest(BaseModel):
    """A bank template manifest for import/export.

    Version field enables forward-compatible schema evolution: the API
    auto-upgrades older manifest versions to the current schema on import.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "version": "1",
                "bank": {
                    "reflect_mission": "You are helping a support agent remember customer interactions.",
                    "retain_mission": "Extract customer issues, resolutions, and sentiment.",
                    "disposition_empathy": 5,
                    "enable_observations": True,
                },
                "mental_models": [
                    {
                        "id": "sentiment-overview",
                        "name": "Customer Sentiment Overview",
                        "source_query": "What is the overall sentiment trend?",
                        "trigger": {"refresh_after_consolidation": True},
                    }
                ],
                "directives": [
                    {
                        "name": "Always be empathetic",
                        "content": "Always respond with empathy and understanding.",
                        "priority": 10,
                    }
                ],
            }
        }
    )

    version: str = Field(description="Manifest schema version (currently '1')")
    bank: BankTemplateConfig | None = Field(
        default=None, description="Bank configuration to apply. Omit to leave config unchanged."
    )
    mental_models: list[BankTemplateMentalModel] | None = Field(
        default=None, description="Mental models to create or update (matched by id). Omit to leave unchanged."
    )
    directives: list[BankTemplateDirective] | None = Field(
        default=None, description="Directives to create or update (matched by name). Omit to leave unchanged."
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        try:
            ver = int(v)
        except ValueError:
            raise ValueError(f"version must be a numeric string, got '{v}'")
        if ver < 1:
            raise ValueError("version must be >= 1")
        if ver > int(BANK_TEMPLATE_CURRENT_VERSION):
            raise ValueError(
                f"version '{v}' is not supported by this server "
                f"(max supported: {BANK_TEMPLATE_CURRENT_VERSION}). Please upgrade Hindsight."
            )
        return v

    @field_validator("mental_models")
    @classmethod
    def validate_unique_mental_model_ids(
        cls,
        v: list[BankTemplateMentalModel] | None,
    ) -> list[BankTemplateMentalModel] | None:
        if v is None:
            return v
        ids = [m.id for m in v]
        duplicates = [mid for mid in ids if ids.count(mid) > 1]
        if duplicates:
            raise ValueError(f"Duplicate mental model ids: {sorted(set(duplicates))}")
        return v

    @field_validator("directives")
    @classmethod
    def validate_unique_directive_names(
        cls,
        v: list[BankTemplateDirective] | None,
    ) -> list[BankTemplateDirective] | None:
        if v is None:
            return v
        names = [d.name for d in v]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            raise ValueError(f"Duplicate directive names: {sorted(set(duplicates))}")
        return v


class BankTemplateImportResponse(BaseModel):
    """Response model for the bank template import endpoint."""

    bank_id: str = Field(description="Bank that was imported into")
    config_applied: bool = Field(description="Whether bank config was updated")
    mental_models_created: list[str] = FieldWithDefault(list, description="IDs of newly created mental models")
    mental_models_updated: list[str] = FieldWithDefault(list, description="IDs of updated mental models")
    directives_created: list[str] = FieldWithDefault(list, description="Names of newly created directives")
    directives_updated: list[str] = FieldWithDefault(list, description="Names of updated directives")
    operation_ids: list[str] = FieldWithDefault(
        list, description="Operation IDs for mental model content generation (async)"
    )
    dry_run: bool = Field(default=False, description="True if this was a validation-only run")


def validate_bank_template(manifest: "BankTemplateManifest") -> list[str]:
    """Validate a parsed manifest beyond Pydantic's structural checks.

    Returns a list of human-readable error strings (e.g. invalid
    extraction mode values, conflicting settings).
    """
    errors: list[str] = []
    if manifest.bank:
        bank = manifest.bank
        if bank.retain_extraction_mode is not None:
            if bank.retain_extraction_mode not in RETAIN_EXTRACTION_MODES:
                errors.append(
                    "bank.retain_extraction_mode: "
                    f"must be one of {RETAIN_EXTRACTION_MODES}, got '{bank.retain_extraction_mode}'"
                )
        if bank.retain_custom_instructions and bank.retain_extraction_mode != "custom":
            errors.append("bank.retain_custom_instructions: requires retain_extraction_mode='custom'")
    if manifest.mental_models:
        seen_mental_model_ids: set[str] = set()
        for i, mm in enumerate(manifest.mental_models):
            if not mm.name.strip():
                errors.append(f"mental_models[{i}].name: must not be empty")
            if not mm.source_query.strip():
                errors.append(f"mental_models[{i}].source_query: must not be empty")
            if mm.id in seen_mental_model_ids:
                errors.append(f"mental_models[{i}].id: duplicate id '{mm.id}'")
            seen_mental_model_ids.add(mm.id)
    if manifest.directives:
        seen_directive_names: set[str] = set()
        for i, d in enumerate(manifest.directives):
            if not d.name.strip():
                errors.append(f"directives[{i}].name: must not be empty")
            if not d.content.strip():
                errors.append(f"directives[{i}].content: must not be empty")
            if d.name in seen_directive_names:
                errors.append(f"directives[{i}].name: duplicate name '{d.name}'")
            seen_directive_names.add(d.name)
    return errors


def load_default_bank_template_manifest() -> "BankTemplateManifest | None":
    """Parse and semantically validate the configured default bank template."""
    template_dict = get_config().default_bank_template
    if not template_dict:
        return None
    manifest = BankTemplateManifest.model_validate(template_dict)
    semantic_errors = validate_bank_template(manifest)
    if semantic_errors:
        raise ValueError("; ".join(semantic_errors))
    return manifest


async def apply_bank_template_manifest(
    memory: "MemoryEngine",
    bank_id: str,
    manifest: "BankTemplateManifest",
    request_context: "RequestContext",
) -> "BankTemplateImportResponse":
    """Apply a client-provided BankTemplateManifest to a bank.

    The authorization context creates a missing bank after validating every
    requested operation. The caller remains responsible for manifest validation
    (Pydantic + validate_bank_template). Server-owned defaults use
    ``apply_default_bank_template_resources`` instead, so this function always
    owns persistence of its client-provided config.
    """
    config_updates = manifest.bank.get_config_updates() if manifest.bank else {}
    bank_exists = (
        await memory.get_bank_profile(
            bank_id,
            request_context=request_context,
        )
        is not None
    )

    # A missing bank receives the server-owned default template during
    # provisioning. Project those resources into the authorization decision so
    # the client's import is authorized as an update when the default owns the
    # same key, while still keeping every client check before bank creation.
    default_manifest: BankTemplateManifest | None = None
    if not bank_exists:
        try:
            default_manifest = load_default_bank_template_manifest()
        except (ValueError, ValidationError):
            # Provisioning owns error logging and the best-effort fallback for a
            # malformed server template. Client authorization must not change it.
            pass
    imported_mental_model_ids = {item.id for item in manifest.mental_models or []}
    imported_directive_names = {item.name for item in manifest.directives or []}
    default_mental_models = (default_manifest.mental_models or []) if default_manifest else []
    default_directives = (default_manifest.directives or []) if default_manifest else []
    projected_mental_model_ids = {item.id for item in default_mental_models} & imported_mental_model_ids
    projected_directive_names = {item.name for item in default_directives} & imported_directive_names

    # limit=None throughout the import path: a create/update decision per imported
    # resource is only correct against the bank's *whole* set. Under the default
    # page size a bank with more than 100 models would look like it lacked the
    # ones past the first page, and the import would create duplicates.
    existing_by_id: dict[str, dict[str, Any]] = {}
    if bank_exists and manifest.mental_models:
        existing = await memory.list_mental_models(
            bank_id=bank_id, limit=None, detail="metadata", request_context=request_context
        )
        existing_by_id = {m["id"]: m for m in existing.items}

    existing_by_name: dict[str, dict[str, Any]] = {}
    if bank_exists and manifest.directives:
        existing_directives = await memory.list_directives(
            bank_id=bank_id, active_only=False, limit=None, request_context=request_context
        )
        existing_by_name = {d["name"]: d for d in existing_directives.items}

    bank_writes: list[BankTemplateImportWrite] = []
    if config_updates:
        bank_writes.append(BankTemplateImportWrite(BankWriteOperation.UPDATE_BANK_CONFIG))
    for mental_model in manifest.mental_models or []:
        if mental_model.id in existing_by_id:
            bank_writes.append(BankTemplateImportWrite(BankWriteOperation.UPDATE_MENTAL_MODEL, mental_model.id))
        elif mental_model.id in projected_mental_model_ids:
            # Default-template application is best-effort. Authorize both
            # outcomes before provisioning so a failed default create can
            # safely fall back to the client's create operation.
            bank_writes.extend(
                [
                    BankTemplateImportWrite(BankWriteOperation.UPDATE_MENTAL_MODEL, mental_model.id),
                    BankTemplateImportWrite(BankWriteOperation.CREATE_MENTAL_MODEL, mental_model.id),
                ]
            )
        else:
            bank_writes.append(BankTemplateImportWrite(BankWriteOperation.CREATE_MENTAL_MODEL, mental_model.id))
    for directive in manifest.directives or []:
        if directive.name in existing_by_name:
            bank_writes.append(BankTemplateImportWrite(BankWriteOperation.UPDATE_DIRECTIVE, directive.name))
        elif directive.name in projected_directive_names:
            bank_writes.extend(
                [
                    BankTemplateImportWrite(BankWriteOperation.UPDATE_DIRECTIVE, directive.name),
                    BankTemplateImportWrite(BankWriteOperation.CREATE_DIRECTIVE, directive.name),
                ]
            )
        else:
            bank_writes.append(BankTemplateImportWrite(BankWriteOperation.CREATE_DIRECTIVE, directive.name))

    async with memory.bank_template_import_authorization(
        bank_id,
        config_updates=config_updates,
        bank_writes=bank_writes,
        mental_model_ids=[mental_model.id for mental_model in manifest.mental_models or []],
        bank_exists=bank_exists,
        request_context=request_context,
    ):
        # The snapshot above only chose which operation to authorize; a concurrent
        # create or delete (and the server default template applied during
        # provisioning) can flip a resource between create and update before this
        # request writes. Re-read the committed state here, inside the scope and
        # immediately before the writes, and decide against that instead.
        if manifest.mental_models:
            provisioned = await memory.list_mental_models(
                bank_id=bank_id,
                limit=None,
                detail="config",
                request_context=request_context,
            )
            existing_by_id = {item["id"]: item for item in provisioned.items}

        if manifest.directives:
            provisioned_directives = await memory.list_directives(
                bank_id=bank_id,
                active_only=False,
                limit=None,
                request_context=request_context,
            )
            existing_by_name = {item["name"]: item for item in provisioned_directives.items}

        # Authorize whatever the fresh state now calls for. Resources whose
        # classification held are already preauthorized and this is a no-op; only a
        # flipped one reaches the validator, so an ordinary import still costs one
        # decision per resource.
        for mental_model in manifest.mental_models or []:
            await memory.authorize_bank_template_import_write(
                bank_id,
                BankWriteOperation.UPDATE_MENTAL_MODEL
                if mental_model.id in existing_by_id
                else BankWriteOperation.CREATE_MENTAL_MODEL,
                target=mental_model.id,
                request_context=request_context,
            )
        for directive in manifest.directives or []:
            await memory.authorize_bank_template_import_write(
                bank_id,
                BankWriteOperation.UPDATE_DIRECTIVE
                if directive.name in existing_by_name
                else BankWriteOperation.CREATE_DIRECTIVE,
                target=directive.name,
                request_context=request_context,
            )

        if config_updates:
            await memory.update_bank_config(bank_id, config_updates, request_context=request_context)

        return await _apply_bank_template_resources(
            memory,
            bank_id,
            manifest,
            existing_by_id,
            existing_by_name,
            request_context,
            config_applied=bool(config_updates),
        )


async def apply_default_bank_template_resources(
    memory: "MemoryEngine",
    bank_id: str,
    manifest: "BankTemplateManifest",
    request_context: "RequestContext",
) -> None:
    """Apply only the resources from a server-owned default template."""
    existing_by_id: dict[str, dict[str, Any]] = {}
    if manifest.mental_models:
        existing = await memory.list_mental_models(
            bank_id=bank_id, limit=None, detail="config", request_context=request_context
        )
        existing_by_id = {model["id"]: model for model in existing.items}

    existing_by_name: dict[str, dict[str, Any]] = {}
    if manifest.directives:
        existing_directives = await memory.list_directives(
            bank_id=bank_id,
            active_only=False,
            limit=None,
            request_context=request_context,
        )
        existing_by_name = {directive["name"]: directive for directive in existing_directives.items}

    await _apply_bank_template_resources(
        memory,
        bank_id,
        manifest,
        existing_by_id,
        existing_by_name,
        request_context,
        config_applied=False,
    )


def _mental_model_matches_template(stored: dict[str, Any], mm: "BankTemplateMentalModel") -> bool:
    """True when re-applying ``mm`` would leave the stored model as it is.

    Mirrors what the update writes: empty manifest tags leave the stored tags
    alone, so they never count as a difference. The stored trigger goes through
    ``MentalModelTrigger`` so its unset fields get the same defaults as the
    manifest's; one that no longer parses counts as changed.
    """
    try:
        stored_trigger = MentalModelTrigger.model_validate(stored["trigger"] or {})
    except ValidationError:
        return False
    return (
        stored["name"] == mm.name
        and stored["source_query"] == mm.source_query
        and stored["max_tokens"] == mm.max_tokens
        and (not mm.tags or set(stored["tags"]) == set(mm.tags))
        and stored_trigger == mm.trigger
    )


def _directive_matches_template(stored: dict[str, Any], directive: "BankTemplateDirective") -> bool:
    """True when re-applying ``directive`` would leave the stored one as it is."""
    return (
        stored["content"] == directive.content
        and stored["priority"] == directive.priority
        and stored["is_active"] == directive.is_active
        and (not directive.tags or set(stored["tags"]) == set(directive.tags))
    )


async def _apply_bank_template_resources(
    memory: "MemoryEngine",
    bank_id: str,
    manifest: "BankTemplateManifest",
    existing_mental_models: dict[str, dict[str, Any]],
    existing_directives: dict[str, dict[str, Any]],
    request_context: "RequestContext",
    *,
    config_applied: bool,
) -> "BankTemplateImportResponse":
    """Apply template resources after the caller has handled config and access."""
    created_ids: list[str] = []
    updated_ids: list[str] = []
    operation_ids: list[str] = []

    if manifest.mental_models:
        for mm in manifest.mental_models:
            if mm.id in existing_mental_models:
                # Unchanged: skip it, or every re-apply of the same manifest would
                # regenerate the model with an LLM call for nothing (#5271).
                if _mental_model_matches_template(existing_mental_models[mm.id], mm):
                    continue
                await memory.update_mental_model(
                    bank_id=bank_id,
                    mental_model_id=mm.id,
                    name=mm.name,
                    source_query=mm.source_query,
                    max_tokens=mm.max_tokens,
                    tags=mm.tags if mm.tags else None,
                    trigger=mm.trigger.model_dump() if mm.trigger else None,
                    request_context=request_context,
                )
                result = await memory.submit_async_refresh_mental_model(
                    bank_id=bank_id,
                    mental_model_id=mm.id,
                    request_context=request_context,
                )
                operation_ids.append(result["operation_id"])
                updated_ids.append(mm.id)
            else:
                mental_model = await memory.create_mental_model(
                    bank_id=bank_id,
                    name=mm.name,
                    source_query=mm.source_query,
                    content="",
                    mental_model_id=mm.id,
                    tags=mm.tags if mm.tags else None,
                    max_tokens=mm.max_tokens,
                    trigger=mm.trigger.model_dump() if mm.trigger else None,
                    request_context=request_context,
                )
                result = await memory.submit_async_refresh_mental_model(
                    bank_id=bank_id,
                    mental_model_id=mental_model["id"],
                    request_context=request_context,
                )
                operation_ids.append(result["operation_id"])
                created_ids.append(mm.id)

    directives_created: list[str] = []
    directives_updated: list[str] = []

    if manifest.directives:
        for directive in manifest.directives:
            if directive.name in existing_directives:
                if _directive_matches_template(existing_directives[directive.name], directive):
                    continue
                await memory.update_directive(
                    bank_id=bank_id,
                    directive_id=existing_directives[directive.name]["id"],
                    name=directive.name,
                    content=directive.content,
                    priority=directive.priority,
                    is_active=directive.is_active,
                    tags=directive.tags if directive.tags else None,
                    request_context=request_context,
                )
                directives_updated.append(directive.name)
            else:
                await memory.create_directive(
                    bank_id=bank_id,
                    name=directive.name,
                    content=directive.content,
                    priority=directive.priority,
                    is_active=directive.is_active,
                    tags=directive.tags if directive.tags else None,
                    request_context=request_context,
                )
                directives_created.append(directive.name)

    return BankTemplateImportResponse(
        bank_id=bank_id,
        config_applied=config_applied,
        mental_models_created=created_ids,
        mental_models_updated=updated_ids,
        directives_created=directives_created,
        directives_updated=directives_updated,
        operation_ids=operation_ids,
        dry_run=False,
    )
