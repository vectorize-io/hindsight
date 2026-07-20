create table if not exists organizations (
  id text primary key,
  name text not null,
  config jsonb not null default '{}',
  created_at timestamptz not null default now()
);

do $$
begin
  if not exists (select 1 from pg_type where typname = 'organization_role') then
    create type organization_role as enum ('owner', 'admin', 'member');
  end if;
  if not exists (select 1 from pg_type where typname = 'api_key_permission_mode') then
    create type api_key_permission_mode as enum ('scoped', 'full_access');
  end if;
end $$;

create table if not exists organization_members (
  id uuid primary key default gen_random_uuid(),
  org_id text not null references organizations(id) on delete cascade,
  user_id uuid not null,
  email text,
  role organization_role not null,
  created_at timestamptz not null default now(),
  removed_at timestamptz,
  removed_by_user_id uuid
);

create table if not exists organization_invites (
  id uuid primary key default gen_random_uuid(),
  org_id text not null references organizations(id) on delete cascade,
  email text not null,
  role organization_role not null default 'member',
  token_hash text not null unique,
  expires_at timestamptz not null,
  accepted_at timestamptz,
  revoked_at timestamptz,
  created_by_user_id uuid not null,
  created_at timestamptz not null default now()
);

create table if not exists hindsight_api_keys (
  id uuid primary key default gen_random_uuid(),
  org_id text not null references organizations(id) on delete cascade,
  created_by_user_id uuid,
  name text not null,
  key_hash text not null unique,
  encrypted_key text,
  permission_mode api_key_permission_mode not null default 'scoped',
  allowed_operations jsonb,
  expires_at timestamptz,
  revoked_at timestamptz,
  created_at timestamptz not null default now(),
  constraint hindsight_api_keys_permission_mode_operations_check
    check (
      (permission_mode = 'full_access' and allowed_operations is null)
      or
      (permission_mode = 'scoped' and allowed_operations is not null)
    )
);

create table if not exists hindsight_api_key_operation_scopes (
  api_key_id uuid not null references hindsight_api_keys(id) on delete cascade,
  operation text not null,
  bank_scope_mode text not null default 'all' check (bank_scope_mode in ('all', 'selected')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (api_key_id, operation)
);

create table if not exists hindsight_api_key_operation_bank_scopes (
  api_key_id uuid not null,
  operation text not null,
  bank_id text not null,
  bank_internal_id text not null,
  primary key (api_key_id, operation, bank_internal_id),
  foreign key (api_key_id, operation)
    references hindsight_api_key_operation_scopes(api_key_id, operation)
    on delete cascade
);

create table if not exists hindsight_api_key_created_banks (
  api_key_id uuid not null references hindsight_api_keys(id) on delete cascade,
  bank_id text not null,
  bank_internal_id text not null,
  created_at timestamptz not null default now(),
  deleted_at timestamptz,
  primary key (api_key_id, bank_internal_id)
);

-- Keep the key row and both scope levels in one transaction so a failed
-- restriction cannot leave the prior, broader scope active.
create function replace_hindsight_api_key_permissions(
  p_api_key_id uuid,
  p_org_id text,
  p_permission_mode api_key_permission_mode,
  p_allowed_operations jsonb,
  p_operation_scopes jsonb
)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  if p_permission_mode = 'full_access' and p_allowed_operations is not null then
    raise exception 'Full-access API keys cannot store allowed operations';
  end if;
  if p_permission_mode = 'scoped' and p_allowed_operations is null then
    raise exception 'Scoped API keys require allowed operations';
  end if;

  update hindsight_api_keys
  set
    permission_mode = p_permission_mode,
    allowed_operations = p_allowed_operations
  where id = p_api_key_id
    and org_id = p_org_id
    and revoked_at is null;

  if not found then
    raise exception 'Active API key not found';
  end if;

  delete from hindsight_api_key_operation_bank_scopes
  where api_key_id = p_api_key_id;

  delete from hindsight_api_key_operation_scopes
  where api_key_id = p_api_key_id;

  if p_permission_mode = 'scoped' then
    insert into hindsight_api_key_operation_scopes (
      api_key_id,
      operation,
      bank_scope_mode
    )
    select
      p_api_key_id,
      scope.operation,
      scope.bank_scope_mode
    from jsonb_to_recordset(coalesce(p_operation_scopes, '[]'::jsonb)) as scope(
      operation text,
      bank_scope_mode text,
      bank_scopes jsonb
    )
    where scope.operation <> 'create_bank';

    insert into hindsight_api_key_operation_bank_scopes (
      api_key_id,
      operation,
      bank_id,
      bank_internal_id
    )
    select
      p_api_key_id,
      scope.operation,
      bank.bank_id,
      bank.bank_internal_id
    from jsonb_to_recordset(coalesce(p_operation_scopes, '[]'::jsonb)) as scope(
      operation text,
      bank_scope_mode text,
      bank_scopes jsonb
    )
    cross join lateral jsonb_to_recordset(coalesce(scope.bank_scopes, '[]'::jsonb)) as bank(
      bank_id text,
      bank_internal_id text
    )
    where scope.operation <> 'create_bank'
      and scope.bank_scope_mode = 'selected';
  end if;
end;
$$;

create function create_hindsight_api_key(
  p_org_id text,
  p_created_by_user_id uuid,
  p_name text,
  p_key_hash text,
  p_encrypted_key text,
  p_permission_mode api_key_permission_mode,
  p_allowed_operations jsonb,
  p_operation_scopes jsonb
)
returns table(id uuid)
language plpgsql
security definer
set search_path = public
as $$
declare
  new_api_key_id uuid;
begin
  insert into hindsight_api_keys (
    org_id,
    created_by_user_id,
    name,
    key_hash,
    encrypted_key,
    permission_mode,
    allowed_operations
  )
  values (
    p_org_id,
    p_created_by_user_id,
    p_name,
    p_key_hash,
    p_encrypted_key,
    p_permission_mode,
    p_allowed_operations
  )
  returning hindsight_api_keys.id into new_api_key_id;

  perform replace_hindsight_api_key_permissions(
    new_api_key_id,
    p_org_id,
    p_permission_mode,
    p_allowed_operations,
    p_operation_scopes
  );

  return query select new_api_key_id;
end;
$$;

-- Membership removal and credential revocation are one authorization state
-- transition. A key must never remain active after its creator is removed.
create function remove_organization_member(
  p_org_id text,
  p_user_id uuid,
  p_removed_by_user_id uuid
)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  update hindsight_api_keys
  set revoked_at = coalesce(revoked_at, now())
  where org_id = p_org_id
    and created_by_user_id = p_user_id;

  -- Membership periods are security audit records. End the active period
  -- instead of deleting who belonged to the organization.
  update organization_members
  set
    removed_at = coalesce(removed_at, now()),
    removed_by_user_id = coalesce(removed_by_user_id, p_removed_by_user_id)
  where org_id = p_org_id
    and user_id = p_user_id
    and removed_at is null;
end;
$$;

-- Remove all authorization metadata that points at a bank deleted in the data plane.
create function delete_hindsight_bank_references(p_bank_internal_id text)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  delete from hindsight_api_key_operation_bank_scopes
  where bank_internal_id = p_bank_internal_id;

  -- Creation provenance is an audit fact. Tombstone it for online ownership
  -- checks while retaining who created the deleted bank.
  update hindsight_api_key_created_banks
  set deleted_at = coalesce(deleted_at, now())
  where bank_internal_id = p_bank_internal_id;
end;
$$;

create index if not exists organization_members_user_id_idx on organization_members(user_id);
create unique index if not exists organization_members_active_org_user_idx
  on organization_members(org_id, user_id)
  where removed_at is null;
create index if not exists organization_invites_org_id_idx on organization_invites(org_id);
create index if not exists hindsight_api_keys_org_id_idx on hindsight_api_keys(org_id);
create index if not exists hindsight_api_key_operation_bank_scopes_bank_id_idx on hindsight_api_key_operation_bank_scopes(bank_id);
create index if not exists hindsight_api_key_operation_bank_scopes_bank_internal_id_idx on hindsight_api_key_operation_bank_scopes(bank_internal_id);
create index if not exists hindsight_api_key_created_banks_bank_id_idx on hindsight_api_key_created_banks(bank_id);
create index if not exists hindsight_api_key_created_banks_bank_internal_id_idx on hindsight_api_key_created_banks(bank_internal_id);

alter table organizations enable row level security;
alter table organization_members enable row level security;
alter table organization_invites enable row level security;
alter table hindsight_api_keys enable row level security;
alter table hindsight_api_key_operation_scopes enable row level security;
alter table hindsight_api_key_operation_bank_scopes enable row level security;
alter table hindsight_api_key_created_banks enable row level security;

grant usage on type organization_role, api_key_permission_mode to anon, authenticated, service_role;

grant select on organizations to anon, authenticated;
grant select on organization_members to anon, authenticated;
grant select on organization_invites to anon, authenticated;
grant select on hindsight_api_keys to anon, authenticated;
grant select on hindsight_api_key_operation_scopes to anon, authenticated;
grant select on hindsight_api_key_operation_bank_scopes to anon, authenticated;
grant select on hindsight_api_key_created_banks to anon, authenticated;

grant all on organizations to service_role;
grant all on organization_members to service_role;
grant all on organization_invites to service_role;
grant all on hindsight_api_keys to service_role;
grant all on hindsight_api_key_operation_scopes to service_role;
grant all on hindsight_api_key_operation_bank_scopes to service_role;
grant all on hindsight_api_key_created_banks to service_role;

revoke all on function replace_hindsight_api_key_permissions(
  uuid,
  text,
  api_key_permission_mode,
  jsonb,
  jsonb
) from public;
revoke all on function create_hindsight_api_key(
  text,
  uuid,
  text,
  text,
  text,
  api_key_permission_mode,
  jsonb,
  jsonb
) from public;
revoke all on function remove_organization_member(text, uuid, uuid) from public;
revoke all on function delete_hindsight_bank_references(text) from public;
grant execute on function replace_hindsight_api_key_permissions(
  uuid,
  text,
  api_key_permission_mode,
  jsonb,
  jsonb
) to service_role;
grant execute on function create_hindsight_api_key(
  text,
  uuid,
  text,
  text,
  text,
  api_key_permission_mode,
  jsonb,
  jsonb
) to service_role;
grant execute on function remove_organization_member(text, uuid, uuid) to service_role;
grant execute on function delete_hindsight_bank_references(text) to service_role;
