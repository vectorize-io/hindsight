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
end $$;

create table if not exists organization_members (
  org_id text not null references organizations(id) on delete cascade,
  user_id uuid not null,
  email text,
  role organization_role not null,
  created_at timestamptz not null default now(),
  primary key (org_id, user_id)
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
  role organization_role not null default 'member',
  allowed_operations jsonb,
  expires_at timestamptz,
  revoked_at timestamptz,
  created_at timestamptz not null default now()
);

create table if not exists hindsight_api_key_bank_scopes (
  api_key_id uuid not null references hindsight_api_keys(id) on delete cascade,
  bank_id text not null,
  primary key (api_key_id, bank_id)
);

create index if not exists organization_members_user_id_idx on organization_members(user_id);
create index if not exists organization_invites_org_id_idx on organization_invites(org_id);
create index if not exists hindsight_api_keys_org_id_idx on hindsight_api_keys(org_id);
create index if not exists hindsight_api_key_bank_scopes_bank_id_idx on hindsight_api_key_bank_scopes(bank_id);

alter table organizations enable row level security;
alter table organization_members enable row level security;
alter table organization_invites enable row level security;
alter table hindsight_api_keys enable row level security;
alter table hindsight_api_key_bank_scopes enable row level security;

grant usage on type organization_role to anon, authenticated, service_role;

grant select on organizations to anon, authenticated;
grant select on organization_members to anon, authenticated;
grant select on organization_invites to anon, authenticated;
grant select on hindsight_api_keys to anon, authenticated;
grant select on hindsight_api_key_bank_scopes to anon, authenticated;

grant all on organizations to service_role;
grant all on organization_members to service_role;
grant all on organization_invites to service_role;
grant all on hindsight_api_keys to service_role;
grant all on hindsight_api_key_bank_scopes to service_role;
