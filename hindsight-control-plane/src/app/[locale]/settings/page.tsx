"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { ArrowLeft, Copy, KeyRound, Plus, RefreshCw, Save, Trash2, UserPlus } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { withBasePath } from "@/lib/base-path";

type Role = "owner" | "admin" | "member";

interface Organization {
  id: string;
  name: string;
  role: Role;
}

interface Member {
  org_id: string;
  user_id: string;
  email?: string;
  role: Role;
}

interface Invite {
  id: string;
  email: string;
  role: Role;
  expires_at: string;
  accepted_at?: string | null;
  revoked_at?: string | null;
}

interface ApiKeySummary {
  id: string;
  name: string;
  allowed_operations?: string[] | null;
  revoked_at?: string | null;
  created_at: string;
}

interface VersionInfo {
  features?: {
    auth_provider?: "disabled" | "access_key" | "supabase_org";
    profile_match?: boolean;
  };
}

export default function SettingsPage() {
  const t = useTranslations("settings");
  const router = useRouter();
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [selectedOrgId, setSelectedOrgId] = useState("");
  const [members, setMembers] = useState<Member[]>([]);
  const [invites, setInvites] = useState<Invite[]>([]);
  const [apiKeys, setApiKeys] = useState<ApiKeySummary[]>([]);
  const [orgName, setOrgName] = useState("");
  const [newOrgName, setNewOrgName] = useState("");
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<Role>("member");
  const [apiKeyName, setApiKeyName] = useState("");
  const [apiKeyBanks, setApiKeyBanks] = useState("");
  const [newInviteLink, setNewInviteLink] = useState<string | null>(null);
  const [newApiKey, setNewApiKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const currentOrg = useMemo(
    () => organizations.find((organization) => organization.id === selectedOrgId),
    [organizations, selectedOrgId]
  );
  const canAdmin = currentOrg?.role === "owner" || currentOrg?.role === "admin";
  const canOwner = currentOrg?.role === "owner";

  async function loadAll() {
    setLoading(true);
    try {
      const version = await fetchJson<VersionInfo>("/api/version");
      if (
        version.features?.auth_provider !== "supabase_org" ||
        version.features?.profile_match === false
      ) {
        router.replace("/dashboard");
        return;
      }

      const me = await fetchJson<{
        organizations: Organization[];
        current: { org_id: string; role: Role } | null;
      }>("/api/me");
      setOrganizations(me.organizations);
      const nextOrgId = me.current?.org_id || me.organizations[0]?.id || "";
      setSelectedOrgId(nextOrgId);
      setOrgName(
        me.organizations.find((organization) => organization.id === nextOrgId)?.name || ""
      );
      const [team, inviteList, keyList] = await Promise.all([
        fetchJson<{ members: Member[] }>("/api/team"),
        fetchJson<{ invites: Invite[] }>("/api/team/invites"),
        fetchJson<{ api_keys: ApiKeySummary[] }>("/api/api-keys"),
      ]);
      setMembers(team.members);
      setInvites(inviteList.invites);
      setApiKeys(keyList.api_keys);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to load settings");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadAll();
  }, []);

  async function createOrg(event: FormEvent) {
    event.preventDefault();
    const response = await fetchJson<{ organization: Organization }>("/api/organizations", {
      method: "POST",
      body: JSON.stringify({ name: newOrgName }),
    });
    setNewOrgName("");
    setOrganizations((items) => [...items, { ...response.organization, role: "owner" }]);
    await selectOrganization(response.organization.id);
    toast.success("Organization created");
  }

  async function selectOrganization(orgId: string) {
    await fetchJson("/api/auth/select-org", {
      method: "POST",
      body: JSON.stringify({ org_id: orgId }),
    });
    setSelectedOrgId(orgId);
    setOrgName(organizations.find((organization) => organization.id === orgId)?.name || "");
    await loadAll();
  }

  async function renameOrganization(event: FormEvent) {
    event.preventDefault();
    if (!selectedOrgId) return;
    const response = await fetchJson<{ organization: Organization }>(
      `/api/organizations/${encodeURIComponent(selectedOrgId)}`,
      {
        method: "PATCH",
        body: JSON.stringify({ name: orgName }),
      }
    );
    setOrganizations((items) =>
      items.map((item) =>
        item.id === response.organization.id ? { ...item, name: response.organization.name } : item
      )
    );
    setOrgName(response.organization.name);
    toast.success("Organization updated");
  }

  async function inviteMember(event: FormEvent) {
    event.preventDefault();
    const response = await fetchJson<{ invite: { invite_url: string } }>("/api/team/invites", {
      method: "POST",
      body: JSON.stringify({ email: inviteEmail, role: inviteRole }),
    });
    setInviteEmail("");
    setNewInviteLink(response.invite.invite_url);
    try {
      await navigator.clipboard.writeText(response.invite.invite_url);
      toast.success("Invite link copied");
    } catch {
      toast.success("Invite link created");
    }
    await loadAll();
  }

  async function updateMember(userId: string, role: Role) {
    await fetchJson(`/api/team/members/${encodeURIComponent(userId)}`, {
      method: "PATCH",
      body: JSON.stringify({ role }),
    });
    await loadAll();
  }

  async function removeMember(userId: string) {
    await fetchJson(`/api/team/members/${encodeURIComponent(userId)}`, { method: "DELETE" });
    await loadAll();
  }

  async function createApiKey(event: FormEvent) {
    event.preventDefault();
    const bankIds = apiKeyBanks
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    const response = await fetchJson<{ api_key: { key: string } }>("/api/api-keys", {
      method: "POST",
      body: JSON.stringify({ name: apiKeyName, bank_ids: bankIds.length > 0 ? bankIds : null }),
    });
    setApiKeyName("");
    setApiKeyBanks("");
    setNewApiKey(response.api_key.key);
    await loadAll();
  }

  async function revokeApiKey(id: string) {
    await fetchJson(`/api/api-keys/${encodeURIComponent(id)}`, { method: "DELETE" });
    await loadAll();
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">{t("organizationSettings")}</h1>
            <p className="text-sm text-muted-foreground">
              {currentOrg?.name || t("noOrganizationSelected")}
            </p>
          </div>
          <Button variant="outline" onClick={() => router.push("/dashboard")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Dashboard
          </Button>
        </div>

        <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
          <Card>
            <CardHeader>
              <h2 className="text-lg font-medium">Organizations</h2>
            </CardHeader>
            <CardContent className="space-y-4">
              <Select value={selectedOrgId} onValueChange={selectOrganization}>
                <SelectTrigger>
                  <SelectValue placeholder={t("selectOrganization")} />
                </SelectTrigger>
                <SelectContent>
                  {organizations.map((organization) => (
                    <SelectItem key={organization.id} value={organization.id}>
                      {organization.name} ({organization.role})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <form className="flex gap-2" onSubmit={renameOrganization}>
                <Input
                  value={orgName}
                  onChange={(event) => setOrgName(event.target.value)}
                  placeholder={t("organizationName")}
                  disabled={!canOwner}
                />
                <Button
                  type="submit"
                  size="icon"
                  disabled={!canOwner || !orgName.trim() || orgName.trim() === currentOrg?.name}
                >
                  <Save className="h-4 w-4" />
                </Button>
              </form>
              <form className="flex gap-2" onSubmit={createOrg}>
                <Input
                  value={newOrgName}
                  onChange={(event) => setNewOrgName(event.target.value)}
                  placeholder={t("newOrganization")}
                />
                <Button type="submit" size="icon" disabled={!newOrgName.trim()}>
                  <Plus className="h-4 w-4" />
                </Button>
              </form>
              <Button variant="outline" className="w-full" onClick={loadAll} disabled={loading}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <h2 className="text-lg font-medium">Team</h2>
              </CardHeader>
              <CardContent className="space-y-4">
                {canAdmin && (
                  <form
                    className="grid gap-2 md:grid-cols-[1fr_150px_auto]"
                    onSubmit={inviteMember}
                  >
                    <Input
                      type="email"
                      value={inviteEmail}
                      onChange={(event) => setInviteEmail(event.target.value)}
                      placeholder="Email"
                    />
                    <Select
                      value={inviteRole}
                      onValueChange={(value) => setInviteRole(value as Role)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="member">member</SelectItem>
                        <SelectItem value="admin">admin</SelectItem>
                      </SelectContent>
                    </Select>
                    <Button type="submit" disabled={!inviteEmail.trim()}>
                      <UserPlus className="mr-2 h-4 w-4" />
                      Invite
                    </Button>
                  </form>
                )}

                {newInviteLink && (
                  <div className="space-y-2 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950">
                    <div className="font-medium">{t("inviteLinkCreated")}</div>
                    <p>{t("inviteLinkOneTime")}</p>
                    <div className="flex items-center gap-2">
                      <code className="min-w-0 flex-1 truncate">{newInviteLink}</code>
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={() => navigator.clipboard.writeText(newInviteLink)}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                )}

                <div className="divide-y rounded-md border">
                  {members.map((member) => (
                    <div
                      key={member.user_id}
                      className="grid gap-3 p-3 md:grid-cols-[1fr_150px_auto] md:items-center"
                    >
                      <div>
                        <div className="font-medium">{member.email || member.user_id}</div>
                        <div className="text-xs text-muted-foreground">{member.user_id}</div>
                      </div>
                      <Select
                        value={member.role}
                        disabled={!canOwner}
                        onValueChange={(value) => updateMember(member.user_id, value as Role)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="owner">owner</SelectItem>
                          <SelectItem value="admin">admin</SelectItem>
                          <SelectItem value="member">member</SelectItem>
                        </SelectContent>
                      </Select>
                      <Button
                        variant="ghost"
                        size="icon"
                        disabled={!canOwner}
                        onClick={() => removeMember(member.user_id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>

                {invites.length > 0 && (
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium">Invites</h3>
                    {invites.map((invite) => (
                      <div
                        key={invite.id}
                        className="flex items-center justify-between rounded-md border p-3 text-sm"
                      >
                        <span>
                          {invite.email} ({invite.role})
                        </span>
                        <span className="text-muted-foreground">
                          {invite.revoked_at
                            ? "revoked"
                            : invite.accepted_at
                              ? "accepted"
                              : "pending"}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <h2 className="text-lg font-medium">{t("apiKeys")}</h2>
              </CardHeader>
              <CardContent className="space-y-4">
                {canAdmin && (
                  <form className="grid gap-2 md:grid-cols-[1fr_1fr_auto]" onSubmit={createApiKey}>
                    <Input
                      value={apiKeyName}
                      onChange={(event) => setApiKeyName(event.target.value)}
                      placeholder={t("keyName")}
                    />
                    <Input
                      value={apiKeyBanks}
                      onChange={(event) => setApiKeyBanks(event.target.value)}
                      placeholder="Bank ids, comma separated"
                    />
                    <Button type="submit" disabled={!apiKeyName.trim()}>
                      <KeyRound className="mr-2 h-4 w-4" />
                      Create
                    </Button>
                  </form>
                )}
                {newApiKey && (
                  <div className="flex items-center gap-2 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950">
                    <code className="min-w-0 flex-1 truncate">{newApiKey}</code>
                    <Button
                      size="icon"
                      variant="ghost"
                      onClick={() => navigator.clipboard.writeText(newApiKey)}
                    >
                      <Copy className="h-4 w-4" />
                    </Button>
                  </div>
                )}
                <div className="divide-y rounded-md border">
                  {apiKeys.map((apiKey) => (
                    <div
                      key={apiKey.id}
                      className="grid gap-3 p-3 md:grid-cols-[1fr_160px_auto] md:items-center"
                    >
                      <div>
                        <div className="font-medium">{apiKey.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {apiKey.allowed_operations?.join(", ") || "all operations"}
                        </div>
                      </div>
                      <span className="text-sm text-muted-foreground">
                        {apiKey.revoked_at ? "revoked" : "active"}
                      </span>
                      <Button
                        variant="ghost"
                        size="icon"
                        disabled={!canAdmin || Boolean(apiKey.revoked_at)}
                        onClick={() => revokeApiKey(apiKey.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </main>
  );
}

async function fetchJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(withBasePath(path), {
    ...init,
    headers: { "Content-Type": "application/json", ...init.headers },
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new Error(data?.error || `Request failed: ${response.status}`);
  return data as T;
}
