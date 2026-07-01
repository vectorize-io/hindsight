"use client";

import { Suspense, useState, FormEvent, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card";
import { LanguageSwitcher } from "@/components/language-switcher";
import { sanitizeReturnTo, withBasePath } from "@/lib/base-path";
import Image from "next/image";

function LoginForm() {
  const t = useTranslations("login");
  const [authProvider, setAuthProvider] = useState<
    "access_key" | "supabase_org" | "disabled" | null
  >(null);
  const [key, setKey] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [organizationName, setOrganizationName] = useState("");
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();

  // Validate returnTo to prevent open-redirect via crafted login links.
  const returnTo = sanitizeReturnTo(searchParams.get("returnTo"));
  const inviteToken = searchParams.get("invite") || undefined;

  useEffect(() => {
    const loadAuthProvider = async () => {
      try {
        const response = await fetch(withBasePath("/api/version"));
        const data = await response.json();
        setAuthProvider(data?.features?.auth_provider || "access_key");
      } catch {
        setAuthProvider("access_key");
      }
    };
    loadAuthProvider();
  }, []);

  useEffect(() => {
    const input = document.getElementById(authProvider === "supabase_org" ? "email" : "access-key");
    input?.focus();
  }, [authProvider]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const body =
        authProvider === "supabase_org"
          ? {
              mode,
              email,
              password,
              organization_name: organizationName || undefined,
              invite_token: inviteToken,
            }
          : { key };
      const res = await fetch(withBasePath("/api/auth/login"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (res.ok) {
        // Navigate to the returnTo URL
        router.push(returnTo);
        router.refresh();
      } else {
        const data = await res.json().catch(() => null);
        setError(data?.error || t("invalidKey"));
      }
    } catch {
      setError(t("connectFailed"));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="absolute top-4 right-4">
        <LanguageSwitcher />
      </div>
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <Image
            src={withBasePath("/logo.png")}
            alt="Hindsight"
            width={160}
            height={160}
            className="mx-auto"
            unoptimized
          />
          <CardDescription>{t("description")}</CardDescription>
        </CardHeader>
        <CardContent>
          {authProvider === "supabase_org" && (
            <div className="mb-4 grid grid-cols-2 rounded-md border p-1">
              <button
                type="button"
                className={`rounded px-3 py-2 text-sm ${mode === "login" ? "bg-primary text-primary-foreground" : ""}`}
                onClick={() => setMode("login")}
              >
                {t("signInAction")}
              </button>
              <button
                type="button"
                className={`rounded px-3 py-2 text-sm ${mode === "signup" ? "bg-primary text-primary-foreground" : ""}`}
                onClick={() => setMode("signup")}
              >
                {t("signUpAction")}
              </button>
            </div>
          )}
          <form onSubmit={handleSubmit} className="space-y-4">
            {authProvider === "supabase_org" ? (
              <>
                <Input
                  id="email"
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                />
                <Input
                  id="password"
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete={mode === "signup" ? "new-password" : "current-password"}
                />
                {mode === "signup" && !inviteToken && (
                  <Input
                    id="organization-name"
                    type="text"
                    placeholder={t("organizationNamePlaceholder")}
                    value={organizationName}
                    onChange={(e) => setOrganizationName(e.target.value)}
                    autoComplete="organization"
                  />
                )}
              </>
            ) : (
              <Input
                id="access-key"
                type="password"
                placeholder={t("accessKeyPlaceholder")}
                value={key}
                onChange={(e) => setKey(e.target.value)}
                autoComplete="off"
              />
            )}

            {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

            <Button
              type="submit"
              className="w-full"
              disabled={
                loading ||
                authProvider === null ||
                (authProvider === "supabase_org" ? !email || !password : !key)
              }
            >
              {loading
                ? t("signingIn")
                : mode === "signup" && authProvider === "supabase_org"
                  ? t("signUpAction")
                  : t("signIn")}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense>
      <LoginForm />
    </Suspense>
  );
}
