"use client";

import { Suspense, useState, FormEvent, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useTranslations } from "next-intl";
import Image from "next/image";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { LanguageSwitcher } from "@/components/language-switcher";
import { sanitizeReturnTo, withBasePath } from "@/lib/base-path";

interface LoginField {
  name: string;
  type: "email" | "password" | "text";
  placeholder: string;
  autocomplete?: string;
  required?: boolean;
  modes?: string[];
  hiddenWhenInvite?: boolean;
}

interface LoginConfig {
  provider: string;
  modes?: Array<{ id: string; label: string }>;
  defaultMode?: string;
  fields: LoginField[];
  submitLabel: string;
  submitLabelsByMode?: Record<string, string>;
}

const DEFAULT_LOGIN_CONFIG: LoginConfig = {
  provider: "access_key",
  fields: [
    {
      name: "key",
      type: "password",
      placeholder: "Access Key",
      autocomplete: "off",
      required: true,
    },
  ],
  submitLabel: "Sign in",
};

function LoginForm() {
  const t = useTranslations("login");
  const [config, setConfig] = useState<LoginConfig | null>(null);
  const [mode, setMode] = useState("");
  const [values, setValues] = useState<Record<string, string>>({});
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
        const response = await fetch(withBasePath("/api/auth/provider"));
        const data = await response.json();
        const nextConfig = data?.login || DEFAULT_LOGIN_CONFIG;
        setConfig(nextConfig);
        setMode(nextConfig.defaultMode || nextConfig.modes?.[0]?.id || "");
      } catch {
        setConfig(DEFAULT_LOGIN_CONFIG);
      }
    };
    loadAuthProvider();
  }, []);

  useEffect(() => {
    const firstField = visibleFields(config, mode, Boolean(inviteToken))[0];
    if (firstField) document.getElementById(`login-${firstField.name}`)?.focus();
  }, [config, mode, inviteToken]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!config) return;
    setError("");
    setLoading(true);

    try {
      const res = await fetch(withBasePath("/api/auth/login"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...values,
          mode: mode || undefined,
          invite_token: inviteToken,
        }),
      });

      if (res.ok) {
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

  const fields = visibleFields(config, mode, Boolean(inviteToken));
  const submitLabel =
    (mode && config?.submitLabelsByMode?.[mode]) || config?.submitLabel || t("signIn");

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
          {config?.modes && config.modes.length > 1 && (
            <div className="mb-4 grid grid-cols-2 rounded-md border p-1">
              {config.modes.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`rounded px-3 py-2 text-sm ${mode === item.id ? "bg-primary text-primary-foreground" : ""}`}
                  onClick={() => setMode(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          )}
          <form onSubmit={handleSubmit} className="space-y-4">
            {fields.map((field) => (
              <Input
                key={field.name}
                id={`login-${field.name}`}
                type={field.type}
                placeholder={field.placeholder}
                value={values[field.name] || ""}
                onChange={(e) =>
                  setValues((current) => ({ ...current, [field.name]: e.target.value }))
                }
                autoComplete={field.autocomplete}
              />
            ))}

            {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

            <Button
              type="submit"
              className="w-full"
              disabled={
                loading || !config || fields.some((field) => field.required && !values[field.name])
              }
            >
              {loading ? t("signingIn") : submitLabel}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

function visibleFields(config: LoginConfig | null, mode: string, hasInvite: boolean): LoginField[] {
  if (!config) return [];
  return config.fields.filter((field) => {
    if (field.modes && !field.modes.includes(mode)) return false;
    if (field.hiddenWhenInvite && hasInvite) return false;
    return true;
  });
}

export default function LoginPage() {
  return (
    <Suspense>
      <LoginForm />
    </Suspense>
  );
}
