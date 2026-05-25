"use client";

import { Languages } from "lucide-react";
import { useTranslation } from "react-i18next";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useLocale } from "@/lib/i18n/provider";
import type { SupportedLocale } from "@/lib/i18n/resources";
import { cn } from "@/lib/utils";

export function LanguageSelector({ className }: { className?: string }) {
  const { t } = useTranslation();
  const { locale, setLocale, supportedLocales } = useLocale();

  return (
    <Select value={locale} onValueChange={(value) => setLocale(value as SupportedLocale)}>
      <SelectTrigger
        aria-label={t("common.language.label")}
        className={cn("h-9 w-[150px] gap-2", className)}
      >
        <Languages className="h-4 w-4 shrink-0 opacity-70" />
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {supportedLocales.map((language) => (
          <SelectItem key={language.code} value={language.code}>
            {language.nativeLabel}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
