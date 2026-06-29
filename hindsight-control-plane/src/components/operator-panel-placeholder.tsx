"use client";

import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { ElementType } from "react";

interface PlaceholderProps {
  panel: string;
  icon: ElementType;
  description?: string;
}

export function OperatorPanelPlaceholder({ panel, icon: Icon, description }: PlaceholderProps) {
  const t = useTranslations("operator");

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <Icon className="h-6 w-6 text-primary" />
            {t(`panels.${panel}`)}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            {description || t(`descriptions.${panel}`)}
          </p>
        </div>
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Icon className="h-5 w-5" />
              {t(`panels.${panel}`)}
            </CardTitle>
            <CardDescription>
              {description || t(`descriptions.${panel}`)}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-center py-16 text-muted-foreground">
              <div className="text-center space-y-2">
                <Icon className="h-12 w-12 mx-auto opacity-20" />
                <p>Panel coming soon</p>
                <p className="text-sm">This panel will be populated in the next phase</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </OperatorShell>
  );
}
