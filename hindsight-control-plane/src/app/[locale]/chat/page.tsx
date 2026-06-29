"use client";

import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { ChatAssistant } from "@/components/chat-assistant";
import { MessageSquare } from "lucide-react";

export default function ChatPage() {
  const t = useTranslations("operator");

  return (
    <OperatorShell>
      <div className="flex flex-col h-full">
        {/* Page header */}
        <div className="px-6 pt-4 pb-2">
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <MessageSquare className="h-6 w-6 text-primary" />
            {t("panels.chat")}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">{t("descriptions.chat")}</p>
        </div>

        {/* Full-height chat assistant */}
        <div className="flex-1 px-6 pb-4 min-h-0">
          <ChatAssistant
            standalone
            placeholder="Ask anything about your system, memories, operations..."
          />
        </div>
      </div>
    </OperatorShell>
  );
}
