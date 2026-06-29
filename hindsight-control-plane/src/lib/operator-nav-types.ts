export type OperatorPanel =
  | "cockpit"
  | "chat"
  | "runs"
  | "agents"
  | "memories"
  | "vector"
  | "evaluation"
  | "directives"
  | "backplane"
  | "tools"
  | "config"
  | "audit"
  | "settings"
  | "constitution";

export const OPERATOR_PANELS: { id: OperatorPanel; route: string; icon: string }[] = [
  { id: "cockpit", route: "/cockpit", icon: "LayoutDashboard" },
  { id: "chat", route: "/chat", icon: "MessageSquare" },
  { id: "runs", route: "/runs", icon: "Timeline" },
  { id: "agents", route: "/agents", icon: "Bot" },
  { id: "memories", route: "/memories", icon: "Brain" },
  { id: "vector", route: "/vector", icon: "Database" },
  { id: "evaluation", route: "/evaluation", icon: "FlaskConical" },
  { id: "directives", route: "/directives", icon: "ScrollText" },
  { id: "backplane", route: "/backplane", icon: "Network" },
  { id: "tools", route: "/tools", icon: "Wrench" },
  { id: "config", route: "/config", icon: "KeyRound" },
  { id: "audit", route: "/audit", icon: "Shield" },
  { id: "settings", route: "/settings", icon: "Settings" },
  { id: "constitution", route: "/constitution", icon: "BookOpen" },
];

export const OPERATOR_PANEL_LABELS: Record<OperatorPanel, { primary: string; secondary: string }> =
  {
    cockpit: { primary: "overview", secondary: "cockpit" },
    chat: { primary: "assistant", secondary: "chat" },
    runs: { primary: "timeline", secondary: "runs" },
    agents: { primary: "registry", secondary: "agents" },
    memories: { primary: "console", secondary: "memories" },
    vector: { primary: "explorer", secondary: "vector" },
    evaluation: { primary: "lab", secondary: "evaluation" },
    directives: { primary: "directives", secondary: "" },
    backplane: { primary: "system map", secondary: "backplane" },
    tools: { primary: "registry", secondary: "tools" },
    config: { primary: "secrets", secondary: "config" },
    audit: { primary: "& provenance", secondary: "audit" },
    settings: { primary: "system", secondary: "settings" },
    constitution: { primary: "locked", secondary: "prime directives" },
  };
