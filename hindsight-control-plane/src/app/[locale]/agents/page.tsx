"use client";

import { OperatorPanelPlaceholder } from "@/components/operator-panel-placeholder";
import { 
  History, Bot, Brain, Database, FlaskConical, ScrollText, 
  Network, Wrench, KeyRound, Shield, Settings, BookOpen 
} from "lucide-react";

const ICONS: Record<string, React.ElementType> = {
  runs: History, agents: Bot, memories: Brain, vector: Database,
  evaluation: FlaskConical, directives: ScrollText, backplane: Network,
  tools: Wrench, config: KeyRound, audit: Shield, settings: Settings,
  constitution: BookOpen,
};

export default function Page() {
  return <OperatorPanelPlaceholder panel="agents" icon={ICONS['agents']} />;
}
