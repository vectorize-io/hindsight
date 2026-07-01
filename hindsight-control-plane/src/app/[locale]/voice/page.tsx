"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { OperatorShell } from "@/components/operator-shell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Mic,
  Volume2,
  Activity,
  Headphones,
  Play,
  Loader2,
  CheckCircle2,
  Radio,
  Languages,
  Zap,
  Sliders,
  Clock,
} from "lucide-react";

interface StatCardProps {
  label: string;
  value: string;
  icon: React.ReactNode;
}

function StatCard({ label, value, icon }: StatCardProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
        <div className="text-muted-foreground">{icon}</div>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
      </CardContent>
    </Card>
  );
}

interface VoiceAgent {
  id: string;
  name: string;
  sttModel: string;
  ttsModel: string;
  wakeWord: string;
  sessions: number;
  status: "active" | "inactive";
}

export default function VoicePage() {
  const t = useTranslations("operator");
  const v = useTranslations("operator.voice");
  const [enableWakeWord, setEnableWakeWord] = useState(true);
  const [wakeWord, setWakeWord] = useState("Hey Hindsight");
  const [sttModel, setSttModel] = useState("whisper-large-v3");
  const [ttsModel, setTtsModel] = useState("elevenlabs-turbo");
  const [ttsVoice, setTtsVoice] = useState("nova");
  const [ttsSpeed, setTtsSpeed] = useState(1.0);
  const [language, setLanguage] = useState("auto");

  const agents: VoiceAgent[] = [
    {
      id: "va1",
      name: "Main Assistant",
      sttModel: "whisper-large-v3",
      ttsModel: "elevenlabs-turbo",
      wakeWord: "Hey Hindsight",
      sessions: 4,
      status: "active",
    },
    {
      id: "va2",
      name: "Meeting Transcriber",
      sttModel: "whisper-large-v3",
      ttsModel: "kokoro",
      wakeWord: "Start Meeting",
      sessions: 2,
      status: "active",
    },
    {
      id: "va3",
      name: "Voice Commands",
      sttModel: "whisper-medium",
      ttsModel: "openai-tts",
      wakeWord: "Command",
      sessions: 1,
      status: "active",
    },
    {
      id: "va4",
      name: "Legacy Agent",
      sttModel: "whisper-small",
      ttsModel: "elevenlabs-turbo",
      wakeWord: "Hey Hindsight",
      sessions: 0,
      status: "inactive",
    },
  ];

  return (
    <OperatorShell>
      <div className="p-6 space-y-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight flex items-center gap-3">
            <Mic className="h-6 w-6 text-primary" />
            {t("panels.voice")}
          </h1>
          <p className="text-sm text-muted-foreground mt-1">{t("descriptions.voice")}</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard label={v("statModels")} value="3" icon={<Headphones className="h-4 w-4" />} />
          <StatCard
            label={v("statActiveSessions")}
            value="7"
            icon={<Radio className="h-4 w-4" />}
          />
          <StatCard
            label={v("statUtterancesToday")}
            value="12,430"
            icon={<Activity className="h-4 w-4" />}
          />
          <StatCard label={v("statAvgLatency")} value="210ms" icon={<Zap className="h-4 w-4" />} />
        </div>

        <Tabs defaultValue="stt" className="space-y-4">
          <TabsList>
            <TabsTrigger value="stt" className="flex items-center gap-2">
              <Mic className="h-4 w-4" />
              {v("sttSettings")}
            </TabsTrigger>
            <TabsTrigger value="tts" className="flex items-center gap-2">
              <Volume2 className="h-4 w-4" />
              {v("ttsSettings")}
            </TabsTrigger>
            <TabsTrigger value="agents" className="flex items-center gap-2">
              <Headphones className="h-4 w-4" />
              {v("voiceAgents")}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="stt" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">{v("sttSettings")}</CardTitle>
                <CardDescription>
                  Configure speech-to-text models and wake word detection
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">{v("model")}</label>
                    <select
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                      value={sttModel}
                      onChange={(e) => setSttModel(e.target.value)}
                    >
                      <option value="whisper-large-v3">Whisper Large V3</option>
                      <option value="whisper-medium">Whisper Medium</option>
                      <option value="whisper-small">Whisper Small</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">{v("language")}</label>
                    <select
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                      value={language}
                      onChange={(e) => setLanguage(e.target.value)}
                    >
                      <option value="auto">Auto-detect</option>
                      <option value="en">English</option>
                      <option value="fr">French</option>
                      <option value="de">German</option>
                      <option value="es">Spanish</option>
                      <option value="ja">Japanese</option>
                      <option value="zh">Chinese</option>
                    </select>
                  </div>
                </div>
                <div className="flex items-center justify-between py-2 border-t">
                  <div>
                    <p className="text-sm font-medium">{v("wakeWord")}</p>
                    <p className="text-xs text-muted-foreground">
                      Trigger phrase for voice activation
                    </p>
                  </div>
                  <div className="flex items-center gap-3">
                    <Input
                      className="h-8 w-48 text-xs"
                      value={wakeWord}
                      disabled={!enableWakeWord}
                      onChange={(e) => setWakeWord(e.target.value)}
                    />
                    <Switch checked={enableWakeWord} onCheckedChange={setEnableWakeWord} />
                  </div>
                </div>
                <Button variant="outline" size="sm" className="flex items-center gap-2">
                  <Mic className="w-3.5 h-3.5" />
                  {v("testMicrophone")}
                </Button>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="tts" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">{v("ttsSettings")}</CardTitle>
                <CardDescription>
                  Configure text-to-speech voices and playback settings
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">{v("model")}</label>
                    <select
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                      value={ttsModel}
                      onChange={(e) => setTtsModel(e.target.value)}
                    >
                      <option value="elevenlabs-turbo">ElevenLabs Turbo</option>
                      <option value="openai-tts">OpenAI TTS</option>
                      <option value="kokoro">Kokoro</option>
                    </select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Voice</label>
                    <select
                      className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                      value={ttsVoice}
                      onChange={(e) => setTtsVoice(e.target.value)}
                    >
                      <option value="nova">Nova</option>
                      <option value="alloy">Alloy</option>
                      <option value="echo">Echo</option>
                      <option value="fable">Fable</option>
                      <option value="onyx">Onyx</option>
                      <option value="shimmer">Shimmer</option>
                    </select>
                  </div>
                </div>
                <div className="space-y-2 py-2 border-t">
                  <label className="text-sm font-medium">Speed: {ttsSpeed.toFixed(1)}x</label>
                  <input
                    type="range"
                    min={0.5}
                    max={2.0}
                    step={0.1}
                    value={ttsSpeed}
                    onChange={(e) => setTtsSpeed(Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="flex justify-between text-[10px] text-muted-foreground">
                    <span>0.5x</span>
                    <span>1.0x</span>
                    <span>2.0x</span>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Input
                    className="flex-1 h-9 text-xs"
                    placeholder="Enter sample text to speak..."
                  />
                  <Button size="sm" className="flex items-center gap-2">
                    <Play className="w-3.5 h-3.5" />
                    Test
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="agents" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {agents.length === 0 ? (
                <Card className="md:col-span-2">
                  <CardContent className="py-12 text-center text-muted-foreground">
                    {v("noModels")}
                  </CardContent>
                </Card>
              ) : (
                agents.map((agent) => (
                  <Card key={agent.id} className="hover:bg-accent/30 transition-colors">
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-sm font-medium flex items-center gap-2">
                          <Headphones className="h-4 w-4 text-primary" />
                          {agent.name}
                        </CardTitle>
                        <Badge
                          variant={agent.status === "active" ? "default" : "secondary"}
                          className="text-[10px]"
                        >
                          {agent.status}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-1.5 text-xs text-muted-foreground">
                      <div className="flex justify-between">
                        <span>STT: {agent.sttModel}</span>
                        <span>TTS: {agent.ttsModel}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>
                          {v("wakeWord")}: "{agent.wakeWord}"
                        </span>
                        <span>{agent.sessions} active sessions</span>
                      </div>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </OperatorShell>
  );
}
