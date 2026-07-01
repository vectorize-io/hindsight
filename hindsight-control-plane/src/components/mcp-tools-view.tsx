"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { RefreshCw, Loader2, Wrench, ArrowRight, Terminal } from "lucide-react";

interface MCPTool {
  name: string;
  method: string;
  api_path: string;
  description: string;
}

export function MCPToolsView() {
  const [tools, setTools] = useState<MCPTool[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchTools = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/central/mcp/tools");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setTools(data.tools || []);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTools();
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-medium">MCP Tools Registry</h3>
          <p className="text-xs text-muted-foreground">
            {loading ? "Loading..." : `${tools.length} registered tools`}
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={fetchTools} disabled={loading}>
          <RefreshCw className={`h-3 w-3 mr-1 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-8 text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin mr-2" /> Loading MCP tools...
        </div>
      ) : error ? (
        <Card>
          <CardContent className="p-6 text-center">
            <Terminal className="h-6 w-6 mx-auto mb-2 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">MCP tools unavailable</p>
            <p className="text-xs mt-1 text-muted-foreground">{error}</p>
          </CardContent>
        </Card>
      ) : tools.length === 0 ? (
        <Card>
          <CardContent className="p-6 text-center text-muted-foreground">
            <Wrench className="h-6 w-6 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No MCP tools registered</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {tools.map((tool) => (
            <Card key={tool.name} className="hover:bg-accent/30 transition-colors">
              <CardHeader className="pb-1 pt-2.5 px-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-xs font-mono font-medium">{tool.name}</CardTitle>
                  <Badge variant="outline" className="text-[9px] h-4 px-1">
                    {tool.method}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="px-3 pb-2.5 space-y-1">
                <p className="text-[11px] text-muted-foreground">{tool.description}</p>
                <p className="text-[9px] font-mono text-muted-foreground truncate">
                  {tool.api_path}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
