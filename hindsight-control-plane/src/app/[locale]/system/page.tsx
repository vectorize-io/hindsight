"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  RefreshCw,
  Play,
  Square,
  RotateCw,
  Activity,
  Database,
  Cpu,
  HardDrive,
  Monitor,
  FileText,
  Plus,
  Minus,
  Pause,
} from "lucide-react";
import { toast } from "sonner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";

interface ServiceStatus {
  name: string;
  port: number;
  status: "running" | "stopped" | "error";
  pid?: number;
  uptime?: string;
  health?: string;
  cpu?: number;
  memory?: number;
}

interface Operation {
  id: string;
  task_type: string;
  items_count: number;
  document_id: string | null;
  created_at: string;
  status: "pending" | "processing" | "completed" | "failed";
  error_message: string | null;
  retry_count: number;
  next_retry_at: string | null;
}

export default function SystemPage() {
  const [services, setServices] = useState<ServiceStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [mounted, setMounted] = useState(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [logs, setLogs] = useState<{ [key: string]: string[] }>({});
  const [selectedLogService, setSelectedLogService] = useState<string>("api");
  const [workerCount, setWorkerCount] = useState(2);
  const [operations, setOperations] = useState<Operation[]>([]);
  const [operationsLoading, setOperationsLoading] = useState(false);

  const fetchServiceStatus = async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/system/services");
      const data = await response.json();
      setServices(data.services || []);
      setLastUpdate(new Date());
    } catch (error) {
      console.error("Failed to fetch service status:", error);
      toast.error("Failed to fetch service status");
    } finally {
      setLoading(false);
    }
  };

  const handleServiceAction = async (serviceName: string, action: "start" | "stop" | "restart") => {
    const serviceKey = serviceName.toLowerCase().replace(/ /g, "-");
    setActionLoading(`${serviceKey}-${action}`);

    try {
      const response = await fetch(`/api/system/services/${serviceKey}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });

      const result = await response.json();

      if (result.success) {
        toast.success(`${serviceName}: ${action} successful`);
        setTimeout(fetchServiceStatus, 2000); // Refresh after 2s
      } else {
        toast.error(`${serviceName}: ${action} failed - ${result.error || "Unknown error"}`);
      }
    } catch (error) {
      console.error(`Failed to ${action} ${serviceName}:`, error);
      toast.error(`Failed to ${action} ${serviceName}`);
    } finally {
      setActionLoading(null);
    }
  };

  const fetchLogs = async (service: string) => {
    try {
      const response = await fetch(`/api/system/logs?service=${service}&lines=100`);
      const data = await response.json();
      if (data.lines) {
        setLogs((prev) => ({ ...prev, [service]: data.lines }));
      }
    } catch (error) {
      console.error(`Failed to fetch logs for ${service}:`, error);
    }
  };

  const fetchOperations = async () => {
    setOperationsLoading(true);
    try {
      const response = await fetch("/api/system/operations?bank_id=default&limit=20");
      const data = await response.json();
      setOperations(data.operations || []);
    } catch (error) {
      console.error("Failed to fetch operations:", error);
    } finally {
      setOperationsLoading(false);
    }
  };

  const handleWorkerScale = async (newCount: number) => {
    if (newCount < 0 || newCount > 10) {
      toast.error("Worker count must be between 0 and 10");
      return;
    }

    setActionLoading("workers-scale");
    try {
      const response = await fetch("/api/system/services/workers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "scale", count: newCount }),
      });

      const result = await response.json();
      if (result.success) {
        setWorkerCount(newCount);
        toast.success(`Workers scaled to ${newCount}`);
        setTimeout(fetchServiceStatus, 2000);
      } else {
        toast.error(`Failed to scale workers: ${result.error}`);
      }
    } catch (error) {
      console.error("Failed to scale workers:", error);
      toast.error("Failed to scale workers");
    } finally {
      setActionLoading(null);
    }
  };

  useEffect(() => {
    setMounted(true);
    fetchServiceStatus();
    fetchLogs(selectedLogService);
    fetchOperations();

    const statusInterval = autoRefresh ? setInterval(fetchServiceStatus, 10000) : null;
    const logsInterval = setInterval(() => fetchLogs(selectedLogService), 5000);
    const opsInterval = autoRefresh ? setInterval(fetchOperations, 5000) : null;

    return () => {
      if (statusInterval) clearInterval(statusInterval);
      clearInterval(logsInterval);
      if (opsInterval) clearInterval(opsInterval);
    };
  }, [autoRefresh, selectedLogService]);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "running":
        return <Badge className="bg-green-500">Running</Badge>;
      case "stopped":
        return <Badge variant="destructive">Stopped</Badge>;
      case "error":
        return <Badge variant="destructive">Error</Badge>;
      default:
        return <Badge variant="secondary">Unknown</Badge>;
    }
  };

  const getServiceIcon = (name: string) => {
    if (name.includes("API")) return <Database className="h-5 w-5" />;
    if (name.includes("Control")) return <Monitor className="h-5 w-5" />;
    if (name.includes("Ollama")) return <Cpu className="h-5 w-5" />;
    if (name.includes("PostgreSQL")) return <HardDrive className="h-5 w-5" />;
    if (name.includes("Workers")) return <Activity className="h-5 w-5" />;
    return <Monitor className="h-5 w-5" />;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">System Monitor</h1>
          <p className="text-muted-foreground">Real-time service health and control</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Auto-refresh</span>
            <Switch checked={autoRefresh} onCheckedChange={setAutoRefresh} />
          </div>
          {mounted && lastUpdate && (
            <span className="text-sm text-muted-foreground">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </span>
          )}
          <Button variant="outline" size="sm" onClick={fetchServiceStatus} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </div>

      <Tabs defaultValue="services" className="space-y-4">
        <TabsList>
          <TabsTrigger value="services">Services</TabsTrigger>
          <TabsTrigger value="operations">Operations</TabsTrigger>
          <TabsTrigger value="logs">Logs</TabsTrigger>
        </TabsList>

        <TabsContent value="services" className="space-y-4">
          {/* Services Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {loading && services.length === 0 ? (
              <>
                {[1, 2, 3, 4, 5].map((i) => (
                  <Card key={i} className="animate-pulse">
                    <CardHeader>
                      <div className="h-6 w-32 bg-muted rounded" />
                      <div className="h-4 w-20 bg-muted rounded mt-2" />
                    </CardHeader>
                    <CardContent>
                      <div className="h-20 w-full bg-muted rounded" />
                    </CardContent>
                  </Card>
                ))}
              </>
            ) : (
              services.map((service) => {
                const serviceKey = service.name.toLowerCase().replace(/ /g, "-");
                const isWorkers = service.name === "Workers";
                const isControlPlane = service.name === "Control Plane";

                return (
                  <Card key={service.name}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          {getServiceIcon(service.name)}
                          <CardTitle className="text-lg">{service.name}</CardTitle>
                        </div>
                        {getStatusBadge(service.status)}
                      </div>
                      <CardDescription>Port: {service.port}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {/* Service Info */}
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        {service.pid && (
                          <div>
                            <span className="text-muted-foreground">PID:</span> {service.pid}
                          </div>
                        )}
                        {service.uptime && (
                          <div>
                            <span className="text-muted-foreground">Uptime:</span> {service.uptime}
                          </div>
                        )}
                        {service.health && (
                          <div className="col-span-2">
                            <span className="text-muted-foreground">Health:</span> {service.health}
                          </div>
                        )}
                        {service.cpu !== undefined && (
                          <div>
                            <span className="text-muted-foreground">CPU:</span>{" "}
                            {service.cpu.toFixed(1)}%
                          </div>
                        )}
                        {service.memory !== undefined && (
                          <div>
                            <span className="text-muted-foreground">Memory:</span>{" "}
                            {service.memory.toFixed(1)}%
                          </div>
                        )}
                      </div>

                      {/* Service Controls */}
                      {!isControlPlane && (
                        <div className="space-y-2">
                          {/* Worker Scaling */}
                          {isWorkers && service.status === "running" && (
                            <div className="flex items-center gap-2 pb-2">
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => handleWorkerScale(workerCount - 1)}
                                disabled={workerCount <= 0 || actionLoading === "workers-scale"}
                              >
                                <Minus className="h-3 w-3" />
                              </Button>
                              <span className="text-sm flex-1 text-center">
                                {workerCount} workers
                              </span>
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => handleWorkerScale(workerCount + 1)}
                                disabled={workerCount >= 10 || actionLoading === "workers-scale"}
                              >
                                <Plus className="h-3 w-3" />
                              </Button>
                            </div>
                          )}

                          <div className="flex gap-2 pt-2 border-t">
                            {service.status === "stopped" ? (
                              <Button
                                size="sm"
                                variant="default"
                                className="flex-1"
                                onClick={() => handleServiceAction(service.name, "start")}
                                disabled={actionLoading === `${serviceKey}-start`}
                              >
                                <Play className="h-3 w-3 mr-1" />
                                Start
                              </Button>
                            ) : (
                              <>
                                <Button
                                  size="sm"
                                  variant="destructive"
                                  className="flex-1"
                                  onClick={() => handleServiceAction(service.name, "stop")}
                                  disabled={actionLoading === `${serviceKey}-stop`}
                                >
                                  <Square className="h-3 w-3 mr-1" />
                                  Stop
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="flex-1"
                                  onClick={() => handleServiceAction(service.name, "restart")}
                                  disabled={actionLoading === `${serviceKey}-restart`}
                                >
                                  <RotateCw className="h-3 w-3 mr-1" />
                                  Restart
                                </Button>
                              </>
                            )}
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                );
              })
            )}
          </div>
        </TabsContent>

        <TabsContent value="operations" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Total Operations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{operations.length}</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Processing
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-blue-600">
                  {operations.filter((op) => op.status === "processing").length}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Pending</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-yellow-600">
                  {operations.filter((op) => op.status === "pending").length}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">Failed</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-red-600">
                  {operations.filter((op) => op.status === "failed").length}
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Active Operations
              </CardTitle>
              <CardDescription>Real-time worker operations and task status</CardDescription>
            </CardHeader>
            <CardContent>
              {operationsLoading && operations.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">Loading operations...</div>
              ) : operations.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">No operations found</div>
              ) : (
                <div className="space-y-2">
                  {operations.map((op) => {
                    const age = Math.floor((Date.now() - new Date(op.created_at).getTime()) / 1000);
                    const ageStr =
                      age < 60
                        ? `${age}s`
                        : age < 3600
                          ? `${Math.floor(age / 60)}m`
                          : `${Math.floor(age / 3600)}h`;
                    const isStuck = age > 300; // 5 minutes

                    return (
                      <div
                        key={op.id}
                        className={`flex items-center justify-between p-3 rounded border ${
                          isStuck ? "border-red-500 bg-red-50" : "border-gray-200"
                        }`}
                      >
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-sm">{op.task_type}</span>
                            {op.status === "processing" && (
                              <Badge variant="default" className="bg-blue-500">
                                Processing
                              </Badge>
                            )}
                            {op.status === "pending" && <Badge variant="secondary">Pending</Badge>}
                            {op.status === "failed" && <Badge variant="destructive">Failed</Badge>}
                            {op.status === "completed" && (
                              <Badge className="bg-green-500">Completed</Badge>
                            )}
                            {isStuck && (
                              <Badge variant="destructive" className="animate-pulse">
                                STUCK
                              </Badge>
                            )}
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">
                            ID: {op.id.substring(0, 8)}... | Items: {op.items_count} | Age: {ageStr}
                            {op.retry_count > 0 && ` | Retries: ${op.retry_count}`}
                          </div>
                          {op.error_message && (
                            <div className="text-xs text-red-600 mt-1">
                              Error: {op.error_message}
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="logs" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Service Logs
                </CardTitle>
                <select
                  value={selectedLogService}
                  onChange={(e) => setSelectedLogService(e.target.value)}
                  className="px-3 py-1 border rounded text-sm"
                >
                  <option value="api">Hindsight API</option>
                  <option value="control-plane">Control Plane</option>
                  <option value="ollama-embeddings">Ollama Embeddings</option>
                  <option value="ollama-llm">Ollama LLM</option>
                  <option value="worker-1">Worker 1</option>
                  <option value="worker-2">Worker 2</option>
                  <option value="worker-3">Worker 3</option>
                  <option value="worker-4">Worker 4</option>
                </select>
              </div>
            </CardHeader>
            <CardContent>
              <div className="bg-black text-green-400 p-4 rounded font-mono text-xs h-[600px] overflow-y-auto">
                {logs[selectedLogService]?.length > 0 ? (
                  logs[selectedLogService].map((line, i) => (
                    <div key={i} className="whitespace-pre-wrap break-words">
                      {line}
                    </div>
                  ))
                ) : (
                  <div className="text-gray-500">No logs available or file not found</div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
