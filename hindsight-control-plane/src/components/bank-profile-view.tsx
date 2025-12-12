'use client';

import { useState, useEffect } from 'react';
import { client } from '@/lib/api';
import { useBank } from '@/lib/bank-context';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { RefreshCw, Save, Brain, FileText, Clock, AlertCircle, CheckCircle, Database, Link2, FolderOpen, Activity } from 'lucide-react';

interface DispositionTraits {
  skepticism: number;
  literalism: number;
  empathy: number;
}

interface BankProfile {
  bank_id: string;
  name: string;
  disposition: DispositionTraits;
  background: string;
}

interface BankStats {
  bank_id: string;
  total_nodes: number;
  total_links: number;
  total_documents: number;
  nodes_by_fact_type: {
    world?: number;
    experience?: number;
    opinion?: number;
  };
  links_by_link_type: {
    temporal?: number;
    semantic?: number;
    entity?: number;
  };
  pending_operations: number;
  failed_operations: number;
}

interface Operation {
  id: string;
  task_type: string;
  items_count: number;
  document_id?: string;
  created_at: string;
  status: string;
  error_message?: string;
}

const TRAIT_LABELS: Record<keyof DispositionTraits, { label: string; shortLabel: string; description: string; lowLabel: string; highLabel: string }> = {
  skepticism: {
    label: 'Skepticism',
    shortLabel: 'S',
    description: 'How skeptical vs trusting when forming opinions',
    lowLabel: 'Trusting',
    highLabel: 'Skeptical'
  },
  literalism: {
    label: 'Literalism',
    shortLabel: 'L',
    description: 'How literally to interpret information when forming opinions',
    lowLabel: 'Flexible',
    highLabel: 'Literal'
  },
  empathy: {
    label: 'Empathy',
    shortLabel: 'E',
    description: 'How much to consider emotional context when forming opinions',
    lowLabel: 'Detached',
    highLabel: 'Empathetic'
  }
};

function DispositionEditor({ disposition, editMode, editDisposition, onEditChange }: {
  disposition: DispositionTraits;
  editMode: boolean;
  editDisposition: DispositionTraits;
  onEditChange: (trait: keyof DispositionTraits, value: number) => void;
}) {
  const data = editMode ? editDisposition : disposition;

  return (
    <div className="space-y-4">
      {(Object.keys(TRAIT_LABELS) as Array<keyof DispositionTraits>).map((trait) => (
        <div key={trait} className="space-y-2">
          <div className="flex justify-between items-center">
            <div>
              <label className="text-sm font-medium text-foreground">{TRAIT_LABELS[trait].label}</label>
              <p className="text-xs text-muted-foreground">{TRAIT_LABELS[trait].description}</p>
            </div>
            <span className="text-sm font-bold text-primary">{data[trait]}/5</span>
          </div>
          {editMode ? (
            <>
              <div className="flex justify-between text-[10px] text-muted-foreground">
                <span>{TRAIT_LABELS[trait].lowLabel}</span>
                <span>{TRAIT_LABELS[trait].highLabel}</span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                step="1"
                value={editDisposition[trait]}
                onChange={(e) => onEditChange(trait, parseInt(e.target.value))}
                className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
              />
            </>
          ) : (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">{TRAIT_LABELS[trait].lowLabel}</span>
              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all"
                  style={{ width: `${((data[trait] - 1) / 4) * 100}%` }}
                />
              </div>
              <span className="text-xs text-muted-foreground">{TRAIT_LABELS[trait].highLabel}</span>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function BankProfileView() {
  const { currentBank } = useBank();
  const [profile, setProfile] = useState<BankProfile | null>(null);
  const [stats, setStats] = useState<BankStats | null>(null);
  const [operations, setOperations] = useState<Operation[]>([]);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [editMode, setEditMode] = useState(false);

  // Edit state
  const [editBackground, setEditBackground] = useState('');
  const [editDisposition, setEditDisposition] = useState<DispositionTraits>({
    skepticism: 3,
    literalism: 3,
    empathy: 3
  });

  const loadData = async () => {
    if (!currentBank) return;

    setLoading(true);
    try {
      const [profileData, statsData, opsData] = await Promise.all([
        client.getBankProfile(currentBank),
        client.getBankStats(currentBank),
        client.listOperations(currentBank)
      ]);
      setProfile(profileData);
      setStats(statsData as BankStats);
      setOperations((opsData as any)?.operations || []);

      // Initialize edit state
      setEditBackground(profileData.background);
      setEditDisposition(profileData.disposition);
    } catch (error) {
      console.error('Error loading bank profile:', error);
      alert('Error loading bank profile: ' + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!currentBank) return;

    setSaving(true);
    try {
      await client.updateBankProfile(currentBank, {
        background: editBackground,
        disposition: editDisposition
      });
      await loadData();
      setEditMode(false);
    } catch (error) {
      console.error('Error saving bank profile:', error);
      alert('Error saving bank profile: ' + (error as Error).message);
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    if (profile) {
      setEditBackground(profile.background);
      setEditDisposition(profile.disposition);
    }
    setEditMode(false);
  };

  useEffect(() => {
    if (currentBank) {
      loadData();
      // Refresh operations every 5 seconds
      const interval = setInterval(loadData, 5000);
      return () => clearInterval(interval);
    }
  }, [currentBank]);

  if (!currentBank) {
    return (
      <Card>
        <CardContent className="p-10 text-center">
          <h3 className="text-xl font-semibold mb-2 text-card-foreground">No Bank Selected</h3>
          <p className="text-muted-foreground">Please select a memory bank from the dropdown above to view its profile.</p>
        </CardContent>
      </Card>
    );
  }

  if (loading && !profile) {
    return (
      <Card>
        <CardContent className="text-center py-10">
          <Clock className="w-12 h-12 mx-auto mb-3 text-muted-foreground animate-pulse" />
          <div className="text-lg text-muted-foreground">Loading profile...</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with actions */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-foreground">{profile?.name || currentBank}</h2>
          <p className="text-sm text-muted-foreground font-mono">{currentBank}</p>
        </div>
        <div className="flex gap-2">
          {editMode ? (
            <>
              <Button onClick={handleCancel} variant="outline" disabled={saving}>
                Cancel
              </Button>
              <Button onClick={handleSave} disabled={saving}>
                {saving ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-2" />
                    Save Changes
                  </>
                )}
              </Button>
            </>
          ) : (
            <>
              <Button onClick={loadData} variant="outline" size="sm">
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
              <Button onClick={() => setEditMode(true)} size="sm">
                Edit Profile
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Stats Overview - Compact cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="bg-gradient-to-br from-blue-500/10 to-blue-600/5 border-blue-500/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-blue-500/20">
                  <Database className="w-5 h-5 text-blue-500" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground font-medium">Memories</p>
                  <p className="text-2xl font-bold text-foreground">{stats.total_nodes}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-purple-500/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-500/20">
                  <Link2 className="w-5 h-5 text-purple-500" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground font-medium">Links</p>
                  <p className="text-2xl font-bold text-foreground">{stats.total_links}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-emerald-500/10 to-emerald-600/5 border-emerald-500/20">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-emerald-500/20">
                  <FolderOpen className="w-5 h-5 text-emerald-500" />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground font-medium">Documents</p>
                  <p className="text-2xl font-bold text-foreground">{stats.total_documents}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className={`bg-gradient-to-br ${stats.pending_operations > 0 ? 'from-amber-500/10 to-amber-600/5 border-amber-500/20' : 'from-slate-500/10 to-slate-600/5 border-slate-500/20'}`}>
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${stats.pending_operations > 0 ? 'bg-amber-500/20' : 'bg-slate-500/20'}`}>
                  <Activity className={`w-5 h-5 ${stats.pending_operations > 0 ? 'text-amber-500 animate-pulse' : 'text-slate-500'}`} />
                </div>
                <div>
                  <p className="text-xs text-muted-foreground font-medium">Pending</p>
                  <p className="text-2xl font-bold text-foreground">{stats.pending_operations}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Memory Type Breakdown */}
      {stats && (
        <div className="grid grid-cols-3 gap-3">
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 text-center">
            <p className="text-xs text-blue-600 dark:text-blue-400 font-semibold uppercase tracking-wide">World Facts</p>
            <p className="text-2xl font-bold text-blue-600 dark:text-blue-400 mt-1">{stats.nodes_by_fact_type?.world || 0}</p>
          </div>
          <div className="bg-purple-500/10 border border-purple-500/20 rounded-xl p-4 text-center">
            <p className="text-xs text-purple-600 dark:text-purple-400 font-semibold uppercase tracking-wide">Experience</p>
            <p className="text-2xl font-bold text-purple-600 dark:text-purple-400 mt-1">{stats.nodes_by_fact_type?.experience || 0}</p>
          </div>
          <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 text-center">
            <p className="text-xs text-amber-600 dark:text-amber-400 font-semibold uppercase tracking-wide">Opinions</p>
            <p className="text-2xl font-bold text-amber-600 dark:text-amber-400 mt-1">{stats.nodes_by_fact_type?.opinion || 0}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Disposition Chart */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Brain className="w-5 h-5 text-primary" />
              Disposition Profile
            </CardTitle>
            <CardDescription>Traits that shape how opinions are formed via Reflect</CardDescription>
          </CardHeader>
          <CardContent>
            {profile && (
              <DispositionEditor
                disposition={profile.disposition}
                editMode={editMode}
                editDisposition={editDisposition}
                onEditChange={(trait, value) => setEditDisposition(prev => ({ ...prev, [trait]: value }))}
              />
            )}
          </CardContent>
        </Card>

        {/* Background */}
        <div className="space-y-6">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-lg">
                <FileText className="w-5 h-5 text-primary" />
                Background
              </CardTitle>
              <CardDescription>Context used when forming opinions via Reflect</CardDescription>
            </CardHeader>
            <CardContent>
              {editMode ? (
                <Textarea
                  value={editBackground}
                  onChange={(e) => setEditBackground(e.target.value)}
                  placeholder="Enter background information..."
                  rows={5}
                  className="resize-none"
                />
              ) : (
                <p className="text-sm text-foreground whitespace-pre-wrap leading-relaxed">
                  {profile?.background || 'No background information provided.'}
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Operations Section */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Activity className="w-5 h-5 text-primary" />
                Background Operations
              </CardTitle>
              <CardDescription>Async tasks processing memories</CardDescription>
            </div>
            {stats && (stats.pending_operations > 0 || stats.failed_operations > 0) && (
              <div className="flex gap-3">
                {stats.pending_operations > 0 && (
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-amber-500/10 border border-amber-500/20">
                    <Clock className="w-3.5 h-3.5 text-amber-500" />
                    <span className="text-xs font-semibold text-amber-600 dark:text-amber-400">{stats.pending_operations} pending</span>
                  </div>
                )}
                {stats.failed_operations > 0 && (
                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-red-500/10 border border-red-500/20">
                    <AlertCircle className="w-3.5 h-3.5 text-red-500" />
                    <span className="text-xs font-semibold text-red-600 dark:text-red-400">{stats.failed_operations} failed</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {operations.length > 0 ? (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[100px]">ID</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead className="text-center">Items</TableHead>
                    <TableHead>Document</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {operations.slice(0, 10).map((op) => (
                    <TableRow key={op.id} className={op.status === 'failed' ? 'bg-red-500/5' : ''}>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {op.id.substring(0, 8)}
                      </TableCell>
                      <TableCell className="font-medium">{op.task_type}</TableCell>
                      <TableCell className="text-center">{op.items_count}</TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {op.document_id ? op.document_id.substring(0, 12) + '...' : '—'}
                      </TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {new Date(op.created_at).toLocaleString()}
                      </TableCell>
                      <TableCell>
                        {op.status === 'pending' && (
                          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                            <Clock className="w-3 h-3" />
                            pending
                          </span>
                        )}
                        {op.status === 'failed' && (
                          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/10 text-red-600 dark:text-red-400 border border-red-500/20" title={op.error_message}>
                            <AlertCircle className="w-3 h-3" />
                            failed
                          </span>
                        )}
                        {op.status === 'completed' && (
                          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                            <CheckCircle className="w-3 h-3" />
                            done
                          </span>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : (
            <p className="text-muted-foreground text-center py-8 text-sm">No background operations</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
