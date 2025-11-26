'use client';

import { useState, useEffect } from 'react';
import { dataplaneClient } from '@/lib/api';
import { useAgent } from '@/lib/agent-context';

interface Entity {
  id: string;
  canonical_name: string;
  mention_count: number;
  first_seen?: string;
  last_seen?: string;
  metadata?: Record<string, any>;
}

interface EntityDetail extends Entity {
  observations: Array<{
    text: string;
    mentioned_at?: string;
  }>;
}

export function EntitiesView() {
  const { currentAgent } = useAgent();
  const [entities, setEntities] = useState<Entity[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedEntity, setSelectedEntity] = useState<EntityDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [regenerating, setRegenerating] = useState(false);

  const loadEntities = async () => {
    if (!currentAgent) return;

    setLoading(true);
    try {
      const result: any = await dataplaneClient.listEntities({
        agent_id: currentAgent,
        limit: 100,
      });
      setEntities(result.entities || []);
    } catch (error) {
      console.error('Error loading entities:', error);
      alert('Error loading entities: ' + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const loadEntityDetail = async (entityId: string) => {
    if (!currentAgent) return;

    setLoadingDetail(true);
    try {
      const result: any = await dataplaneClient.getEntity(entityId, currentAgent);
      setSelectedEntity(result);
    } catch (error) {
      console.error('Error loading entity detail:', error);
      alert('Error loading entity detail: ' + (error as Error).message);
    } finally {
      setLoadingDetail(false);
    }
  };

  const regenerateObservations = async () => {
    if (!currentAgent || !selectedEntity) return;

    setRegenerating(true);
    try {
      await dataplaneClient.regenerateEntityObservations(selectedEntity.id, currentAgent);
      // Reload entity detail to show new observations
      await loadEntityDetail(selectedEntity.id);
    } catch (error) {
      console.error('Error regenerating observations:', error);
      alert('Error regenerating observations: ' + (error as Error).message);
    } finally {
      setRegenerating(false);
    }
  };

  useEffect(() => {
    if (currentAgent) {
      loadEntities();
      setSelectedEntity(null);
    }
  }, [currentAgent]);

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return 'N/A';
    return new Date(dateStr).toLocaleDateString();
  };

  return (
    <div className="flex gap-4">
      {/* Entity List */}
      <div className="flex-1">
        <div className="mb-4 p-2.5 bg-card rounded-lg border-2 border-primary flex gap-4 items-center">
          <button
            onClick={loadEntities}
            disabled={loading}
            className="px-5 py-2 bg-primary text-primary-foreground rounded font-bold text-sm hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Loading...' : entities.length > 0 ? 'Refresh Entities' : 'Load Entities'}
          </button>
          {entities.length > 0 && (
            <span className="text-muted-foreground text-sm">
              ({entities.length} entities)
            </span>
          )}
        </div>

        {entities.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr>
                  <th className="p-2.5 text-left border border-border bg-card text-card-foreground">ID</th>
                  <th className="p-2.5 text-left border border-border bg-card text-card-foreground">Name</th>
                  <th className="p-2.5 text-left border border-border bg-card text-card-foreground">Mentions</th>
                  <th className="p-2.5 text-left border border-border bg-card text-card-foreground">First Seen</th>
                  <th className="p-2.5 text-left border border-border bg-card text-card-foreground">Last Seen</th>
                </tr>
              </thead>
              <tbody>
                {entities.map((entity) => (
                  <tr
                    key={entity.id}
                    onClick={() => loadEntityDetail(entity.id)}
                    className={`cursor-pointer hover:bg-muted ${
                      selectedEntity?.id === entity.id ? 'bg-accent' : 'bg-background'
                    }`}
                  >
                    <td className="p-2 border border-border text-xs text-muted-foreground font-mono" title={entity.id}>{entity.id.slice(0, 8)}...</td>
                    <td className="p-2 border border-border font-medium">{entity.canonical_name}</td>
                    <td className="p-2 border border-border">{entity.mention_count}</td>
                    <td className="p-2 border border-border">{formatDate(entity.first_seen)}</td>
                    <td className="p-2 border border-border">{formatDate(entity.last_seen)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : !loading && (
          <div className="p-10 text-center text-muted-foreground bg-muted rounded">
            No entities found. Entities are extracted from facts when memories are added.
          </div>
        )}
      </div>

      {/* Entity Detail Panel */}
      {selectedEntity && (
        <div className="w-96 bg-card border-2 border-primary rounded-lg p-4">
          <div className="flex justify-between items-start mb-4">
            <h3 className="text-lg font-bold text-card-foreground">{selectedEntity.canonical_name}</h3>
            <button
              onClick={() => setSelectedEntity(null)}
              className="text-muted-foreground hover:text-foreground"
            >
              X
            </button>
          </div>

          <div className="text-sm text-muted-foreground mb-4">
            <div>Mentions: {selectedEntity.mention_count}</div>
            <div>First seen: {formatDate(selectedEntity.first_seen)}</div>
            <div>Last seen: {formatDate(selectedEntity.last_seen)}</div>
          </div>

          <div className="mb-4">
            <div className="flex justify-between items-center mb-2">
              <h4 className="font-bold text-card-foreground">Observations</h4>
              <button
                onClick={regenerateObservations}
                disabled={regenerating}
                className="px-3 py-1 bg-secondary text-secondary-foreground rounded text-xs font-bold hover:opacity-90 disabled:opacity-50"
              >
                {regenerating ? 'Regenerating...' : 'Regenerate'}
              </button>
            </div>

            {loadingDetail ? (
              <div className="text-muted-foreground text-sm">Loading observations...</div>
            ) : selectedEntity.observations && selectedEntity.observations.length > 0 ? (
              <ul className="space-y-2">
                {selectedEntity.observations.map((obs, idx) => (
                  <li key={idx} className="p-2 bg-muted rounded text-sm">
                    <div>{obs.text}</div>
                    {obs.mentioned_at && (
                      <div className="text-xs text-muted-foreground mt-1">
                        {formatDate(obs.mentioned_at)}
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            ) : (
              <div className="text-muted-foreground text-sm">
                No observations yet. Click &quot;Regenerate&quot; to generate observations from facts.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
