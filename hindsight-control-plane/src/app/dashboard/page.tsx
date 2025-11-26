'use client';

import { useState } from 'react';
import { AgentSelector } from '@/components/agent-selector';
import { DataView } from '@/components/data-view';
import { DocumentsView } from '@/components/documents-view';
import { EntitiesView } from '@/components/entities-view';
import { ThinkView } from '@/components/think-view';
import { AddMemoryView } from '@/components/add-memory-view';
import { StatsView } from '@/components/stats-view';
import { SearchDebugView } from '@/components/search-debug-view';
import { useAgent } from '@/lib/agent-context';

type MainTab = 'data' | 'documents' | 'entities' | 'search' | 'stats' | 'think' | 'add';
type DataSubTab = 'world' | 'agent' | 'opinion';

export default function DashboardPage() {
  const [mainTab, setMainTab] = useState<MainTab>('data');
  const [dataSubTab, setDataSubTab] = useState<DataSubTab>('world');
  const { currentAgent } = useAgent();

  const TabButton = ({ tab, label }: { tab: MainTab; label: string }) => (
    <button
      onClick={() => setMainTab(tab)}
      className={`px-6 py-3 font-bold text-base transition-colors border-t-2 border-l-2 border-r-2 ${
        mainTab === tab
          ? 'bg-background text-foreground border-primary border-b-2 border-b-background -mb-0.5'
          : 'bg-muted text-muted-foreground border-transparent hover:bg-accent'
      }`}
    >
      {label}
    </button>
  );

  const DataSubTabButton = ({ tab, label }: { tab: DataSubTab; label: string }) => (
    <button
      onClick={() => setDataSubTab(tab)}
      className={`px-5 py-2 font-bold text-sm rounded transition-all border-2 ${
        dataSubTab === tab
          ? 'bg-primary text-primary-foreground border-primary'
          : 'bg-background text-foreground border-primary hover:bg-accent'
      }`}
    >
      {label}
    </button>
  );

  const NoAgentMessage = ({ message }: { message: string }) => (
    <div className="p-10 text-center text-muted-foreground bg-muted">
      <h3 className="text-xl font-semibold mb-2">No Agent Selected</h3>
      <p>{message}</p>
    </div>
  );

  return (
    <div className="min-h-screen bg-background">
      <AgentSelector />

      {/* Main Tabs */}
      <div className="bg-muted border-b-2 border-primary">
        <TabButton tab="data" label="Data" />
        <TabButton tab="documents" label="Documents" />
        <TabButton tab="entities" label="Entities" />
        <TabButton tab="search" label="Search Debug" />
        <TabButton tab="stats" label="Stats & Operations" />
        <TabButton tab="think" label="Think" />
        <TabButton tab="add" label="Add Memory" />
      </div>

      {/* Tab Content - All tabs rendered but hidden to preserve state */}
      <div className="p-5">
        {/* Data Tab */}
        <div className={mainTab !== 'data' ? 'hidden' : ''}>
          {/* Data Sub Tabs */}
          <div className="bg-accent px-5 py-2.5 border-b-2 border-primary flex gap-2.5">
            <DataSubTabButton tab="world" label="World" />
            <DataSubTabButton tab="agent" label="Agent" />
            <DataSubTabButton tab="opinion" label="Opinions" />
          </div>

          {/* Data Sub Tab Content - Render all but hide inactive */}
          <div className="mt-5">
            {!currentAgent ? (
              <NoAgentMessage message="Please select an agent from the dropdown above to view data." />
            ) : (
              <div>
                <div className={dataSubTab !== 'world' ? 'hidden' : ''}>
                  <DataView factType="world" />
                </div>
                <div className={dataSubTab !== 'agent' ? 'hidden' : ''}>
                  <DataView factType="agent" />
                </div>
                <div className={dataSubTab !== 'opinion' ? 'hidden' : ''}>
                  <DataView factType="opinion" />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Documents Tab */}
        <div className={mainTab !== 'documents' ? 'hidden' : ''}>
          <h2 className="text-2xl font-bold mb-4">Documents</h2>
          {!currentAgent ? (
            <NoAgentMessage message="Please select an agent from the dropdown above to view documents." />
          ) : (
            <DocumentsView />
          )}
        </div>

        {/* Entities Tab */}
        <div className={mainTab !== 'entities' ? 'hidden' : ''}>
          <h2 className="text-2xl font-bold mb-4">Entities</h2>
          {!currentAgent ? (
            <NoAgentMessage message="Please select an agent from the dropdown above to view entities." />
          ) : (
            <EntitiesView />
          )}
        </div>

        {/* Search Debug Tab */}
        <div className={mainTab !== 'search' ? 'hidden' : ''}>
          <h2 className="text-2xl font-bold mb-4">Search Debug</h2>
          <SearchDebugView />
        </div>

        {/* Stats Tab */}
        <div className={mainTab !== 'stats' ? 'hidden' : ''}>
          <h2 className="text-2xl font-bold mb-4">Statistics & Operations</h2>
          <StatsView />
        </div>

        {/* Think Tab */}
        <div className={mainTab !== 'think' ? 'hidden' : ''}>
          <h2 className="text-2xl font-bold mb-4">Think - AI-Powered Answers</h2>
          {!currentAgent ? (
            <NoAgentMessage message="Please select an agent from the dropdown above to use the think feature." />
          ) : (
            <ThinkView />
          )}
        </div>

        {/* Add Memory Tab */}
        <div className={mainTab !== 'add' ? 'hidden' : ''}>
          <h2 className="text-2xl font-bold mb-4">Add Memory</h2>
          {!currentAgent ? (
            <NoAgentMessage message="Please select an agent from the dropdown above to add memories." />
          ) : (
            <AddMemoryView />
          )}
        </div>
      </div>
    </div>
  );
}
