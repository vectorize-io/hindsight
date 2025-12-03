'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import { client } from '@/lib/api';
import { useBank } from '@/lib/bank-context';
import cytoscape from 'cytoscape';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Copy, Check, Calendar, ZoomIn, ZoomOut, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { MemoryDetailPanel } from './memory-detail-panel';

type FactType = 'world' | 'bank' | 'opinion';
type ViewMode = 'graph' | 'table' | 'timeline';

interface DataViewProps {
  factType: FactType;
}

export function DataView({ factType }: DataViewProps) {
  const { currentBank } = useBank();
  const [viewMode, setViewMode] = useState<ViewMode>('graph');
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [nodeLimit, setNodeLimit] = useState(50);
  const [layout, setLayout] = useState('circle');
  const [searchQuery, setSearchQuery] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedGraphNode, setSelectedGraphNode] = useState<any>(null);
  const [selectedTableMemory, setSelectedTableMemory] = useState<any>(null);
  const itemsPerPage = 100;
  const cyRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(text);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const loadData = async () => {
    if (!currentBank) return;

    setLoading(true);
    try {
      const graphData: any = await client.getGraph({
        bank_id: currentBank,
        type: factType,
      });
      console.log('Loaded graph data:', {
        total_units: graphData.total_units,
        nodes: graphData.nodes?.length,
        edges: graphData.edges?.length,
        table_rows: graphData.table_rows?.length,
      });
      setData(graphData);
    } catch (error) {
      console.error('Error loading data:', error);
      alert(`Error loading ${factType} data: ` + (error as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const renderGraph = () => {
    if (!data || !containerRef.current || !data.nodes || !data.edges) return;

    if (cyRef.current) {
      cyRef.current.destroy();
    }

    const limitedNodes = (data.nodes || []).slice(0, nodeLimit);
    const nodeIds = new Set(limitedNodes.map((n: any) => n.data.id));
    const limitedEdges = (data.edges || []).filter((e: any) =>
      nodeIds.has(e.data.source) && nodeIds.has(e.data.target)
    );

    const layouts: any = {
      circle: {
        name: 'circle',
        animate: false,
        radius: 300,
        spacingFactor: 1.5,
      },
      grid: {
        name: 'grid',
        animate: false,
        rows: Math.ceil(Math.sqrt(limitedNodes.length)),
        cols: Math.ceil(Math.sqrt(limitedNodes.length)),
        spacingFactor: 2,
      },
      cose: {
        name: 'cose',
        animate: false,
        nodeRepulsion: 15000,
        idealEdgeLength: 150,
        edgeElasticity: 100,
        nestingFactor: 1.2,
        gravity: 1,
        numIter: 1000,
        initialTemp: 200,
        coolingFactor: 0.95,
        minTemp: 1.0,
      },
    };

    cyRef.current = cytoscape({
      container: containerRef.current,
      elements: [
        ...limitedNodes.map((n: any) => ({ data: n.data })),
        ...limitedEdges.map((e: any) => ({ data: e.data })),
      ],
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)' as any,
            label: 'data(label)' as any,
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '10px',
            'font-weight': 'bold',
            'text-wrap': 'wrap',
            'text-max-width': '100px',
            width: 40,
            height: 40,
            'border-width': 2,
            'border-color': '#333',
          },
        },
        {
          selector: 'edge',
          style: {
            width: 1,
            'line-color': 'data(color)' as any,
            'line-style': 'data(lineStyle)' as any,
            'target-arrow-shape': 'triangle',
            'target-arrow-color': 'data(color)' as any,
            'curve-style': 'bezier',
            opacity: 0.6,
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': '#000',
          },
        },
      ] as any,
      layout: layouts[layout] || layouts.circle,
    });

    // Add click handler for nodes
    cyRef.current.on('tap', 'node', (evt: any) => {
      const nodeId = evt.target.id();
      // Find the corresponding table row data
      const nodeData = data.table_rows?.find((row: any) => row.id === nodeId);
      if (nodeData) {
        setSelectedGraphNode(nodeData);
      }
    });

    // Click on background to deselect
    cyRef.current.on('tap', (evt: any) => {
      if (evt.target === cyRef.current) {
        setSelectedGraphNode(null);
      }
    });
  };

  useEffect(() => {
    if (viewMode === 'graph' && data) {
      renderGraph();
    }
  }, [viewMode, data, nodeLimit, layout]);

  // Reset to first page when search query changes
  useEffect(() => {
    setCurrentPage(1);
  }, [searchQuery]);

  // Auto-load data when component mounts or factType/currentBank changes
  useEffect(() => {
    if (currentBank) {
      loadData();
    }
  }, [factType, currentBank]);

  return (
    <div>
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <div className="text-4xl mb-2">⏳</div>
            <div className="text-sm text-muted-foreground">Loading memories...</div>
          </div>
        </div>
      ) : data ? (
        <>
          <div className="flex items-center justify-between mb-6">
            <div className="text-sm text-muted-foreground">
              {data.total_units} total memories
            </div>
            <div className="flex items-center gap-2 bg-muted rounded-lg p-1">
              <button
                onClick={() => setViewMode('graph')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  viewMode === 'graph'
                    ? 'bg-background text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Graph View
              </button>
              <button
                onClick={() => setViewMode('table')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  viewMode === 'table'
                    ? 'bg-background text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Table View
              </button>
              <button
                onClick={() => setViewMode('timeline')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
                  viewMode === 'timeline'
                    ? 'bg-background text-foreground shadow-sm'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                Timeline View
              </button>
            </div>
          </div>

          {viewMode === 'graph' && (
            <div className="flex gap-4">
              <div className={`relative transition-all ${selectedGraphNode ? 'w-2/3' : 'w-full'}`}>
                <div className="p-4 bg-card border-b-2 border-primary flex gap-4 items-center flex-wrap">
                  <div className="flex items-center gap-2">
                    <label className="font-semibold text-card-foreground">Limit nodes:</label>
                    <Input
                      type="number"
                      value={nodeLimit}
                      onChange={(e) => setNodeLimit(parseInt(e.target.value))}
                      min="10"
                      max="1000"
                      step="10"
                      className="w-20"
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <label className="font-semibold text-card-foreground">Layout:</label>
                    <Select value={layout} onValueChange={setLayout}>
                      <SelectTrigger className="w-[180px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="circle">Circle (fast)</SelectItem>
                        <SelectItem value="grid">Grid (fast)</SelectItem>
                        <SelectItem value="cose">Force-directed (slow)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="text-sm text-muted-foreground ml-auto">
                    Click on a node to view details
                  </div>
                </div>
                <div ref={containerRef} className="w-full h-[800px] bg-background" />
                <div className="absolute top-20 left-5 bg-card p-4 border-2 border-primary rounded-lg shadow-lg max-w-[250px]">
                  <h3 className="font-bold mb-2 border-b-2 border-primary pb-1 text-card-foreground">Legend</h3>
                  <h4 className="font-bold mt-2 mb-1 text-sm text-card-foreground">Link Types:</h4>
                  <div className="flex items-center my-2">
                    <div className="w-8 h-0.5 mr-2.5 bg-cyan-500 border-t border-dashed border-cyan-500" />
                    <span className="text-sm"><strong>Temporal</strong></span>
                  </div>
                  <div className="flex items-center my-2">
                    <div className="w-8 h-0.5 mr-2.5 bg-pink-500" />
                    <span className="text-sm"><strong>Semantic</strong></span>
                  </div>
                  <div className="flex items-center my-2">
                    <div className="w-8 h-0.5 mr-2.5 bg-yellow-500" />
                    <span className="text-sm"><strong>Entity</strong></span>
                  </div>
                  <h4 className="font-bold mt-2 mb-1 text-sm">Nodes:</h4>
                  <div className="flex items-center my-2">
                    <div className="w-5 h-5 mr-2.5 bg-gray-300 border border-gray-500 rounded" />
                    <span className="text-sm">No entities</span>
                  </div>
                  <div className="flex items-center my-2">
                    <div className="w-5 h-5 mr-2.5 bg-blue-300 border border-gray-500 rounded" />
                    <span className="text-sm">1 entity</span>
                  </div>
                  <div className="flex items-center my-2">
                    <div className="w-5 h-5 mr-2.5 bg-blue-500 border border-gray-500 rounded" />
                    <span className="text-sm">2+ entities</span>
                  </div>
                </div>
              </div>

              {/* Memory Detail Panel for Graph View */}
              {selectedGraphNode && (
                <div className="w-1/3">
                  <MemoryDetailPanel
                    memory={selectedGraphNode}
                    onClose={() => setSelectedGraphNode(null)}
                  />
                </div>
              )}
            </div>
          )}

          {viewMode === 'table' && (
            <div>
              <div className="w-full">
                <div className="px-5 mb-4">
                  <Input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search memories (text, context, ID)..."
                    className="max-w-2xl"
                  />
                </div>
                <div className="px-5 pb-5">
                  {data.table_rows && data.table_rows.length > 0 ? (
                    (() => {
                      const filteredRows = data.table_rows.filter((row: any) => {
                        if (!searchQuery) return true;
                        const query = searchQuery.toLowerCase();
                        return (
                          row.text?.toLowerCase().includes(query) ||
                          row.context?.toLowerCase().includes(query) ||
                          row.id?.toLowerCase().includes(query)
                        );
                      });

                      const totalPages = Math.ceil(filteredRows.length / itemsPerPage);
                      const startIndex = (currentPage - 1) * itemsPerPage;
                      const endIndex = startIndex + itemsPerPage;
                      const paginatedRows = filteredRows.slice(startIndex, endIndex);

                      return (
                        <>
                          <div className="border rounded-lg overflow-hidden">
                            <Table>
                              <TableHeader>
                                <TableRow className="bg-muted/50">
                                  <TableHead className="w-[80px]">ID</TableHead>
                                  <TableHead>Text</TableHead>
                                  <TableHead className="w-[150px]">Context</TableHead>
                                  <TableHead className="w-[100px]">Occurred</TableHead>
                                  <TableHead className="w-[100px]">Mentioned</TableHead>
                                  <TableHead className="w-[60px]">Actions</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {paginatedRows.map((row: any, idx: number) => {
                                  const occurredDisplay = row.occurred_start
                                    ? new Date(row.occurred_start).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
                                    : null;
                                  const mentionedDisplay = row.mentioned_at
                                    ? new Date(row.mentioned_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
                                    : null;

                                  return (
                                    <TableRow
                                      key={row.id || idx}
                                      onClick={() => setSelectedTableMemory(row)}
                                      className={`cursor-pointer hover:bg-muted/50 ${
                                        selectedTableMemory?.id === row.id ? 'bg-primary/10' : ''
                                      }`}
                                    >
                                      <TableCell className="font-mono text-xs text-muted-foreground" title={row.id}>
                                        {row.id?.substring(0, 8)}...
                                      </TableCell>
                                      <TableCell>
                                        <div className="line-clamp-2 text-sm">{row.text}</div>
                                        {row.entities && (
                                          <div className="flex gap-1 mt-1 flex-wrap">
                                            {row.entities.split(', ').slice(0, 3).map((entity: string, i: number) => (
                                              <span key={i} className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-secondary-foreground">
                                                {entity}
                                              </span>
                                            ))}
                                            {row.entities.split(', ').length > 3 && (
                                              <span className="text-[10px] text-muted-foreground">
                                                +{row.entities.split(', ').length - 3}
                                              </span>
                                            )}
                                          </div>
                                        )}
                                      </TableCell>
                                      <TableCell className="text-xs text-muted-foreground truncate max-w-[150px]" title={row.context}>
                                        {row.context || '-'}
                                      </TableCell>
                                      <TableCell className="text-xs">
                                        {occurredDisplay ? (
                                          <span className="flex items-center gap-1">
                                            <Calendar className="h-3 w-3" />
                                            {occurredDisplay}
                                          </span>
                                        ) : '-'}
                                      </TableCell>
                                      <TableCell className="text-xs">
                                        {mentionedDisplay ? (
                                          <span className="flex items-center gap-1">
                                            <Calendar className="h-3 w-3" />
                                            {mentionedDisplay}
                                          </span>
                                        ) : '-'}
                                      </TableCell>
                                      <TableCell>
                                        <Button
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            copyToClipboard(row.id);
                                          }}
                                          size="sm"
                                          variant="ghost"
                                          className="h-7 w-7 p-0"
                                          title="Copy ID"
                                        >
                                          {copiedId === row.id ? (
                                            <Check className="h-3 w-3 text-green-600" />
                                          ) : (
                                            <Copy className="h-3 w-3" />
                                          )}
                                        </Button>
                                      </TableCell>
                                    </TableRow>
                                  );
                                })}
                              </TableBody>
                            </Table>
                          </div>

                          {/* Pagination Controls */}
                          {totalPages > 1 && (
                            <div className="flex items-center justify-between mt-4 pt-4 border-t">
                              <div className="text-sm text-muted-foreground">
                                Showing {startIndex + 1} to {Math.min(endIndex, filteredRows.length)} of {filteredRows.length}
                              </div>
                              <div className="flex items-center gap-1">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => setCurrentPage(1)}
                                  disabled={currentPage === 1}
                                  className="h-8 w-8 p-0"
                                >
                                  <ChevronsLeft className="h-4 w-4" />
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                                  disabled={currentPage === 1}
                                  className="h-8 w-8 p-0"
                                >
                                  <ChevronLeft className="h-4 w-4" />
                                </Button>
                                <span className="text-sm px-3">
                                  {currentPage} / {totalPages}
                                </span>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                                  disabled={currentPage === totalPages}
                                  className="h-8 w-8 p-0"
                                >
                                  <ChevronRight className="h-4 w-4" />
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => setCurrentPage(totalPages)}
                                  disabled={currentPage === totalPages}
                                  className="h-8 w-8 p-0"
                                >
                                  <ChevronsRight className="h-4 w-4" />
                                </Button>
                              </div>
                            </div>
                          )}
                        </>
                      );
                    })()
                  ) : (
                    <div className="text-center py-12 text-muted-foreground">
                      {data.table_rows ? 'No memories match your search' : 'No memories found'}
                    </div>
                  )}
                </div>
              </div>

              {/* Memory Detail Panel for Table View - Fixed on Right */}
              {selectedTableMemory && (
                <div className="fixed right-0 top-0 h-screen w-96 bg-background border-l border-border shadow-lg z-50 overflow-y-auto p-4">
                  <MemoryDetailPanel
                    memory={selectedTableMemory}
                    onClose={() => setSelectedTableMemory(null)}
                  />
                </div>
              )}
            </div>
          )}

          {viewMode === 'timeline' && (
            <TimelineView data={data} />
          )}
        </>
      ) : (
        <div className="flex items-center justify-center py-20">
          <div className="text-center">
            <div className="text-4xl mb-2">📊</div>
            <div className="text-sm text-muted-foreground">No data available</div>
          </div>
        </div>
      )}
    </div>
  );
}

// Timeline View Component - Custom compact timeline with zoom and navigation
type Granularity = 'year' | 'month' | 'week' | 'day';

function TimelineView({ data }: { data: any }) {
  const [selectedItem, setSelectedItem] = useState<any>(null);
  const [granularity, setGranularity] = useState<Granularity>('month');
  const [currentIndex, setCurrentIndex] = useState(0);
  const timelineRef = useRef<HTMLDivElement>(null);

  // Filter and sort items that have occurred_start dates
  const { sortedItems, itemsWithoutDates } = useMemo(() => {
    if (!data?.table_rows) return { sortedItems: [], itemsWithoutDates: [] };

    const withDates = data.table_rows
      .filter((row: any) => row.occurred_start)
      .sort((a: any, b: any) => {
        const dateA = new Date(a.occurred_start).getTime();
        const dateB = new Date(b.occurred_start).getTime();
        return dateA - dateB;
      });

    const withoutDates = data.table_rows.filter((row: any) => !row.occurred_start);

    // Debug logging
    console.log('Timeline data:', {
      total: data.table_rows.length,
      withDates: withDates.length,
      withoutDates: withoutDates.length,
      sampleWithDate: withDates[0],
      sampleWithoutDate: withoutDates[0]
    });

    return { sortedItems: withDates, itemsWithoutDates: withoutDates };
  }, [data]);

  // Group items by granularity
  const timelineGroups = useMemo(() => {
    if (sortedItems.length === 0) return [];

    const getGroupKey = (date: Date): string => {
      const year = date.getFullYear();
      const month = date.getMonth();
      const day = date.getDate();

      switch (granularity) {
        case 'year':
          return `${year}`;
        case 'month':
          return `${year}-${String(month + 1).padStart(2, '0')}`;
        case 'week':
          const startOfWeek = new Date(date);
          startOfWeek.setDate(day - date.getDay());
          return `${startOfWeek.getFullYear()}-W${String(Math.ceil((startOfWeek.getDate()) / 7)).padStart(2, '0')}-${String(startOfWeek.getMonth() + 1).padStart(2, '0')}-${String(startOfWeek.getDate()).padStart(2, '0')}`;
        case 'day':
          return `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
      }
    };

    const getGroupLabel = (key: string, date: Date): string => {
      switch (granularity) {
        case 'year':
          return key;
        case 'month':
          return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
        case 'week':
          const endOfWeek = new Date(date);
          endOfWeek.setDate(date.getDate() + 6);
          return `${date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} - ${endOfWeek.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`;
        case 'day':
          return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
      }
    };

    const groups: { [key: string]: { items: any[]; date: Date } } = {};
    sortedItems.forEach((row: any) => {
      const date = new Date(row.occurred_start);
      const key = getGroupKey(date);
      if (!groups[key]) {
        // For week, parse the start date from key
        let groupDate = date;
        if (granularity === 'week') {
          const parts = key.split('-');
          groupDate = new Date(parseInt(parts[0]), parseInt(parts[2]) - 1, parseInt(parts[3]));
        }
        groups[key] = { items: [], date: groupDate };
      }
      groups[key].items.push(row);
    });

    return Object.entries(groups)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, { items, date }]) => ({
        key,
        label: getGroupLabel(key, date),
        items,
        date,
      }));
  }, [sortedItems, granularity]);

  // Get date range info
  const dateRange = useMemo(() => {
    if (sortedItems.length === 0) return null;
    const first = new Date(sortedItems[0].occurred_start);
    const last = new Date(sortedItems[sortedItems.length - 1].occurred_start);
    return { first, last };
  }, [sortedItems]);

  // Navigation
  const scrollToGroup = (index: number) => {
    const clampedIndex = Math.max(0, Math.min(index, timelineGroups.length - 1));
    setCurrentIndex(clampedIndex);
    const element = document.getElementById(`timeline-group-${clampedIndex}`);
    element?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const zoomIn = () => {
    const levels: Granularity[] = ['year', 'month', 'week', 'day'];
    const currentIdx = levels.indexOf(granularity);
    if (currentIdx < levels.length - 1) {
      setGranularity(levels[currentIdx + 1]);
    }
  };

  const zoomOut = () => {
    const levels: Granularity[] = ['year', 'month', 'week', 'day'];
    const currentIdx = levels.indexOf(granularity);
    if (currentIdx > 0) {
      setGranularity(levels[currentIdx - 1]);
    }
  };

  if (sortedItems.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Calendar className="w-12 h-12 text-muted-foreground mb-3" />
        <div className="text-base font-medium text-foreground mb-1">No Timeline Data</div>
        <div className="text-xs text-muted-foreground text-center max-w-md">
          No memories have occurred_at dates.
          {itemsWithoutDates.length > 0 && (
            <span className="block mt-1">
              {itemsWithoutDates.length} memories without dates in Table View.
            </span>
          )}
        </div>
      </div>
    );
  }

  const formatDateTime = (dateStr: string) => {
    const date = new Date(dateStr);
    const dateFormatted = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    const timeFormatted = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
    return { date: dateFormatted, time: timeFormatted };
  };

  const granularityLabels: Record<Granularity, string> = {
    year: 'Year',
    month: 'Month',
    week: 'Week',
    day: 'Day',
  };

  return (
    <div className="flex gap-3 px-4">
      {/* Timeline */}
      <div className={`transition-all ${selectedItem ? 'w-2/3' : 'w-full'}`}>
        {/* Controls */}
        <div className="flex items-center justify-between mb-3 gap-4">
          <div className="text-xs text-muted-foreground">
            {sortedItems.length} memories
            {itemsWithoutDates.length > 0 && ` · ${itemsWithoutDates.length} without dates`}
            {dateRange && (
              <span className="ml-2 text-foreground">
                ({dateRange.first.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })} → {dateRange.last.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })})
              </span>
            )}
          </div>

          <div className="flex items-center gap-1">
            {/* Zoom controls */}
            <div className="flex items-center border border-border rounded mr-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={zoomOut}
                disabled={granularity === 'year'}
                className="h-7 w-7 p-0"
                title="Zoom out"
              >
                <ZoomOut className="h-3 w-3" />
              </Button>
              <span className="text-[10px] px-2 min-w-[50px] text-center border-x border-border">
                {granularityLabels[granularity]}
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={zoomIn}
                disabled={granularity === 'day'}
                className="h-7 w-7 p-0"
                title="Zoom in"
              >
                <ZoomIn className="h-3 w-3" />
              </Button>
            </div>

            {/* Navigation controls */}
            <div className="flex items-center border border-border rounded">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => scrollToGroup(0)}
                disabled={timelineGroups.length <= 1}
                className="h-7 w-7 p-0"
                title="First"
              >
                <ChevronsLeft className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => scrollToGroup(currentIndex - 1)}
                disabled={currentIndex === 0}
                className="h-7 w-7 p-0"
                title="Previous"
              >
                <ChevronLeft className="h-3 w-3" />
              </Button>
              <span className="text-[10px] px-2 min-w-[60px] text-center border-x border-border">
                {currentIndex + 1} / {timelineGroups.length}
              </span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => scrollToGroup(currentIndex + 1)}
                disabled={currentIndex >= timelineGroups.length - 1}
                className="h-7 w-7 p-0"
                title="Next"
              >
                <ChevronRight className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => scrollToGroup(timelineGroups.length - 1)}
                disabled={timelineGroups.length <= 1}
                className="h-7 w-7 p-0"
                title="Last"
              >
                <ChevronsRight className="h-3 w-3" />
              </Button>
            </div>
          </div>
        </div>

        <div ref={timelineRef} className="relative max-h-[550px] overflow-y-auto pr-2">
          {/* Vertical line */}
          <div className="absolute left-[60px] top-0 bottom-0 w-0.5 bg-border" />

          {timelineGroups.map((group, groupIdx) => (
            <div key={group.key} id={`timeline-group-${groupIdx}`} className="mb-4">
              {/* Group header */}
              <div
                className="flex items-center mb-2 cursor-pointer hover:opacity-80"
                onClick={() => setCurrentIndex(groupIdx)}
              >
                <div className="w-[60px] text-right pr-3">
                  <span className="text-xs font-semibold text-primary">{group.label}</span>
                </div>
                <div className="w-2 h-2 rounded-full bg-primary z-10" />
                <span className="ml-2 text-[10px] text-muted-foreground">
                  {group.items.length} {group.items.length === 1 ? 'item' : 'items'}
                </span>
              </div>

              {/* Items in this month */}
              <div className="space-y-1">
                {group.items.map((item: any, idx: number) => (
                  <div
                    key={item.id || idx}
                    onClick={() => setSelectedItem(item)}
                    className={`flex items-start cursor-pointer group ${
                      selectedItem?.id === item.id ? 'opacity-100' : 'hover:opacity-80'
                    }`}
                  >
                    {/* Date & Time */}
                    <div className="w-[60px] text-right pr-3 pt-1 flex-shrink-0">
                      <div className="text-[10px] text-muted-foreground">
                        {formatDateTime(item.occurred_start).date}
                      </div>
                      <div className="text-[9px] text-muted-foreground/70">
                        {formatDateTime(item.occurred_start).time}
                      </div>
                    </div>

                    {/* Connector dot */}
                    <div className="flex-shrink-0 pt-2">
                      <div className={`w-1.5 h-1.5 rounded-full z-10 ${
                        selectedItem?.id === item.id ? 'bg-primary' : 'bg-muted-foreground/50 group-hover:bg-primary'
                      }`} />
                    </div>

                    {/* Card */}
                    <div className={`ml-3 flex-1 p-2 rounded border transition-colors ${
                      selectedItem?.id === item.id
                        ? 'bg-primary/10 border-primary'
                        : 'bg-card border-border hover:border-primary/50'
                    }`}>
                      <p className="text-xs text-foreground line-clamp-2 leading-relaxed">
                        {item.text}
                      </p>
                      {item.context && (
                        <p className="text-[10px] text-muted-foreground mt-1 truncate">
                          {item.context}
                        </p>
                      )}
                      {item.entities && (
                        <div className="flex gap-1 mt-1 flex-wrap">
                          {item.entities.split(', ').slice(0, 3).map((entity: string, i: number) => (
                            <span key={i} className="text-[9px] px-1 py-0.5 rounded bg-secondary text-secondary-foreground">
                              {entity}
                            </span>
                          ))}
                          {item.entities.split(', ').length > 3 && (
                            <span className="text-[9px] text-muted-foreground">
                              +{item.entities.split(', ').length - 3}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Detail Panel */}
      {selectedItem && (
        <div className="w-1/3">
          <MemoryDetailPanel
            memory={selectedItem}
            onClose={() => setSelectedItem(null)}
            compact
          />
        </div>
      )}
    </div>
  );
}
