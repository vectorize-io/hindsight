'use client';

import { useState } from 'react';
import { useBank } from '@/lib/bank-context';
import { Search, Sparkles, Database, FileText, Users, ChevronLeft, ChevronRight, Box } from 'lucide-react';
import { cn } from '@/lib/utils';
import Link from 'next/link';

type NavItem = 'recall' | 'reflect' | 'data' | 'documents' | 'entities' | 'profile';

interface SidebarProps {
  currentTab: NavItem;
  onTabChange: (tab: NavItem) => void;
}

export function Sidebar({ currentTab, onTabChange }: SidebarProps) {
  const { currentBank } = useBank();
  const [isCollapsed, setIsCollapsed] = useState(true);

  if (!currentBank) {
    return null;
  }

  const navItems = [
    { id: 'data' as NavItem, label: 'Memories', icon: Database },
    { id: 'recall' as NavItem, label: 'Recall', icon: Search },
    { id: 'reflect' as NavItem, label: 'Reflect', icon: Sparkles },
    { id: 'documents' as NavItem, label: 'Documents', icon: FileText },
    { id: 'entities' as NavItem, label: 'Entities', icon: Users },
    { id: 'profile' as NavItem, label: 'Memory Bank', icon: Box },
  ];

  return (
    <aside className={cn(
      'bg-card border-r border-border flex flex-col transition-all duration-300',
      isCollapsed ? 'w-16' : 'w-64'
    )}>
      <nav className="flex-1 p-3 pt-4">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentTab === item.id;
            const href = `/banks/${currentBank}?view=${item.id}`;

            return (
              <li key={item.id}>
                <Link
                  href={href}
                  onClick={(e) => {
                    // For left-click, prevent default and use the callback
                    // This allows the parent to handle navigation without full page reload
                    if (e.button === 0 && !e.ctrlKey && !e.metaKey) {
                      e.preventDefault();
                      onTabChange(item.id);
                    }
                    // Middle-click or Ctrl/Cmd+click will naturally open in new tab
                  }}
                  className={cn(
                    'w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all',
                    isActive
                      ? 'bg-primary-gradient text-white shadow-sm'
                      : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground',
                    isCollapsed && 'justify-center px-0'
                  )}
                  title={isCollapsed ? item.label : undefined}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  {!isCollapsed && <span>{item.label}</span>}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Collapse/Expand button at bottom */}
      <div className="p-3 border-t border-border">
        <button
          onClick={() => setIsCollapsed(!isCollapsed)}
          className={cn(
            'w-full flex items-center gap-3 px-4 py-2 rounded-lg text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors',
            isCollapsed && 'justify-center px-0'
          )}
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {isCollapsed ? (
            <ChevronRight className="w-5 h-5" />
          ) : (
            <>
              <ChevronLeft className="w-5 h-5" />
              <span>Collapse</span>
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
