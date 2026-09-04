"use client";

import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react";

import { Button } from "@/components/ui/button";

/**
 * First/prev/next/last pager with a "x-y of n" range, shared by the paginated
 * tables (documents, entities, mental models, webhooks all render this shape
 * inline). Renders nothing for a single page, matching those views.
 */
interface PaginationProps {
  /** 1-based. */
  page: number;
  pageSize: number;
  total: number;
  /** Disables every control while a page is in flight. */
  disabled?: boolean;
  onPageChange: (page: number) => void;
}

export function Pagination({ page, pageSize, total, disabled, onPageChange }: PaginationProps) {
  const pageCount = Math.ceil(total / pageSize);
  if (pageCount <= 1) return null;

  const offset = (page - 1) * pageSize;

  return (
    <div className="flex items-center justify-between mt-3 pt-3 border-t">
      <div className="text-xs text-muted-foreground">
        {offset + 1}-{Math.min(offset + pageSize, total)} of {total}
      </div>
      <div className="flex items-center gap-1">
        <Button
          variant="outline"
          size="sm"
          onClick={() => onPageChange(1)}
          disabled={page === 1 || disabled}
          className="h-7 w-7 p-0"
        >
          <ChevronsLeft className="h-3 w-3" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => onPageChange(page - 1)}
          disabled={page === 1 || disabled}
          className="h-7 w-7 p-0"
        >
          <ChevronLeft className="h-3 w-3" />
        </Button>
        <span className="text-xs px-2">
          {page} / {pageCount}
        </span>
        <Button
          variant="outline"
          size="sm"
          onClick={() => onPageChange(page + 1)}
          disabled={page === pageCount || disabled}
          className="h-7 w-7 p-0"
        >
          <ChevronRight className="h-3 w-3" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          onClick={() => onPageChange(pageCount)}
          disabled={page === pageCount || disabled}
          className="h-7 w-7 p-0"
        >
          <ChevronsRight className="h-3 w-3" />
        </Button>
      </div>
    </div>
  );
}
