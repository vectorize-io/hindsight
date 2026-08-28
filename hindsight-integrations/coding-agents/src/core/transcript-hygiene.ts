import type { TransportTurn } from "./chat";

export const TRANSCRIPT_HYGIENE_OFF = "off";
export const TRANSCRIPT_HYGIENE_SEMANTIC_BETA = "semantic-beta";
export const DEFAULT_TRANSCRIPT_HYGIENE = TRANSCRIPT_HYGIENE_OFF;
export const TRANSCRIPT_HYGIENE_MODES = [
  TRANSCRIPT_HYGIENE_OFF,
  TRANSCRIPT_HYGIENE_SEMANTIC_BETA,
] as const;

export type TranscriptHygieneMode = (typeof TRANSCRIPT_HYGIENE_MODES)[number];

const ACTION_ROLE = "action";
const USER_ROLE = "user";
const ASSISTANT_ROLE = "assistant";
const ACTION_GROUP_HEADER = "Action breadcrumbs";
const ACTION_GROUP_BULLET_PREFIX = "- ";
const ACTION_GROUP_SEPARATOR = "\n";
const EMPTY_ITEM_COUNT = 0;
const SINGLE_ITEM_COUNT = 1;
const FIRST_ITEM_INDEX = 0;
const COUNT_INCREMENT = 1;
const INLINE_WHITESPACE_RE = /\s+/g;

export interface TranscriptHygieneReceipt {
  mode: TranscriptHygieneMode;
  applied: boolean;
  inputTurns: number;
  outputTurns: number;
  userTurns: number;
  assistantTurns: number;
  actionTurns: number;
  actionGroups: number;
  groupedActionTurns: number;
}

export interface TranscriptHygieneResult {
  turns: TransportTurn[];
  receipt: TranscriptHygieneReceipt;
}

function countRoles(turns: TransportTurn[]): Pick<
  TranscriptHygieneReceipt,
  "userTurns" | "assistantTurns" | "actionTurns"
> {
  let userTurns = EMPTY_ITEM_COUNT;
  let assistantTurns = EMPTY_ITEM_COUNT;
  let actionTurns = EMPTY_ITEM_COUNT;

  for (const turn of turns) {
    if (turn.role === USER_ROLE) userTurns += COUNT_INCREMENT;
    else if (turn.role === ASSISTANT_ROLE) assistantTurns += COUNT_INCREMENT;
    else if (turn.role === ACTION_ROLE) actionTurns += COUNT_INCREMENT;
  }

  return { userTurns, assistantTurns, actionTurns };
}

function cleanActionContent(content: string): string {
  return content.replace(INLINE_WHITESPACE_RE, " ").trim();
}

function renderActionGroup(actions: TransportTurn[]): string {
  return [
    `${ACTION_GROUP_HEADER} (${actions.length} grouped):`,
    ...actions.map((action) => `${ACTION_GROUP_BULLET_PREFIX}${cleanActionContent(action.content)}`),
  ].join(ACTION_GROUP_SEPARATOR);
}

function makeReceipt(args: {
  mode: TranscriptHygieneMode;
  applied: boolean;
  inputTurns: TransportTurn[];
  outputTurns: TransportTurn[];
  actionGroups: number;
  groupedActionTurns: number;
}): TranscriptHygieneReceipt {
  const roleCounts = countRoles(args.inputTurns);
  return {
    mode: args.mode,
    applied: args.applied,
    inputTurns: args.inputTurns.length,
    outputTurns: args.outputTurns.length,
    ...roleCounts,
    actionGroups: args.actionGroups,
    groupedActionTurns: args.groupedActionTurns,
  };
}

/**
 * Optional beta hygiene pass for already-normalized transcript turns.
 *
 * Harness readers keep user and assistant text clean first. This pass is deliberately narrower:
 * it groups consecutive action breadcrumbs into one action turn, preserving the tool lineage while
 * reducing line pressure on the server's conversation extractor. It never rewrites user prose.
 */
export function applyTranscriptHygiene(
  mode: TranscriptHygieneMode,
  turns: TransportTurn[]
): TranscriptHygieneResult {
  if (mode !== TRANSCRIPT_HYGIENE_SEMANTIC_BETA) {
    return {
      turns,
      receipt: makeReceipt({
        mode,
        applied: false,
        inputTurns: turns,
        outputTurns: turns,
        actionGroups: EMPTY_ITEM_COUNT,
        groupedActionTurns: EMPTY_ITEM_COUNT,
      }),
    };
  }

  const output: TransportTurn[] = [];
  let pendingActions: TransportTurn[] = [];
  let actionGroups = EMPTY_ITEM_COUNT;
  let groupedActionTurns = EMPTY_ITEM_COUNT;

  const flushPendingActions = () => {
    if (pendingActions.length === EMPTY_ITEM_COUNT) return;
    if (pendingActions.length === SINGLE_ITEM_COUNT) {
      output.push(pendingActions[FIRST_ITEM_INDEX]);
      pendingActions = [];
      return;
    }

    actionGroups += COUNT_INCREMENT;
    groupedActionTurns += pendingActions.length;
    output.push({
      role: ACTION_ROLE,
      content: renderActionGroup(pendingActions),
      ...(pendingActions[FIRST_ITEM_INDEX].timestamp
        ? { timestamp: pendingActions[FIRST_ITEM_INDEX].timestamp }
        : {}),
    });
    pendingActions = [];
  };

  for (const turn of turns) {
    if (turn.role === ACTION_ROLE) {
      pendingActions.push(turn);
      continue;
    }
    flushPendingActions();
    output.push(turn);
  }
  flushPendingActions();

  return {
    turns: output,
    receipt: makeReceipt({
      mode,
      applied: true,
      inputTurns: turns,
      outputTurns: output,
      actionGroups,
      groupedActionTurns,
    }),
  };
}
