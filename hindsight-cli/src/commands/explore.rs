use crate::api::{ApiClient, RecallRequest, ReflectRequest};
use anyhow::Result;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use hindsight_client::types::{BankListItem, RecallResult, EntityListItem, Budget};
use serde_json::{Map, Value};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

// Brand gradient colors: #0074d9 -> #009296
const BRAND_START: Color = Color::Rgb(0, 116, 217);  // #0074d9
const BRAND_END: Color = Color::Rgb(0, 146, 150);    // #009296
const BRAND_MID: Color = Color::Rgb(0, 131, 183);    // Midpoint

/// Main view types (like k9s contexts)
#[derive(Debug, Clone, PartialEq)]
enum View {
    Banks,
    Memories(String),  // bank_id
    Entities(String),  // bank_id
    Documents(String), // bank_id
    Query(String),     // bank_id - combines recall and reflect
}

impl View {
    fn title(&self) -> &str {
        match self {
            View::Banks => "Banks",
            View::Memories(_) => "Memories",
            View::Entities(_) => "Entities",
            View::Documents(_) => "Documents",
            View::Query(_) => "Query",
        }
    }

    fn bank_id(&self) -> Option<&str> {
        match self {
            View::Banks => None,
            View::Memories(id) | View::Entities(id) | View::Documents(id) | View::Query(id) => Some(id),
        }
    }
}

/// Query mode for the Query view
#[derive(Debug, Clone, PartialEq)]
enum QueryMode {
    Recall,
    Reflect,
}

/// Input mode for recall/reflect queries
#[derive(Debug, Clone, PartialEq)]
enum InputMode {
    Normal,
    Query,
}

/// Query result from background thread
enum QueryResult {
    Recall(Result<Vec<RecallResult>, String>),
    Reflect(Result<String, String>),
}

/// Application state
struct App {
    client: ApiClient,
    view: View,
    view_history: Vec<View>,

    // List states
    banks: Vec<BankListItem>,
    banks_state: ListState,
    selected_bank_id: Option<String>,

    memories: Vec<Map<String, Value>>,
    memories_state: ListState,
    viewing_memory: Option<Map<String, Value>>,
    memories_limit: i64,
    memories_offset: i64,
    horizontal_scroll: usize,

    entities: Vec<EntityListItem>,
    entities_state: ListState,
    viewing_entity: Option<EntityListItem>,

    documents: Vec<Map<String, Value>>,
    documents_state: ListState,
    viewing_document: Option<Map<String, Value>>,

    // Query state (unified recall/reflect)
    query_mode: QueryMode,
    query_text: String,
    query_budget: Budget,
    query_max_tokens: i64,
    query_results: Vec<RecallResult>,
    query_results_state: ListState,
    query_response: String,
    viewing_recall_result: Option<RecallResult>,

    // Input mode
    input_mode: InputMode,

    // Status messages
    status_message: String,
    error_message: String,

    // Help visibility
    show_help: bool,

    // Loading state
    loading: bool,

    // Auto-refresh
    auto_refresh_enabled: bool,
    last_refresh: Instant,
    refresh_interval: Duration,

    // Background query receiver
    query_receiver: Option<Receiver<QueryResult>>,
}

impl App {
    fn new(client: ApiClient) -> Self {
        let mut app = Self {
            client,
            view: View::Banks,
            view_history: Vec::new(),

            banks: Vec::new(),
            banks_state: ListState::default(),
            selected_bank_id: None,

            memories: Vec::new(),
            memories_state: ListState::default(),
            viewing_memory: None,
            memories_limit: 500,
            memories_offset: 0,
            horizontal_scroll: 0,

            entities: Vec::new(),
            entities_state: ListState::default(),
            viewing_entity: None,

            documents: Vec::new(),
            documents_state: ListState::default(),
            viewing_document: None,

            query_mode: QueryMode::Recall,
            query_text: String::new(),
            query_budget: Budget::Mid,
            query_max_tokens: 4096,
            query_results: Vec::new(),
            query_results_state: ListState::default(),
            query_response: String::new(),
            viewing_recall_result: None,

            input_mode: InputMode::Normal,
            status_message: String::from("Select a bank to start. Press ? for help"),
            error_message: String::new(),
            show_help: false,
            loading: false,

            auto_refresh_enabled: true,
            last_refresh: Instant::now(),
            refresh_interval: Duration::from_secs(5),

            query_receiver: None,
        };

        // Select first item by default
        app.banks_state.select(Some(0));
        app.memories_state.select(Some(0));
        app.entities_state.select(Some(0));
        app.documents_state.select(Some(0));
        app.query_results_state.select(Some(0));

        app
    }

    fn refresh(&mut self) -> Result<()> {
        self.loading = true;
        self.error_message.clear();

        let result = match self.view.clone() {
            View::Banks => self.load_banks(),
            View::Memories(bank_id) => self.load_memories(&bank_id),
            View::Entities(bank_id) => self.load_entities(&bank_id),
            View::Documents(bank_id) => self.load_documents(&bank_id),
            View::Query(_) => Ok(()), // Query is query-driven
        };

        self.loading = false;

        if let Err(e) = result {
            self.error_message = format!("Error: {}", e);
        }

        Ok(())
    }

    fn toggle_auto_refresh(&mut self) {
        self.auto_refresh_enabled = !self.auto_refresh_enabled;
        if self.auto_refresh_enabled {
            self.status_message = "Auto-refresh enabled (5s)".to_string();
            self.last_refresh = Instant::now();
        } else {
            self.status_message = "Auto-refresh disabled".to_string();
        }
    }

    fn should_refresh(&self) -> bool {
        self.auto_refresh_enabled && self.last_refresh.elapsed() >= self.refresh_interval
    }

    fn do_auto_refresh(&mut self) -> Result<()> {
        if self.should_refresh() {
            self.last_refresh = Instant::now();
            self.refresh()?;
        }
        Ok(())
    }

    fn load_banks(&mut self) -> Result<()> {
        self.banks = self.client.list_agents(false)?;

        if !self.banks.is_empty() && self.banks_state.selected().is_none() {
            self.banks_state.select(Some(0));
        }

        self.status_message = format!("Loaded {} banks", self.banks.len());
        Ok(())
    }

    fn load_memories(&mut self, bank_id: &str) -> Result<()> {
        let response = self.client.list_memories(
            bank_id,
            None,
            None,
            Some(self.memories_limit),
            Some(self.memories_offset),
            false
        )?;
        self.memories = response.items;

        if !self.memories.is_empty() && self.memories_state.selected().is_none() {
            self.memories_state.select(Some(0));
        }

        self.status_message = format!("Loaded {} memories (limit: {}, offset: {})",
            self.memories.len(), self.memories_limit, self.memories_offset);
        Ok(())
    }

    fn load_more_memories(&mut self) -> Result<()> {
        if let View::Memories(bank_id) = &self.view {
            let bank_id = bank_id.clone();
            self.memories_offset += self.memories_limit;
            self.load_memories(&bank_id)?;
        }
        Ok(())
    }

    fn load_prev_memories(&mut self) -> Result<()> {
        if let View::Memories(bank_id) = &self.view {
            let bank_id = bank_id.clone();
            self.memories_offset = (self.memories_offset - self.memories_limit).max(0);
            self.load_memories(&bank_id)?;
        }
        Ok(())
    }

    fn load_entities(&mut self, bank_id: &str) -> Result<()> {
        let response = self.client.list_entities(bank_id, Some(100), false)?;
        self.entities = response.items;

        if !self.entities.is_empty() && self.entities_state.selected().is_none() {
            self.entities_state.select(Some(0));
        }

        self.status_message = format!("Loaded {} entities", self.entities.len());
        Ok(())
    }

    fn load_documents(&mut self, bank_id: &str) -> Result<()> {
        let response = self.client.list_documents(bank_id, None, Some(100), Some(0), false)?;
        self.documents = response.items;

        if !self.documents.is_empty() && self.documents_state.selected().is_none() {
            self.documents_state.select(Some(0));
        }

        self.status_message = format!("Loaded {} documents", self.documents.len());
        Ok(())
    }

    fn execute_query(&mut self) {
        if let View::Query(bank_id) = &self.view {
            if self.query_text.is_empty() {
                self.error_message = "Query cannot be empty".to_string();
                return;
            }

            self.loading = true;
            self.error_message.clear();
            self.input_mode = InputMode::Normal;

            // Create channel for receiving results
            let (tx, rx) = mpsc::channel();
            self.query_receiver = Some(rx);

            // Clone data for the thread
            let client = self.client.clone();
            let bank_id = bank_id.clone();
            let query_mode = self.query_mode.clone();
            let query_text = self.query_text.clone();
            let query_budget = self.query_budget.clone();
            let query_max_tokens = self.query_max_tokens;

            // Spawn background thread
            thread::spawn(move || {
                match query_mode {
                    QueryMode::Recall => {
                        let request = RecallRequest {
                            query: query_text,
                            types: None,
                            budget: Some(query_budget),
                            max_tokens: query_max_tokens,
                            trace: false,
                            query_timestamp: None,
                            include: None,
                        };

                        let result = client.recall(&bank_id, &request, false)
                            .map(|r| r.results)
                            .map_err(|e| e.to_string());

                        let _ = tx.send(QueryResult::Recall(result));
                    }
                    QueryMode::Reflect => {
                        let request = ReflectRequest {
                            query: query_text,
                            budget: Some(query_budget),
                            context: None,
                            include: None,
                        };

                        let result = client.reflect(&bank_id, &request, false)
                            .map(|r| r.text)
                            .map_err(|e| e.to_string());

                        let _ = tx.send(QueryResult::Reflect(result));
                    }
                }
            });
        }
    }

    fn check_query_result(&mut self) {
        if let Some(receiver) = &self.query_receiver {
            match receiver.try_recv() {
                Ok(QueryResult::Recall(Ok(results))) => {
                    self.query_results = results;
                    if !self.query_results.is_empty() {
                        self.query_results_state.select(Some(0));
                    }
                    self.loading = false;
                    self.status_message = format!("Found {} results", self.query_results.len());
                    self.query_receiver = None;
                }
                Ok(QueryResult::Recall(Err(e))) => {
                    self.error_message = format!("Recall failed: {}", e);
                    self.loading = false;
                    self.query_receiver = None;
                }
                Ok(QueryResult::Reflect(Ok(text))) => {
                    self.query_response = text;
                    self.loading = false;
                    self.status_message = "Reflection complete".to_string();
                    self.query_receiver = None;
                }
                Ok(QueryResult::Reflect(Err(e))) => {
                    self.error_message = format!("Reflect failed: {}", e);
                    self.loading = false;
                    self.query_receiver = None;
                }
                Err(TryRecvError::Empty) => {
                    // Still waiting for result
                }
                Err(TryRecvError::Disconnected) => {
                    self.error_message = "Query thread disconnected".to_string();
                    self.loading = false;
                    self.query_receiver = None;
                }
            }
        }
    }

    fn toggle_query_mode(&mut self) {
        self.query_mode = match self.query_mode {
            QueryMode::Recall => QueryMode::Reflect,
            QueryMode::Reflect => QueryMode::Recall,
        };
        self.status_message = format!("Switched to {} mode", match self.query_mode {
            QueryMode::Recall => "Recall",
            QueryMode::Reflect => "Reflect",
        });
    }

    fn cycle_budget(&mut self) {
        self.query_budget = match self.query_budget {
            Budget::Low => Budget::Mid,
            Budget::Mid => Budget::High,
            Budget::High => Budget::Low,
        };
        self.status_message = format!("Budget: {:?}", self.query_budget);
    }

    fn adjust_max_tokens(&mut self, increase: bool) {
        if increase {
            self.query_max_tokens = (self.query_max_tokens + 1024).min(16384);
        } else {
            self.query_max_tokens = (self.query_max_tokens - 1024).max(512);
        }
        self.status_message = format!("Max tokens: {}", self.query_max_tokens);
    }

    fn scroll_left(&mut self) {
        self.horizontal_scroll = self.horizontal_scroll.saturating_sub(10);
    }

    fn scroll_right(&mut self) {
        self.horizontal_scroll = (self.horizontal_scroll + 10).min(200);
    }

    fn reset_horizontal_scroll(&mut self) {
        self.horizontal_scroll = 0;
    }

    fn next_item(&mut self) {
        match &self.view {
            View::Banks => {
                let i = match self.banks_state.selected() {
                    Some(i) => {
                        if i >= self.banks.len().saturating_sub(1) {
                            0
                        } else {
                            i + 1
                        }
                    }
                    None => 0,
                };
                self.banks_state.select(Some(i));
            }
            View::Memories(_) => {
                let i = match self.memories_state.selected() {
                    Some(i) => {
                        if i >= self.memories.len().saturating_sub(1) {
                            0
                        } else {
                            i + 1
                        }
                    }
                    None => 0,
                };
                self.memories_state.select(Some(i));
            }
            View::Entities(_) => {
                let i = match self.entities_state.selected() {
                    Some(i) => {
                        if i >= self.entities.len().saturating_sub(1) {
                            0
                        } else {
                            i + 1
                        }
                    }
                    None => 0,
                };
                self.entities_state.select(Some(i));
            }
            View::Documents(_) => {
                let i = match self.documents_state.selected() {
                    Some(i) => {
                        if i >= self.documents.len().saturating_sub(1) {
                            0
                        } else {
                            i + 1
                        }
                    }
                    None => 0,
                };
                self.documents_state.select(Some(i));
            }
            View::Query(_) => {
                if self.query_mode == QueryMode::Recall {
                    let i = match self.query_results_state.selected() {
                        Some(i) => {
                            if i >= self.query_results.len().saturating_sub(1) {
                                0
                            } else {
                                i + 1
                            }
                        }
                        None => 0,
                    };
                    self.query_results_state.select(Some(i));
                }
            }
        }
    }

    fn previous_item(&mut self) {
        match &self.view {
            View::Banks => {
                let i = match self.banks_state.selected() {
                    Some(i) => {
                        if i == 0 {
                            self.banks.len().saturating_sub(1)
                        } else {
                            i - 1
                        }
                    }
                    None => 0,
                };
                self.banks_state.select(Some(i));
            }
            View::Memories(_) => {
                let i = match self.memories_state.selected() {
                    Some(i) => {
                        if i == 0 {
                            self.memories.len().saturating_sub(1)
                        } else {
                            i - 1
                        }
                    }
                    None => 0,
                };
                self.memories_state.select(Some(i));
            }
            View::Entities(_) => {
                let i = match self.entities_state.selected() {
                    Some(i) => {
                        if i == 0 {
                            self.entities.len().saturating_sub(1)
                        } else {
                            i - 1
                        }
                    }
                    None => 0,
                };
                self.entities_state.select(Some(i));
            }
            View::Documents(_) => {
                let i = match self.documents_state.selected() {
                    Some(i) => {
                        if i == 0 {
                            self.documents.len().saturating_sub(1)
                        } else {
                            i - 1
                        }
                    }
                    None => 0,
                };
                self.documents_state.select(Some(i));
            }
            View::Query(_) => {
                if self.query_mode == QueryMode::Recall {
                    let i = match self.query_results_state.selected() {
                        Some(i) => {
                            if i == 0 {
                                self.query_results.len().saturating_sub(1)
                            } else {
                                i - 1
                            }
                        }
                        None => 0,
                    };
                    self.query_results_state.select(Some(i));
                }
            }
        }
    }

    fn enter_view(&mut self) -> Result<()> {
        match &self.view {
            View::Banks => {
                if let Some(i) = self.banks_state.selected() {
                    if let Some(bank) = self.banks.get(i) {
                        let bank_id = bank.bank_id.clone();
                        self.selected_bank_id = Some(bank_id.clone());
                        self.view_history.push(self.view.clone());
                        self.view = View::Memories(bank_id.clone());
                        self.load_memories(&bank_id)?;
                    }
                }
            }
            View::Memories(_) => {
                if let Some(i) = self.memories_state.selected() {
                    if let Some(memory) = self.memories.get(i) {
                        self.viewing_memory = Some(memory.clone());
                        self.status_message = "Viewing memory details (Esc to close)".to_string();
                    }
                }
            }
            View::Entities(_) => {
                if let Some(i) = self.entities_state.selected() {
                    if let Some(entity) = self.entities.get(i).cloned() {
                        self.viewing_entity = Some(entity);
                        self.status_message = "Viewing entity details (Esc to close)".to_string();
                    }
                }
            }
            View::Documents(bank_id) => {
                if let Some(i) = self.documents_state.selected() {
                    if let Some(doc) = self.documents.get(i) {
                        // Fetch full document content
                        let doc_id = doc.get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");

                        if !doc_id.is_empty() {
                            match self.client.get_document(bank_id, doc_id, false) {
                                Ok(full_doc) => {
                                    // Convert to Map for display
                                    let doc_map: Map<String, Value> = serde_json::from_value(
                                        serde_json::to_value(full_doc)?
                                    )?;
                                    self.viewing_document = Some(doc_map);
                                    self.status_message = format!("Viewing document: {}", doc_id);
                                }
                                Err(e) => {
                                    self.error_message = format!("Failed to load document: {}", e);
                                }
                            }
                        }
                    }
                }
            }
            View::Query(_) => {
                // View recall result details if in recall mode
                if self.query_mode == QueryMode::Recall {
                    if let Some(i) = self.query_results_state.selected() {
                        if let Some(result) = self.query_results.get(i).cloned() {
                            self.viewing_recall_result = Some(result);
                            self.status_message = "Viewing recall result (Esc to close)".to_string();
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn go_back(&mut self) {
        // If viewing a detail view, close it first
        if self.viewing_memory.is_some() {
            self.viewing_memory = None;
            self.status_message = "Closed memory view".to_string();
            return;
        }
        if self.viewing_entity.is_some() {
            self.viewing_entity = None;
            self.status_message = "Closed entity view".to_string();
            return;
        }
        if self.viewing_document.is_some() {
            self.viewing_document = None;
            self.status_message = "Closed document view".to_string();
            return;
        }
        if self.viewing_recall_result.is_some() {
            self.viewing_recall_result = None;
            self.status_message = "Closed recall result view".to_string();
            return;
        }

        // Otherwise go back to previous view
        if let Some(prev_view) = self.view_history.pop() {
            self.view = prev_view;
            let _ = self.refresh();
        }
    }

    fn switch_to_view(&mut self, new_view: View) -> Result<()> {
        if self.view != new_view {
            self.view_history.push(self.view.clone());
            self.view = new_view;
            self.refresh()?;
        }
        Ok(())
    }

    fn delete_selected_document(&mut self) -> Result<()> {
        if let View::Documents(bank_id) = &self.view {
            if let Some(i) = self.documents_state.selected() {
                if let Some(doc) = self.documents.get(i) {
                    let doc_id = doc.get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    if !doc_id.is_empty() {
                        match self.client.delete_document(bank_id, doc_id, false) {
                            Ok(_) => {
                                self.status_message = format!("Deleted document: {}", doc_id);
                                self.refresh()?;
                            }
                            Err(e) => {
                                self.error_message = format!("Failed to delete document: {}", e);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // Shortcuts bar (context + shortcuts, max 3 rows + 2 border)
            Constraint::Length(3),  // Header
            Constraint::Min(0),     // Main content
            Constraint::Length(1),  // Footer/status only (no border)
        ])
        .split(f.area());

    // Control bar
    render_control_bar(f, app, chunks[0]);

    // Header
    render_header(f, app, chunks[1]);

    // Main content
    if app.show_help {
        render_help(f, chunks[2]);
    } else {
        match &app.view {
            View::Banks => render_banks(f, app, chunks[2]),
            View::Memories(_) => render_memories(f, app, chunks[2]),
            View::Entities(_) => render_entities(f, app, chunks[2]),
            View::Documents(_) => render_documents(f, app, chunks[2]),
            View::Query(_) => render_query(f, app, chunks[2]),
        }
    }

    // Footer
    render_footer(f, app, chunks[3]);
}

fn render_control_bar(f: &mut Frame, app: &App, area: Rect) {
    // Build contextual shortcuts based on view and input mode
    let shortcuts = match (&app.view, &app.input_mode) {
        (View::Banks, InputMode::Normal) => vec![
            ("Enter", "Select", BRAND_START),
            ("R", "Refresh", BRAND_MID),
            ("?", "Help", BRAND_END),
            ("q", "Quit", Color::Red),
        ],
        (View::Memories(_), InputMode::Normal) => vec![
            ("Enter", "View", BRAND_START),
            ("/", "Query", BRAND_MID),
            ("←→", "Scroll", BRAND_START),
            ("n", "Next", BRAND_MID),
            ("p", "Prev", BRAND_MID),
            ("Esc", "Back", BRAND_END),
            ("R", "Refresh", BRAND_END),
            ("?", "Help", BRAND_END),
            ("q", "Quit", Color::Red),
        ],
        (View::Entities(_), InputMode::Normal) => vec![
            ("Enter", "View", BRAND_START),
            ("/", "Query", BRAND_MID),
            ("←→", "Scroll", BRAND_START),
            ("Esc", "Back", BRAND_END),
            ("R", "Refresh", BRAND_END),
            ("?", "Help", BRAND_END),
            ("q", "Quit", Color::Red),
        ],
        (View::Documents(_), InputMode::Normal) => vec![
            ("Enter", "View", BRAND_START),
            ("/", "Query", BRAND_MID),
            ("←→", "Scroll", BRAND_START),
            ("Del", "Delete", Color::Red),
            ("Esc", "Back", BRAND_END),
            ("R", "Refresh", BRAND_END),
            ("?", "Help", BRAND_END),
            ("q", "Quit", Color::Red),
        ],
        (View::Query(_), InputMode::Normal) => {
            let mut shortcuts = vec![
                ("/", "Query", BRAND_MID),
                ("m", "Mode", BRAND_START),
            ];
            if app.query_mode == QueryMode::Recall {
                shortcuts.push(("←→", "Scroll", BRAND_START));
            }
            shortcuts.extend_from_slice(&[
                ("b", "Budget", BRAND_END),
                ("+/-", "Tokens", BRAND_END),
                ("Esc", "Back", BRAND_END),
                ("?", "Help", BRAND_END),
                ("q", "Quit", Color::Red),
            ]);
            shortcuts
        },
        (View::Query(_), InputMode::Query) => vec![
            ("Enter", "Execute", BRAND_MID),
            ("Esc", "Cancel", Color::Red),
        ],
        _ => vec![
            ("?", "Help", BRAND_END),
            ("q", "Quit", Color::Red),
        ],
    };

    // Split into left (context) and right (shortcuts) sections
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(30),  // Context on left
            Constraint::Percentage(70),  // Shortcuts on right
        ])
        .split(area);

    // Left: Context info
    let context_info = match &app.view {
        View::Banks => "Context: Banks List".to_string(),
        View::Memories(bank_id) => format!("Context: Memories\nBank: {}", bank_id),
        View::Entities(bank_id) => format!("Context: Entities\nBank: {}", bank_id),
        View::Documents(bank_id) => format!("Context: Documents\nBank: {}", bank_id),
        View::Query(_bank_id) => {
            let mode = match app.query_mode {
                QueryMode::Recall => "Recall",
                QueryMode::Reflect => "Reflect",
            };
            format!("Mode: {}\nBudget: {:?} | Tokens: {}", mode, app.query_budget, app.query_max_tokens)
        }
    };

    let context_widget = Paragraph::new(context_info)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(BRAND_START))
            .title(" Context "))
        .style(Style::default().fg(BRAND_END).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Left);
    f.render_widget(context_widget, columns[0]);

    // Right: Shortcuts in columns if many
    // Calculate shortcuts per column (max 3 lines of shortcuts)
    let max_shortcuts_per_col = 3;
    let num_cols = (shortcuts.len() + max_shortcuts_per_col - 1) / max_shortcuts_per_col;

    let mut shortcut_lines = vec![];
    for row in 0..max_shortcuts_per_col {
        let mut line_spans = vec![];

        for col in 0..num_cols {
            let idx = col * max_shortcuts_per_col + row;
            if idx < shortcuts.len() {
                let (key, desc, color) = &shortcuts[idx];

                // Each shortcut with proper alignment
                let shortcut_text = format!("<{}> {:<10}", key, desc);

                line_spans.push(Span::styled(
                    shortcut_text,
                    Style::default().fg(*color).add_modifier(Modifier::BOLD)
                ));
            }
        }

        if !line_spans.is_empty() {
            shortcut_lines.push(Line::from(line_spans));
        }
    }

    let shortcuts_widget = Paragraph::new(shortcut_lines)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(BRAND_START))
            .title(" Shortcuts "))
        .alignment(Alignment::Left);

    f.render_widget(shortcuts_widget, columns[1]);
}

fn render_header(f: &mut Frame, app: &App, area: Rect) {
    let bank_info = if let Some(bank_id) = app.view.bank_id() {
        format!(" [{}]", bank_id)
    } else {
        String::new()
    };

    let title = format!("Hindsight Explorer - {}{}", app.view.title(), bank_info);

    let header = Paragraph::new(title)
        .style(Style::default().fg(BRAND_START).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::ALL));

    f.render_widget(header, area);
}

fn render_footer(f: &mut Frame, app: &App, area: Rect) {
    // Simple status line only (shortcuts are now at the top, no border)
    let status_line = if !app.error_message.is_empty() {
        Line::from(vec![
            Span::styled(" Error: ", Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)),
            Span::raw(&app.error_message),
        ])
    } else if app.loading {
        Line::from(Span::styled(" Loading...", Style::default().fg(BRAND_END).add_modifier(Modifier::BOLD)))
    } else if !app.status_message.is_empty() {
        Line::from(vec![
            Span::raw(" "),
            Span::styled(&app.status_message, Style::default().fg(BRAND_MID)),
        ])
    } else {
        Line::from("")
    };

    let footer = Paragraph::new(status_line).alignment(Alignment::Left);
    f.render_widget(footer, area);
}

fn render_banks(f: &mut Frame, app: &mut App, area: Rect) {
    let items: Vec<ListItem> = app
        .banks
        .iter()
        .map(|bank| {
            let name = if bank.name.is_empty() { "Unnamed" } else { &bank.name };
            let content = format!("{} - {}", bank.bank_id, name);
            ListItem::new(content).style(Style::default().fg(Color::White))
        })
        .collect();

    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title("Banks"))
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    f.render_stateful_widget(list, area, &mut app.banks_state);
}

fn render_memories(f: &mut Frame, app: &mut App, area: Rect) {
    // If viewing a memory, show its details
    if let Some(memory) = &app.viewing_memory {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(7),  // Memory metadata
                Constraint::Min(0),     // Full text content
            ])
            .split(area);

        // Metadata section
        let mem_type = memory.get("fact_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let mentioned_at = memory.get("mentioned_at")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let occurred_start = memory.get("occurred_start")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let occurred_end = memory.get("occurred_end")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let metadata_text = format!(
            "Type: {}\nMentioned At: {}\nOccurred: {} to {}",
            mem_type, mentioned_at, occurred_start, occurred_end
        );

        let metadata = Paragraph::new(metadata_text)
            .block(Block::default().borders(Borders::ALL).title("Memory Metadata"))
            .style(Style::default().fg(BRAND_START));

        f.render_widget(metadata, chunks[0]);

        // Full text content
        let text = memory.get("text").and_then(|v| v.as_str()).unwrap_or("No text available");

        let content_widget = Paragraph::new(text)
            .block(Block::default().borders(Borders::ALL).title("Full Text (Esc to close)"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(Color::White));

        f.render_widget(content_widget, chunks[1]);
    } else {
        // Show memory list as table
        let mut items = vec![
            // Header row
            ListItem::new(format!("{:<10} {:<18} {:<18} {}", "TYPE", "MENTIONED AT", "OCCURRED AT", "TEXT"))
                .style(Style::default().fg(BRAND_START).add_modifier(Modifier::BOLD))
        ];

        // Data rows
        for memory in &app.memories {
            let mem_type = memory.get("fact_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let mentioned = memory.get("mentioned_at")
                .and_then(|v| v.as_str())
                .and_then(|s| s.split('T').next())
                .unwrap_or("-");
            let occurred = memory.get("occurred_start")
                .and_then(|v| v.as_str())
                .and_then(|s| s.split('T').next())
                .unwrap_or("-");
            let text = memory.get("text").and_then(|v| v.as_str()).unwrap_or("");

            // Apply horizontal scroll
            let scrolled_text: String = text.chars().skip(app.horizontal_scroll).take(80).collect();

            let content = format!("{:<10} {:<18} {:<18} {}", mem_type, mentioned, occurred, scrolled_text);
            items.push(ListItem::new(content).style(Style::default().fg(Color::White)));
        }

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(format!("Memories ({}) - Press Enter to view full text", app.memories.len())))
            .highlight_style(
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(">> ");

        f.render_stateful_widget(list, area, &mut app.memories_state);
    }
}

fn render_entities(f: &mut Frame, app: &mut App, area: Rect) {
    // If viewing an entity, show its details
    if let Some(entity) = &app.viewing_entity {
        let entity_type = entity.metadata.as_ref()
            .and_then(|m| m.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let metadata_text = format!(
            "Name: {}\nType: {}\nMentions: {}\nFirst Seen: {}\nLast Seen: {}",
            entity.canonical_name,
            entity_type,
            entity.mention_count,
            entity.first_seen.as_deref().unwrap_or("unknown"),
            entity.last_seen.as_deref().unwrap_or("unknown")
        );

        let metadata = Paragraph::new(metadata_text)
            .block(Block::default().borders(Borders::ALL).title("Entity Details (Esc to close)"))
            .style(Style::default().fg(BRAND_START))
            .wrap(Wrap { trim: false });

        f.render_widget(metadata, area);
    } else {
        // Show entity list as table
        let mut items = vec![
            // Header row
            ListItem::new(format!("{:<40} {:<15} {:<10}", "NAME", "TYPE", "MENTIONS"))
                .style(Style::default().fg(BRAND_START).add_modifier(Modifier::BOLD))
        ];

        // Data rows
        for entity in &app.entities {
            let name = &entity.canonical_name;
            // Apply horizontal scroll to name
            let scrolled_name: String = name.chars().skip(app.horizontal_scroll).take(40).collect();
            let entity_type = entity.metadata.as_ref()
                .and_then(|m| m.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let mentions = entity.mention_count;

            let content = format!("{:<40} {:<15} {:<10}", scrolled_name, entity_type, mentions);
            items.push(ListItem::new(content).style(Style::default().fg(Color::White)));
        }

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(format!("Entities ({}) - Press Enter to view details", app.entities.len())))
            .highlight_style(
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(">> ");

        f.render_stateful_widget(list, area, &mut app.entities_state);
    }
}

fn render_documents(f: &mut Frame, app: &mut App, area: Rect) {
    // If viewing a document, show its content
    if let Some(doc) = &app.viewing_document {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(5),  // Document metadata
                Constraint::Min(0),     // Content
            ])
            .split(area);

        // Metadata section
        let doc_id = doc.get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let content_type = doc.get("content_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let created_at = doc.get("created_at")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let metadata_text = format!(
            "ID: {}\nType: {}\nCreated: {}\n",
            doc_id, content_type, created_at
        );

        let metadata = Paragraph::new(metadata_text)
            .block(Block::default().borders(Borders::ALL).title("Document Metadata"))
            .style(Style::default().fg(BRAND_START));

        f.render_widget(metadata, chunks[0]);

        // Content section
        let content = doc.get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("No content available");

        let content_widget = Paragraph::new(content)
            .block(Block::default().borders(Borders::ALL).title("Content (Esc to close)"))
            .wrap(Wrap { trim: false })
            .style(Style::default().fg(Color::White));

        f.render_widget(content_widget, chunks[1]);
    } else {
        // Show document list as table
        let mut items = vec![
            // Header row
            ListItem::new(format!("{:<40} {:<20} {}", "ID", "TYPE", "CREATED"))
                .style(Style::default().fg(BRAND_START).add_modifier(Modifier::BOLD))
        ];

        // Data rows
        for doc in &app.documents {
            let id = doc.get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            // Apply horizontal scroll to id
            let scrolled_id: String = id.chars().skip(app.horizontal_scroll).take(40).collect();
            let content_type = doc.get("content_type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let created = doc.get("created_at")
                .and_then(|v| v.as_str())
                .and_then(|s| s.split('T').next())
                .unwrap_or("unknown");

            let content = format!("{:<40} {:<20} {}", scrolled_id, content_type, created);
            items.push(ListItem::new(content).style(Style::default().fg(Color::White)));
        }

        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(format!("Documents ({}) - Press Enter to view content", app.documents.len())))
            .highlight_style(
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(">> ");

        f.render_stateful_widget(list, area, &mut app.documents_state);
    }
}

fn render_query(f: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Query input
            Constraint::Min(0),     // Results or Response
        ])
        .split(area);

    // Query input
    let query_style = if app.input_mode == InputMode::Query {
        Style::default().fg(BRAND_END)
    } else {
        Style::default()
    };

    let mode_label = match app.query_mode {
        QueryMode::Recall => "Recall",
        QueryMode::Reflect => "Reflect",
    };
    let title = format!("{} Query (press / to edit, m to toggle mode)", mode_label);

    let query = Paragraph::new(app.query_text.as_str())
        .style(query_style)
        .block(Block::default().borders(Borders::ALL).title(title));

    f.render_widget(query, chunks[0]);

    // Show loading indicator if loading
    if app.loading {
        let loading_text = match app.query_mode {
            QueryMode::Recall => "Searching memories...",
            QueryMode::Reflect => "Reflecting on memories...",
        };

        // Create animated dots based on time
        let dots = ".".repeat(((std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() / 500) % 4) as usize);

        let loading_lines = vec![
            Line::from(""),
            Line::from(""),
            Line::from(vec![
                Span::styled("    ", Style::default()),
                Span::styled(format!("{}{}", loading_text, dots), Style::default().fg(BRAND_MID).add_modifier(Modifier::BOLD)),
            ]),
            Line::from(""),
            Line::from(Span::styled("    Please wait while we process your query...", Style::default().fg(Color::DarkGray))),
        ];

        let loading_widget = Paragraph::new(loading_lines)
            .block(Block::default().borders(Borders::ALL).title(format!("{} in progress", mode_label)))
            .alignment(Alignment::Left);

        f.render_widget(loading_widget, chunks[1]);
        return;
    }

    // Results or Response based on mode
    match app.query_mode {
        QueryMode::Recall => {
            // If viewing a recall result, show its details
            if let Some(result) = &app.viewing_recall_result {
                let recall_chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(7),  // Metadata
                        Constraint::Min(0),     // Full text
                    ])
                    .split(chunks[1]);

                // Metadata section
                let mem_type = result.type_.as_deref().unwrap_or("unknown");
                let occurred_start = result.occurred_start.as_deref().unwrap_or("unknown");
                let occurred_end = result.occurred_end.as_deref().unwrap_or("unknown");
                let mentioned_at = result.mentioned_at.as_deref().unwrap_or("unknown");

                let metadata_text = format!(
                    "Type: {}\nMentioned At: {}\nOccurred: {} to {}",
                    mem_type, mentioned_at, occurred_start, occurred_end
                );

                let metadata = Paragraph::new(metadata_text)
                    .block(Block::default().borders(Borders::ALL).title("Recall Result Metadata"))
                    .style(Style::default().fg(BRAND_START));

                f.render_widget(metadata, recall_chunks[0]);

                // Full text content
                let content_widget = Paragraph::new(result.text.as_str())
                    .block(Block::default().borders(Borders::ALL).title("Full Text (Esc to close)"))
                    .wrap(Wrap { trim: false })
                    .style(Style::default().fg(Color::White));

                f.render_widget(content_widget, recall_chunks[1]);
            } else {
                // Show results as a table like memories
                let mut items = vec![
                    // Header row
                    ListItem::new(format!("{:<10} {:<18} {:<18} {}", "TYPE", "OCCURRED START", "OCCURRED END", "TEXT"))
                        .style(Style::default().fg(BRAND_START).add_modifier(Modifier::BOLD))
                ];

                // Data rows
                for result in &app.query_results {
                    let mem_type = result.type_.as_deref().unwrap_or("unknown");
                    let occurred_start = result.occurred_start.as_deref()
                        .and_then(|s| s.split('T').next())
                        .unwrap_or("-");
                    let occurred_end = result.occurred_end.as_deref()
                        .and_then(|s| s.split('T').next())
                        .unwrap_or("-");
                    let text = &result.text;

                    // Apply horizontal scroll
                    let scrolled_text: String = text.chars().skip(app.horizontal_scroll).take(80).collect();

                    let content = format!("{:<10} {:<18} {:<18} {}", mem_type, occurred_start, occurred_end, scrolled_text);
                    items.push(ListItem::new(content).style(Style::default().fg(Color::White)));
                }

                let list = List::new(items)
                    .block(Block::default().borders(Borders::ALL).title(format!("Recall Results ({}) - Press Enter to view full text", app.query_results.len())))
                    .highlight_style(
                        Style::default()
                            .bg(Color::DarkGray)
                            .add_modifier(Modifier::BOLD),
                    )
                    .highlight_symbol(">> ");

                f.render_stateful_widget(list, chunks[1], &mut app.query_results_state);
            }
        }
        QueryMode::Reflect => {
            let response_text = if app.query_response.is_empty() {
                "No response yet. Enter a query and press Enter to get a reflection."
            } else {
                app.query_response.as_str()
            };

            let response = Paragraph::new(response_text)
                .style(Style::default().fg(Color::White))
                .block(Block::default().borders(Borders::ALL).title("Reflect Response"))
                .wrap(Wrap { trim: false });

            f.render_widget(response, chunks[1]);
        }
    }
}

fn render_help(f: &mut Frame, area: Rect) {
    let help_text = vec![
        Line::from(Span::styled("Hindsight Explorer - Keyboard Shortcuts", Style::default().fg(BRAND_START).add_modifier(Modifier::BOLD))),
        Line::from(""),
        Line::from(vec![
            Span::styled("Navigation Flow", Style::default().fg(BRAND_END).add_modifier(Modifier::BOLD)),
        ]),
        Line::from("  1. Start by selecting a bank (Enter)"),
        Line::from("  2. View memories, entities, or documents for that bank"),
        Line::from("  3. Press / from any view to query (recall/reflect)"),
        Line::from(""),
        Line::from(vec![
            Span::styled("Basic Navigation", Style::default().fg(BRAND_END).add_modifier(Modifier::BOLD)),
        ]),
        Line::from("  ↑/↓, j/k    - Navigate up/down in lists"),
        Line::from("  ←/→, h/l    - Scroll text left/right in tables"),
        Line::from("  Enter       - Select item / view details"),
        Line::from("  Esc         - Go back / close detail view"),
        Line::from(""),
        Line::from(vec![
            Span::styled("Query View", Style::default().fg(BRAND_END).add_modifier(Modifier::BOLD)),
        ]),
        Line::from("  /           - Start or edit query (from any non-bank view)"),
        Line::from("  m           - Toggle mode (Recall ↔ Reflect)"),
        Line::from("  b           - Cycle budget (Low → Mid → High)"),
        Line::from("  +/-         - Adjust max tokens"),
        Line::from("  Enter       - Execute query"),
        Line::from(""),
        Line::from(vec![
            Span::styled("General", Style::default().fg(BRAND_END).add_modifier(Modifier::BOLD)),
        ]),
        Line::from("  R           - Refresh current view"),
        Line::from("  ?           - Toggle this help screen"),
        Line::from("  q           - Quit"),
        Line::from(""),
        Line::from(Span::styled("Press ? to close help", Style::default().fg(Color::DarkGray))),
    ];

    let help = Paragraph::new(help_text)
        .block(Block::default().borders(Borders::ALL).title("Help"))
        .alignment(Alignment::Left);

    f.render_widget(help, area);
}

fn run_app<B: Backend>(terminal: &mut Terminal<B>, mut app: App) -> Result<()> {
    // Initial load
    app.refresh()?;

    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                // Handle Ctrl+C to exit
                if key.code == KeyCode::Char('c') && key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) {
                    return Ok(());
                }

                match app.input_mode {
                    InputMode::Normal => {
                        match key.code {
                            KeyCode::Char('q') => return Ok(()),
                            KeyCode::Char('?') => app.show_help = !app.show_help,

                            // Navigation
                            KeyCode::Down | KeyCode::Char('j') => app.next_item(),
                            KeyCode::Up | KeyCode::Char('k') => app.previous_item(),
                            KeyCode::Left | KeyCode::Char('h') => app.scroll_left(),
                            KeyCode::Right | KeyCode::Char('l') => app.scroll_right(),
                            KeyCode::Enter => {
                                app.reset_horizontal_scroll();
                                app.enter_view()?;
                            }
                            KeyCode::Esc => {
                                app.reset_horizontal_scroll();
                                app.go_back();
                            }

                            // Refresh
                            KeyCode::Char('R') => {
                                app.refresh()?;
                            }

                            // Query input - start query from any non-bank view
                            KeyCode::Char('/') => {
                                match &app.view {
                                    View::Banks => {
                                        app.error_message = "Select a bank first".to_string();
                                    }
                                    View::Query(_) => {
                                        app.input_mode = InputMode::Query;
                                    }
                                    _ => {
                                        // Switch to Query view using current bank
                                        if let Some(bank_id) = app.selected_bank_id.clone() {
                                            app.switch_to_view(View::Query(bank_id))?;
                                            app.input_mode = InputMode::Query;
                                        } else {
                                            app.error_message = "No bank selected".to_string();
                                        }
                                    }
                                }
                            }

                            // Query view controls
                            KeyCode::Char('m') => {
                                if matches!(app.view, View::Query(_)) {
                                    app.toggle_query_mode();
                                }
                            }
                            KeyCode::Char('b') => {
                                if matches!(app.view, View::Query(_)) {
                                    app.cycle_budget();
                                }
                            }
                            KeyCode::Char('+') | KeyCode::Char('=') => {
                                if matches!(app.view, View::Query(_)) {
                                    app.adjust_max_tokens(true);
                                }
                            }
                            KeyCode::Char('-') => {
                                if matches!(app.view, View::Query(_)) {
                                    app.adjust_max_tokens(false);
                                }
                            }

                            // Delete document
                            KeyCode::Delete => {
                                if matches!(app.view, View::Documents(_)) {
                                    app.delete_selected_document()?;
                                }
                            }

                            // Pagination for memories
                            KeyCode::Char('n') => {
                                if matches!(app.view, View::Memories(_)) {
                                    app.load_more_memories()?;
                                }
                            }
                            KeyCode::Char('p') => {
                                if matches!(app.view, View::Memories(_)) {
                                    app.load_prev_memories()?;
                                }
                            }

                            _ => {}
                        }
                    }
                    InputMode::Query => {
                        match key.code {
                            KeyCode::Enter => {
                                if matches!(app.view, View::Query(_)) {
                                    app.execute_query();
                                }
                            }
                            KeyCode::Esc => {
                                app.input_mode = InputMode::Normal;
                            }
                            KeyCode::Char(c) => {
                                if matches!(app.view, View::Query(_)) {
                                    app.query_text.push(c);
                                }
                            }
                            KeyCode::Backspace => {
                                if matches!(app.view, View::Query(_)) {
                                    app.query_text.pop();
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // Check for query results from background thread
        app.check_query_result();

        // Auto-refresh check
        app.do_auto_refresh()?;
    }
}

pub fn run(client: &ApiClient) -> Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run it
    let app = App::new(client.clone());
    let res = run_app(&mut terminal, app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("Error: {:?}", err);
    }

    Ok(())
}
