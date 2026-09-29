// ─── Video & Analysis Types ──────────────────────────────────────────────────

export interface VideoMetadata {
  duration: number;       // seconds
  width: number;
  height: number;
  aspectRatio: string;    // e.g. "9:16"
  frameRate: number;
  orientation: "portrait" | "landscape" | "square";
  hasAudio: boolean;
  fileSize: number;       // bytes
  format: string;
  bitrate?: number;
}

export interface SceneAnalysis {
  sceneNumber: number;
  startTime: number;      // seconds
  endTime: number;
  duration: number;
  purpose: string;
  subject: string;
  cameraAngle: string;
  cameraMovement: string;
  framing: string;
  composition: string;
  lighting: string;
  background: string;
  action: string;
  transition: string;
  text: string;
  graphics: string;
  sound: string;
  music: string;
  voiceover: string;
  visualEffects: string;
  editingTechnique: string;
  pacing: string;
}

export interface CreativeStrategy {
  hook: string;
  storytellingStructure: string;
  emotionalTone: string;
  targetAudience: string;
  contentCategory: string;
  pacing: string;
  visualIdentity: string;
  cta: string;
  retentionTechniques: string[];
  patternInterrupts: string[];
  useOfCuriosity: string;
  useOfText: string;
}

// ─── Recreation Guide Sections ───────────────────────────────────────────────

export interface ReelOverview {
  about: string;
  whatMakesItEffective: string;
  targetAudience: string;
  contentCategory: string;
  approximateDuration: string;
  overallStyle: string;
  mainCreativeIdea: string;
}

export interface SkillLevel {
  level: "Beginner" | "Intermediate" | "Advanced";
  explanation: string;
  mustKnow: string[];
  optional: string[];
}

export interface SkillToLearn {
  name: string;
  whatItMeans: string;
  whyNeeded: string;
  difficulty: "Easy" | "Medium" | "Hard";
  timeToLearn: string;
  practice: string;
}

export interface EquipmentItem {
  name: string;
  category: "MUST HAVE" | "NICE TO HAVE" | "OPTIONAL / PROFESSIONAL";
  whyNeeded: string;
  cheaperAlternative: string;
  canSkip: boolean;
}

export interface AppRecommendation {
  name: string;
  purpose: "FILMING" | "EDITING" | "AUDIO" | "GRAPHICS" | "COLOR" | "CAPTIONS" | "THUMBNAIL" | "PUBLISHING";
  usedFor: string;
  whyNeeded: string;
  difficulty: "Easy" | "Medium" | "Hard";
  cost: string;
  beginnerAlternative: string;
  professionalAlternative: string;
}

export interface Technique {
  name: string;
  what: string;
  why: string;
  how: string;
  alternative: string;
}

export interface ShotListItem {
  shotNumber: number;
  timestamp: string;
  duration: string;
  shotType: string;
  cameraAngle: string;
  cameraMovement: string;
  subject: string;
  action: string;
  lighting: string;
  audio: string;
  text: string;
  transition: string;
  howToShoot: string;
}

export interface ScriptSection {
  label: string;
  timeRange: string;
  description: string;
  originalStructure: string;
  voiceoverInstructions: string;
  tone: string;
  speakingSpeed: string;
}

export interface ColorGrading {
  style: "natural" | "warm" | "cool" | "high contrast" | "muted" | "cinematic" | "vibrant";
  exposure: string;
  contrast: string;
  highlights: string;
  shadows: string;
  saturation: string;
  temperature: string;
  tint: string;
  skinTones: string;
  beginnerSettings: string;
}

export interface TransitionItem {
  type: string;
  timestamp: string;
  purpose: string;
  howItWorks: string;
  howToRecreate: string;
  beginnerAlternative: string;
}

export interface ExportSettings {
  platform: "Instagram Reels" | "YouTube Shorts" | "TikTok";
  resolution: string;
  aspectRatio: string;
  frameRate: string;
  codec: string;
  bitrate: string;
  audioFormat: string;
}

export interface PublishingGuide {
  captionStructure: string;
  cta: string;
  hashtags: string[];
  title: string;
  coverIdea: string;
  postingChecklist: string[];
}

export interface RecreationGuide {
  projectId: string;
  videoTitle: string;
  analyzedAt: string;

  // All 23 sections
  section1_overview: ReelOverview;
  section2_skillLevel: SkillLevel;
  section3_skillsToLearn: SkillToLearn[];
  section4_equipment: EquipmentItem[];
  section5_apps: AppRecommendation[];
  section6_techniques: Technique[];
  section7_preProduction: { steps: string[]; checklist: string[] };
  section8_script: ScriptSection[];
  section9_shotList: ShotListItem[];
  section10_filmingInstructions: string;
  section11_lightingSetup: string;
  section12_audioMusic: string;
  section13_editingTimeline: string;
  section14_appSpecificGuide: string;
  section15_colorGrading: ColorGrading;
  section16_textGraphics: string;
  section17_transitions: TransitionItem[];
  section18_exportSettings: ExportSettings[];
  section19_publishing: PublishingGuide;
  section20_beginnerVersion: string;
  section21_professionalVersion: string;
  section22_timeBudget: string;
  section23_finalChecklist: string[];
}

// ─── Project ─────────────────────────────────────────────────────────────────

export interface Project {
  id: string;
  title: string;
  videoUrl?: string;
  videoPath?: string;
  thumbnailUrl?: string;
  sourceType: "upload" | "url" | "demo";
  createdAt: string;
  updatedAt: string;
  metadata?: VideoMetadata;
  scenes?: SceneAnalysis[];
  creativeStrategy?: CreativeStrategy;
  guide?: RecreationGuide;
  status: "pending" | "analyzing" | "complete" | "error";
  error?: string;
  userId: string;
  isDemo?: boolean;
}

// ─── Memory / Hindsight ───────────────────────────────────────────────────────

export interface UserProfile {
  skillLevel?: string;
  preferredEditingSoftware?: string;
  preferredFilmingDevice?: string;
  cameraEquipment?: string;
  microphone?: string;
  lightingEquipment?: string;
  preferredPlatforms?: string[];
  preferredReelDuration?: string;
  preferredAspectRatio?: string;
}

export interface CreativePreferences {
  editingStyle?: string;
  pacing?: string;
  transitions?: string;
  colorStyle?: string;
  musicStyle?: string;
  fonts?: string;
  captionStyle?: string;
  hookStyle?: string;
  cameraStyle?: string;
  contentCategories?: string[];
}

export interface WorkflowPreferences {
  appsOwned?: string[];
  editingSoftware?: string;
  filmingSetup?: string;
  availableEquipment?: string[];
  preferredWorkflow?: string;
}

export interface MemoryEntry {
  id: string;
  text: string;
  type: string;
  learnedAt: string;
  category: "profile" | "creative" | "workflow" | "project";
  whyItMatters: string;
}

// ─── Chat ─────────────────────────────────────────────────────────────────────

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  memoryUsed?: string[];
}

// ─── Create My Version ────────────────────────────────────────────────────────

export interface CreateMyVersionRequest {
  referenceProjectId: string;
  topic: string;
  promotingOrShowing: string;
  equipment: string;
  phoneOrCamera: string;
  editingSoftware: string;
  skillLevel: string;
  platform: string;
  desiredDuration: string;
  style: string;
  differences: string;
}

export interface CreateMyVersionResult {
  originalGuideId: string;
  customGuide: string;
  createdAt: string;
}

// ─── API Responses ────────────────────────────────────────────────────────────

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface AnalyzeResponse {
  project: Project;
  memoryUsed: boolean;
  recalledPreferences: string[];
}
