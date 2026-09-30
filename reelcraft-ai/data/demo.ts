/**
 * Demo seed data — a complete pre-analyzed food Reel for the hackathon demo.
 * This allows judges to experience the full product without uploading a video.
 */

import type { Project } from "@/types";

export const DEMO_PROJECT: Project = {
  id: "demo-food-reel-001",
  title: "Crispy Pasta Chips — Viral Food Reel",
  sourceType: "demo",
  createdAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
  updatedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
  status: "complete",
  userId: "demo-user-hackathon",
  isDemo: true,
  thumbnailUrl: undefined,
  metadata: {
    duration: 28,
    width: 1080,
    height: 1920,
    aspectRatio: "9:16",
    frameRate: 30,
    orientation: "portrait",
    hasAudio: true,
    fileSize: 45_000_000,
    format: "mp4",
  },
  creativeStrategy: {
    hook: "Quick reveal of a surprising food transformation (raw pasta → golden crispy chips)",
    storytellingStructure: "Hook → Process → Reveal → CTA",
    emotionalTone: "Satisfying, curious, joyful",
    targetAudience: "Food lovers aged 18-34, home cooks, snack enthusiasts",
    contentCategory: "Food / Recipe",
    pacing: "Fast-paced with rhythm cuts synced to beat drops",
    visualIdentity: "Warm tones, close-up food shots, clean white kitchen background",
    cta: "Save this for later!",
    retentionTechniques: [
      "Curiosity gap in the hook (shows finished product before showing process)",
      "Fast cuts matching music beat",
      "Text reveals synced with actions",
      "Satisfying ASMR sounds (sizzle, crunch)",
    ],
    patternInterrupts: [
      "Unexpected use of pasta as snack chip",
      "Speed ramp on the crunch reveal",
      "Close-up texture shot at 0.5x speed",
    ],
    useOfCuriosity: "Opens with finished chips and asks 'you'll never guess what this is made from'",
    useOfText: "Bold white text on dark background for hook, ingredient callouts, step numbers",
  },
  scenes: [
    {
      sceneNumber: 1,
      startTime: 0,
      endTime: 3,
      duration: 3,
      purpose: "Hook — create curiosity with the finished product",
      subject: "Crispy golden chips in a bowl",
      cameraAngle: "Overhead (bird's eye)",
      cameraMovement: "Slow push-in",
      framing: "Product centered, bowl filling 70% of frame",
      composition: "Centered, rule of thirds on the bowl",
      lighting: "Bright overhead window light, no shadows",
      background: "Clean white marble surface",
      action: "Hand picks up a chip and brings it toward camera",
      transition: "Hard cut to black",
      text: "'Wait... this is just pasta 🤯'",
      graphics: "Bold white text, bottom-center",
      sound: "Music starts — upbeat lo-fi beat",
      music: "Upbeat lo-fi / trap beat with strong kick",
      voiceover: "None — text only",
      visualEffects: "Slight vignette on edges",
      editingTechnique: "Color grade: warm +15, saturation +20",
      pacing: "Slow reveal — holds for 3 seconds to build curiosity",
    },
    {
      sceneNumber: 2,
      startTime: 3,
      endTime: 8,
      duration: 5,
      purpose: "Ingredients reveal",
      subject: "Raw spaghetti, olive oil, seasoning",
      cameraAngle: "45-degree top-down",
      cameraMovement: "Static",
      framing: "Flat lay — ingredients arranged neatly",
      composition: "Diagonal arrangement, color contrast between items",
      lighting: "Natural window light from left",
      background: "White marble surface",
      action: "Hands place ingredients one by one into frame",
      transition: "Whip pan right",
      text: "'Just 3 ingredients:'",
      graphics: "Ingredient name text pop-up on each item",
      sound: "Beat drops on first ingredient reveal",
      music: "Continues",
      voiceover: "None",
      visualEffects: "None",
      editingTechnique: "Each ingredient reveal cuts on beat",
      pacing: "Quick cuts — each ingredient appears every 1 second",
    },
    {
      sceneNumber: 3,
      startTime: 8,
      endTime: 15,
      duration: 7,
      purpose: "Cooking process — fast-paced montage",
      subject: "Pasta in boiling water, then draining, then seasoning",
      cameraAngle: "Mix: 45-degree, side, overhead",
      cameraMovement: "Handheld slight shake on boiling shots",
      framing: "Close-up on the action",
      composition: "Subject centered, action fills frame",
      lighting: "Mixed — stovetop light + window fill",
      background: "Kitchen counter, stove",
      action: "Boil pasta, drain, toss with oil and seasoning",
      transition: "Jump cuts synced to beat",
      text: "Step numbers: '1', '2', '3' in corner",
      graphics: "Minimal — step numbers only",
      sound: "ASMR sounds mixed under music: water boiling, sizzle",
      music: "Continues, energy builds",
      voiceover: "None",
      visualEffects: "None",
      editingTechnique: "Jump cuts every 1-1.5 seconds on beat",
      pacing: "Fastest section — creates urgency and energy",
    },
    {
      sceneNumber: 4,
      startTime: 15,
      endTime: 22,
      duration: 7,
      purpose: "The satisfying reveal — baking and final crunch",
      subject: "Oven, golden chips, final crunch close-up",
      cameraAngle: "Front-on for oven, overhead for chips",
      cameraMovement: "Speed ramp: slow-mo on the crunch bite",
      framing: "Extreme close-up on the crunch",
      composition: "Crunch fills entire frame",
      lighting: "Bright, slightly warm to show golden color",
      background: "Oven, then white surface",
      action: "Open oven → chips come out golden → hand picks one → bite → crunch",
      transition: "Match cut from oven to bowl",
      text: "'Look at that crunch 😍'",
      graphics: "Sound wave animation on crunch moment",
      sound: "ASMR crunch sound mixed louder",
      music: "Beat drops on crunch",
      voiceover: "None",
      visualEffects: "Speed ramp: 1x → 0.3x on the crunch bite",
      editingTechnique: "Speed ramp + sound design",
      pacing: "Slows down for maximum impact on the satisfying moment",
    },
    {
      sceneNumber: 5,
      startTime: 22,
      endTime: 28,
      duration: 6,
      purpose: "CTA — save, follow, try it",
      subject: "Bowl of chips, creator logo/handle",
      cameraAngle: "45-degree overhead",
      cameraMovement: "Slow pull-out",
      framing: "Bowl centered, relaxed framing",
      composition: "Bowl left of frame, CTA text right",
      lighting: "Warm, soft — relaxed ending",
      background: "Same clean white marble",
      action: "Bowl sits still, steam rises from chips",
      transition: "Fade to black",
      text: "'Save this recipe! 🍝→🍟 #foodhack'",
      graphics: "Follow button animation overlay",
      sound: "Music fades down",
      music: "Fades to end",
      voiceover: "None",
      visualEffects: "Warm golden color grade, slight film grain",
      editingTechnique: "Pull-out + fade",
      pacing: "Slow — let the viewer breathe and absorb the CTA",
    },
  ],
  guide: {
    projectId: "demo-food-reel-001",
    videoTitle: "Crispy Pasta Chips — Viral Food Reel",
    analyzedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),

    section1_overview: {
      about: "A satisfying food hack Reel showing how to turn ordinary spaghetti pasta into crispy snack chips. The Reel uses a curiosity hook, fast-paced cooking montage, and a satisfying slow-motion crunch reveal.",
      whatMakesItEffective: "Three things make this Reel work: (1) The hook subverts expectations — you see chips before you know what they're made from. (2) The pacing is perfectly synced to a beat, making cuts feel satisfying. (3) The ASMR crunch moment at the end is deeply rewarding to watch and generates save behavior.",
      targetAudience: "Food lovers aged 18-34, home cooks, snack enthusiasts, people who love food hack content",
      contentCategory: "Food / Recipe / Food Hack",
      approximateDuration: "28 seconds",
      overallStyle: "Fast-paced, warm-toned, ASMR-heavy food content with minimal text and no voiceover",
      mainCreativeIdea: "Transform the familiar (pasta) into the unexpected (chips) — the surprise IS the content",
    },

    section2_skillLevel: {
      level: "Beginner",
      explanation: "This Reel requires only basic phone filming skills and simple editing. The most 'advanced' technique is the speed ramp, which can be done in CapCut with one tap. No tripod, expensive camera, or professional lighting is needed.",
      mustKnow: [
        "Basic phone camera operation",
        "Simple video cutting in CapCut or similar",
        "Adding text overlays",
        "Basic music sync (cut on the beat)",
        "Exporting vertical 9:16 video",
      ],
      optional: [
        "Speed ramp (adds polish but not essential)",
        "ASMR sound mixing",
        "Color grading with LUTs",
        "Beat-synced auto-cut tools",
      ],
    },

    section3_skillsToLearn: [
      {
        name: "Overhead Phone Mounting",
        whatItMeans: "Positioning your phone directly above your subject looking straight down",
        whyNeeded: "Most shots in this Reel use the overhead angle — it's the signature food Reel look",
        difficulty: "Easy",
        timeToLearn: "15 minutes",
        practice: "Stack books to make a flat bridge, rest phone on top. Or buy a ₹200–₹500 overhead phone holder.",
      },
      {
        name: "Cutting on the Beat",
        whatItMeans: "Making your video cuts land exactly when the music has a beat drop or kick drum hit",
        whyNeeded: "This is the main reason the Reel feels energetic and satisfying",
        difficulty: "Easy",
        timeToLearn: "30 minutes",
        practice: "In CapCut: import your song → tap 'Auto Beat' → your cuts snap to the beat automatically",
      },
      {
        name: "Speed Ramp",
        whatItMeans: "Changing the playback speed of a clip mid-clip — e.g., normal speed slowing to 30% at the key moment",
        whyNeeded: "Used on the crunch reveal for maximum impact",
        difficulty: "Easy",
        timeToLearn: "20 minutes",
        practice: "In CapCut: select clip → Speed → Curve → 'Hero' preset → adjust the valley point to make it slow at the right moment",
      },
      {
        name: "Close-up Food Filming",
        whatItMeans: "Getting your camera very close to food to capture texture, steam, and details",
        whyNeeded: "The crunch close-up and ingredient reveals require this",
        difficulty: "Easy",
        timeToLearn: "10 minutes",
        practice: "Place your phone 10-15cm from the food. Make sure you have good light — close-ups show everything including bad lighting.",
      },
      {
        name: "Basic Color Grading",
        whatItMeans: "Adjusting warmth, saturation and brightness to make food look more appetizing",
        whyNeeded: "The warm golden look of this Reel makes the food look more delicious",
        difficulty: "Easy",
        timeToLearn: "30 minutes",
        practice: "In CapCut: Adjust → Brightness +5, Warmth +15, Saturation +10. That's it for food.",
      },
    ],

    section4_equipment: [
      {
        name: "Smartphone",
        category: "MUST HAVE",
        whyNeeded: "Your filming device. The original Reel looks like it was shot on an iPhone.",
        cheaperAlternative: "Any smartphone with a decent rear camera works",
        canSkip: false,
      },
      {
        name: "Overhead Phone Mount / Tripod",
        category: "MUST HAVE",
        whyNeeded: "You need stable overhead shots. Handheld overhead is shaky and tiring.",
        cheaperAlternative: "Stack books and lean phone against something, or use the flexible arm of a lamp to hold your phone",
        canSkip: false,
      },
      {
        name: "Window / Natural Light",
        category: "MUST HAVE",
        whyNeeded: "The Reel uses natural window light. It's the best free food lighting available.",
        cheaperAlternative: "Shoot near a north-facing window during daytime. Avoid direct harsh sunlight.",
        canSkip: false,
      },
      {
        name: "Small LED Panel Light",
        category: "NICE TO HAVE",
        whyNeeded: "Provides consistent light when natural light is unavailable or inconsistent",
        cheaperAlternative: "A desk lamp with a white diffuser (tissue paper works)",
        canSkip: true,
      },
      {
        name: "Marble / White Surface Board",
        category: "NICE TO HAVE",
        whyNeeded: "The clean white background makes the food pop. A marble effect vinyl sheet is cheap.",
        cheaperAlternative: "White cutting board, white A3 paper, or a light-coloured surface",
        canSkip: true,
      },
      {
        name: "Gimbal / Camera Stabiliser",
        category: "OPTIONAL / PROFESSIONAL",
        whyNeeded: "For ultra-smooth moving shots",
        cheaperAlternative: "Not needed for this Reel — most shots are static",
        canSkip: true,
      },
    ],

    section5_apps: [
      {
        name: "CapCut",
        purpose: "EDITING",
        usedFor: "Main editing app — cutting, speed ramp, text, music, color grading, export",
        whyNeeded: "Free, powerful, has auto-beat sync and speed ramp built in — perfect for this Reel",
        difficulty: "Easy",
        cost: "Free (with optional Pro features)",
        beginnerAlternative: "CapCut is already the beginner option",
        professionalAlternative: "Adobe Premiere Pro or DaVinci Resolve",
      },
      {
        name: "Native Phone Camera",
        purpose: "FILMING",
        usedFor: "Recording all shots",
        whyNeeded: "Modern phone cameras shoot 4K — more than good enough",
        difficulty: "Easy",
        cost: "Free",
        beginnerAlternative: "Same — native camera is recommended",
        professionalAlternative: "Mirrorless camera (Sony ZV-E10, etc.)",
      },
      {
        name: "Canva",
        purpose: "THUMBNAIL",
        usedFor: "Creating the Reel cover/thumbnail image",
        whyNeeded: "A good cover increases profile clicks",
        difficulty: "Easy",
        cost: "Free",
        beginnerAlternative: "Use a screenshot from the Reel",
        professionalAlternative: "Photoshop",
      },
    ],

    section6_techniques: [
      {
        name: "Curiosity Hook",
        what: "Opening with the finished result before showing the process, creating a 'how did they do that?' reaction",
        why: "This Reel opens with chips — you want to know how they made pasta into chips before the algorithm decides to keep you watching",
        how: "Film the finished dish FIRST. Add text like 'Wait… this is just pasta 🤯'. Then cut to ingredients. Never show the full recipe in the hook.",
        alternative: "Open with a question: 'What if I told you this was just pasta?'",
      },
      {
        name: "Beat-Synced Jump Cuts",
        what: "Cutting to a new shot every time the music hits a beat",
        why: "Makes the video feel energetic and satisfying. The brain expects the visual and audio rhythm to match.",
        how: "In CapCut: tap 'Auto Sync' or manually move each cut to land on a beat marker. Use the waveform as your guide.",
        alternative: "Cut every 1.5 seconds without music sync — still works but is less satisfying",
      },
      {
        name: "Speed Ramp to Slow Motion",
        what: "The video plays at normal speed, then slows to 30% speed at the key action moment",
        why: "Used on the crunch bite to make the satisfying moment last longer and feel more dramatic",
        how: "Film at 60fps or 120fps. In CapCut: Speed → Curve → select 'Hero'. The lowest point of the curve determines your slowest frame.",
        alternative: "Record the bite at 120fps and use it as-is in slow motion without a ramp",
      },
      {
        name: "ASMR Sound Design",
        what: "Boosting the natural sounds of cooking — sizzle, bubble, crunch — so they're audible and satisfying",
        why: "Food content with good ASMR sounds gets significantly higher save and completion rates",
        how: "Record some shots specifically for sound (phone close to the cooking). In CapCut: extract audio from your clip and boost it under the music.",
        alternative: "Download free ASMR cooking sounds from Pixabay and add them as a separate audio layer",
      },
      {
        name: "Overhead (Bird's Eye) Shot",
        what: "Camera positioned directly above the subject, pointing straight down at 90 degrees",
        why: "The definitive food Reel angle — shows the entire dish or process clearly from above",
        how: "Use an overhead mount or build one from books. Phone must be directly above, not angled. Use a small level app to check.",
        alternative: "45-degree angle shot from slightly above — easier to set up, slightly less dramatic",
      },
    ],

    section7_preProduction: {
      steps: [
        "1. Decide your specific food hack topic (pasta chips in this case)",
        "2. Write your hook text — the one sentence that creates curiosity",
        "3. List every shot you need (refer to the shot list below)",
        "4. Prepare all ingredients and props before filming — mise en place",
        "5. Set up your overhead mount and test your framing",
        "6. Set up your lighting (window or LED) and check for shadows",
        "7. Choose your music track — find one with a strong clear beat",
        "8. Do a test recording to check focus, exposure, and stability",
        "9. Cook your food and film simultaneously — or film separate takes",
        "10. Plan your CTA text and hashtags in advance",
      ],
      checklist: [
        "All ingredients prepared and plated",
        "Background/surface is clean and correct color",
        "Phone charged and storage cleared",
        "Overhead mount set up and stable",
        "Lighting checked — no harsh shadows on food",
        "Music track downloaded and ready",
        "Shot list printed or on second screen",
        "CapCut installed and updated",
      ],
    },

    section8_script: [
      {
        label: "HOOK",
        timeRange: "0:00 – 0:03",
        description: "Show finished chips with curiosity text. No voiceover.",
        originalStructure: "Visual reveal + text question",
        voiceoverInstructions: "None — text only for this section",
        tone: "Curious, surprising",
        speakingSpeed: "N/A",
      },
      {
        label: "INGREDIENTS",
        timeRange: "0:03 – 0:08",
        description: "Each ingredient appears on screen with its name as a text pop-up",
        originalStructure: "Beat-synced ingredient reveal",
        voiceoverInstructions: "If adding voiceover: speak quickly, one word per beat. 'Pasta. Oil. Salt.'",
        tone: "Energetic, clear",
        speakingSpeed: "Fast — 1 word per second",
      },
      {
        label: "PROCESS",
        timeRange: "0:08 – 0:15",
        description: "Fast cooking montage — boil, drain, season, bake",
        originalStructure: "Step-by-step with number overlays",
        voiceoverInstructions: "Optional: brief instruction per step. 'Boil al dente. Drain. Season. Bake at 200°C for 15 mins.'",
        tone: "Instructional but quick",
        speakingSpeed: "Fast",
      },
      {
        label: "REVEAL",
        timeRange: "0:15 – 0:22",
        description: "Oven open → golden chips → slow-motion crunch",
        originalStructure: "Visual payoff — the satisfying moment",
        voiceoverInstructions: "None OR a satisfied sound/reaction",
        tone: "Satisfying, rewarding",
        speakingSpeed: "N/A",
      },
      {
        label: "CTA",
        timeRange: "0:22 – 0:28",
        description: "Save prompt and hashtag",
        originalStructure: "Text CTA with follow prompt",
        voiceoverInstructions: "Optional: 'Save this for your next snack craving!'",
        tone: "Friendly, inviting",
        speakingSpeed: "Normal",
      },
    ],

    section9_shotList: [
      {
        shotNumber: 1,
        timestamp: "0:00–0:03",
        duration: "3 sec",
        shotType: "Extreme Close-Up",
        cameraAngle: "Overhead (90°)",
        cameraMovement: "Slow push-in (move phone 5cm closer over 3 seconds)",
        subject: "Finished pasta chips in bowl",
        action: "Hand reaches in and picks up one chip slowly",
        lighting: "Bright natural window light from above-left",
        audio: "Music starts",
        text: "'Wait... this is just pasta 🤯'",
        transition: "Hard cut",
        howToShoot: "Position phone 40cm above bowl using overhead mount. Enable 2x optical zoom. Slowly move phone 5cm downward during the shot. Focus lock on the chips.",
      },
      {
        shotNumber: 2,
        timestamp: "0:03–0:05",
        duration: "2 sec",
        shotType: "Medium Overhead",
        cameraAngle: "Overhead (90°)",
        cameraMovement: "Static",
        subject: "Raw spaghetti on surface",
        action: "Hand places pasta bundle onto surface",
        lighting: "Natural window light",
        audio: "Beat hit",
        text: "'Spaghetti'",
        transition: "Hard cut on beat",
        howToShoot: "Same overhead setup. Place pasta neatly — no loose strands. Film hand placing it from outside frame.",
      },
      {
        shotNumber: 3,
        timestamp: "0:05–0:07",
        duration: "2 sec",
        shotType: "Close-Up",
        cameraAngle: "Overhead (90°)",
        cameraMovement: "Static",
        subject: "Olive oil bottle + seasoning",
        action: "Hand places items beside pasta",
        lighting: "Natural window light",
        audio: "Beat hit",
        text: "'+ Olive Oil + Salt'",
        transition: "Whip pan right",
        howToShoot: "Same position. Add items one by one. Film each one individually and cut them together.",
      },
      {
        shotNumber: 4,
        timestamp: "0:08–0:10",
        duration: "2 sec",
        shotType: "Close-Up",
        cameraAngle: "Side-on (eye level)",
        cameraMovement: "Slight handheld movement",
        subject: "Pot of boiling water",
        action: "Pasta being dropped into boiling water",
        lighting: "Stovetop light + window fill",
        audio: "Beat + water sound",
        text: "'1. Boil al dente'",
        transition: "Jump cut",
        howToShoot: "Hold phone 30cm from pot at water level. Enable cinematic mode for background blur. Be careful of steam — use manual focus.",
      },
      {
        shotNumber: 5,
        timestamp: "0:10–0:12",
        duration: "2 sec",
        shotType: "Close-Up",
        cameraAngle: "45° top-down",
        cameraMovement: "Static",
        subject: "Pasta being drained in colander",
        action: "Draining — water falling",
        lighting: "Window light",
        audio: "Beat + water draining sound",
        text: "'2. Drain'",
        transition: "Jump cut",
        howToShoot: "Position phone 45° above the colander. The falling water is the visual interest here. Do multiple takes — water moves fast.",
      },
      {
        shotNumber: 6,
        timestamp: "0:12–0:15",
        duration: "3 sec",
        shotType: "Overhead",
        cameraAngle: "Overhead (90°)",
        cameraMovement: "Static",
        subject: "Pasta being tossed with oil and seasoning",
        action: "Hands tossing pasta in bowl with oil",
        lighting: "Natural window light",
        audio: "Beat + rustling sound",
        text: "'3. Season with oil + salt'",
        transition: "Jump cut",
        howToShoot: "Overhead position. Toss pasta quickly so it looks energetic. Film multiple 3-second takes — pick the best toss.",
      },
      {
        shotNumber: 7,
        timestamp: "0:15–0:18",
        duration: "3 sec",
        shotType: "Front-on Medium",
        cameraAngle: "Eye level",
        cameraMovement: "Static",
        subject: "Oven door opening, golden chips inside",
        action: "Open oven door to reveal golden chips on tray",
        lighting: "Oven interior light",
        audio: "Beat + oven click sound",
        text: "None",
        transition: "Match cut (oven open → chips close-up)",
        howToShoot: "Place phone on tripod at oven door level. Start recording, then slowly pull oven door open. The oven light illuminates the chips naturally.",
      },
      {
        shotNumber: 8,
        timestamp: "0:18–0:22",
        duration: "4 sec",
        shotType: "Extreme Close-Up",
        cameraAngle: "Side-on (macro)",
        cameraMovement: "Speed ramp: 1x → 0.3x on bite",
        subject: "Hand picking up chip, bringing to mouth, biting",
        action: "Pick up chip → move toward camera → BITE → CRUNCH",
        lighting: "Warm LED panel or window light from the side",
        audio: "Music drops + ASMR crunch boosted",
        text: "'Look at that crunch 😍'",
        transition: "Cut to CTA shot",
        howToShoot: "Film at 120fps. Place phone at table level 20cm from where you'll bite the chip. The crunch should happen at center frame. Film 5 takes.",
      },
      {
        shotNumber: 9,
        timestamp: "0:22–0:28",
        duration: "6 sec",
        shotType: "Medium Overhead",
        cameraAngle: "Overhead (45°)",
        cameraMovement: "Slow pull-out (move phone up 10cm over 4 seconds)",
        subject: "Bowl of finished chips, steam rising",
        action: "Chips sit still, slight steam visible",
        lighting: "Warm window light — golden hour if possible",
        audio: "Music fades down",
        text: "'Save this recipe! 🍝→🍟'",
        transition: "Fade to black",
        howToShoot: "45° angle, start close (30cm), slowly pull up to 50cm. Backlight slightly for steam visibility. Use slow motion then speed back up.",
      },
    ],

    section10_filmingInstructions: `## FILMING INSTRUCTIONS

### BEGINNER METHOD (Smartphone Only)

**Setup:**
Place your phone in an overhead mount (or DIY with books) 40cm above your cooking surface.

**Settings:**
- Resolution: 4K if available, or 1080p HD
- Frame rate: 60fps for normal shots, 120fps for the crunch shot
- Stabilisation: ON
- Flash: OFF (use natural light)
- Grid lines: ON (helps with centering)

**Lighting:**
Position your cooking surface directly next to a window. The light should come from one side, not overhead. Avoid direct sunlight — it creates harsh shadows.

**Distance per shot:**
- Overhead shots: 30-50cm from subject
- Close-ups: 10-20cm from subject
- Side shots: 30-40cm at subject level

**Tip:** Film every step at least 3 times. The first take is for practice. The second is your safety. The third is often the best.

---

### PRO METHOD (With Accessories)

**Setup:**
- Use a dedicated overhead phone/camera mount (₹500–₹2000)
- Add a small LED panel set to 3200K-4000K (warm)
- Position LED 45° from the subject, at table height
- Add a reflector (white foam board) on the opposite side

**Camera Settings (if using mirrorless):**
- ISO 100-400
- Aperture f/2.8–f/4 (shallow depth of field)
- Shutter speed: 2× frame rate (1/60 at 30fps)

**Audio:**
Use a clip-on microphone positioned near the cooking action to capture ASMR sounds cleanly.`,

    section11_lightingSetup: `## LIGHTING SETUP

### Style Analysis
This Reel uses: **Bright, warm, soft natural light** — the classic food content look.
- Key light: large window (left side)
- Fill: white reflector or second window (right side)
- No hard shadows on the food surface
- Slightly warm temperature (think golden hour glow on the food)

### Beginner Setup (Free)

\`\`\`
       WINDOW (natural light)
           ↓  ↓  ↓
    ← ← ← light ← ← ←
    
    WHITE BOARD (reflector)      PHONE (overhead)
    bounces light back              ↓
    
                    [ FOOD / SUBJECT ]

\`\`\`

**Instructions:**
1. Place your cooking surface directly beside your window
2. The window should be to your LEFT or RIGHT — never behind the phone
3. Place a white foam board or white paper on the OPPOSITE side to bounce light back
4. This creates soft, shadow-free food lighting for free

### Common Mistakes
- ❌ Light behind the phone creates flat, dull food
- ❌ Overhead single light creates harsh shadows in bowls
- ❌ Mixed color temperatures (window + warm bulb) creates ugly color casts
- ✅ One directional window light + white reflector = professional result`,

    section12_audioMusic: `## AUDIO & MUSIC PLAN

### Music Analysis
The Reel uses an **upbeat lo-fi beat with a strong kick drum** — fast tempo (~120-130 BPM), energetic but not aggressive.

### Where to Find Similar Music
- **CapCut built-in library** — search "food", "upbeat", "trending"
- **Instagram Reels audio** — browse trending sounds in the Reels tab
- **Epidemic Sound** — "upbeat kitchen" or "energetic lo-fi" (paid)
- **Pixabay Music** — free, no copyright

### Audio Plan

| Timestamp | What happens |
|-----------|--------------|
| 0:00 | Music starts — medium volume (70%) |
| 0:03 | Beat drop on first ingredient reveal |
| 0:08 | Energy builds through cooking section |
| 0:15 | Beat drop on oven reveal |
| 0:18 | ASMR crunch sound BOOSTED — music ducks to 40% |
| 0:22 | Music rises back, fades to end |

### ASMR Sound Guide
1. Record the crunch separately in a quiet room (close your fridge, turn off fans)
2. Hold phone 5cm from the chip when biting
3. In CapCut: add as separate audio track, boost volume to 150%
4. Set music to 40% volume during the crunch moment

### Sound Effects to Add
- Boiling water: free from Freesound.org
- Oven click: free from Freesound.org
- Sizzle: free from Pixabay

### No Voiceover Note
This Reel intentionally uses NO voiceover — the visuals + text + ASMR sounds do all the work. This is a deliberate choice for food content: it works across all languages.`,

    section13_editingTimeline: `## COMPLETE EDITING WORKFLOW (CapCut)

**Total editing time estimate: 45-90 minutes**

### Step 1 — Project Setup
- Open CapCut → New Project
- Select all your footage → Create
- Change aspect ratio to 9:16
- Set canvas to white or transparent

### Step 2 — Import Your Music
- Tap Audio → Sounds
- Search for your chosen track
- Place it on the audio timeline at 0:00

### Step 3 — Arrange Raw Clips
- Place clips in order: Hook → Ingredients → Cooking → Reveal → CTA
- Don't trim yet — just get them in order

### Step 4 — Trim to Rough Cut
- Target total duration: 25-30 seconds
- Rough cut each section to approximate length

### Step 5 — Sync Cuts to Beat
- Tap Auto Sync or use Beat Markers
- Move each cut point to land on a beat marker
- The cooking section should have the fastest cuts (every 1-1.5 seconds)

### Step 6 — Speed Ramp the Crunch Shot
- Select the bite/crunch clip
- Speed → Curve → Hero preset
- Drag the lowest point to where the crunch happens
- Adjust so it goes from 1x → 0.3x → 1x

### Step 7 — Add Text Overlays
- Hook text: Bold white, bottom-center, fade-in animation
- Ingredient names: White text, pop-in animation, synced to each reveal
- Step numbers: Small, top-right corner

### Step 8 — Add ASMR Sound Layer
- Audio → Extract Audio from your crunch clip
- Boost to 150%
- Duck the music during crunch: select music → Volume → keyframe to 40% at crunch

### Step 9 — Color Grade
- Select all clips → Adjustment → Brightness +5 → Warmth +15 → Saturation +10 → Sharpness +15
- For the reveal shots: add extra Warmth +5

### Step 10 — Add CTA Text
- Final section: large text "Save this recipe! 🍝→🍟"
- Add bookmark animation (CapCut stickers)

### Step 11 — Final Review
- Watch 3x in full
- Check: audio levels, text timing, color consistency, cut timing

### Step 12 — Export
- Export → 1080p → 30fps → Save to Camera Roll`,

    section14_appSpecificGuide: `## APP-SPECIFIC EDITING GUIDE

---

### CAPCUT (Recommended for Beginners)

**1. New Project**
New Project → select all clips → tap the aspect ratio icon → 9:16

**2. Auto Beat Sync**
Tap your music track in the timeline → Auto Sync → CapCut places beat markers automatically

**3. Speed Ramp**
Select clip → Speed → Curve → choose "Hero" → move the center dip to where the crunch happens

**4. Text Overlay**
Text → Add Text → type your hook → Style tab → choose Bold template → Animation → Fade In

**5. Extract Audio for ASMR**
Select your crunch clip → tap the three dots → "Extract Audio" → this creates a separate audio layer you can boost

**6. Color Correction**
Select all clips with Command+A → Adjust → set: Brightness 5, Contrast 5, Warmth 15, Saturation 10, Clarity 20

**7. Export**
Export button (top right) → 1080p → 30fps → save

---

### PREMIERE PRO

**1. New Sequence**
File → New → Sequence → 1080 × 1920 (Vertical 9:16) at 30fps

**2. Import and Arrange**
Import all clips → drag to V1 timeline in order

**3. Beat Sync**
Use the audio waveform as guide → use C (razor) to cut at beat peaks → snap cuts to markers

**4. Speed Ramp**
Right-click clip → Speed/Duration → check "Ripple Edit" → OR use Timeline Keyframes:
Clip speed → add keyframes → lower the speed value at the crunch frame to 30%

**5. Text**
Essential Graphics panel → New Layer → Text → choose bold font (Neue Haas Grotesk or Impact)

**6. Color Grade**
Lumetri Color panel → Basic Correction: Exposure +0.2, Contrast +15, Warmth +15, Saturation 1.1

**7. Audio Mixing**
Essential Sound panel → select music clip → Loudness → Auto-Match → then keyframe volume to -8dB at crunch

**8. Export**
File → Export → Media → Format: H.264 → Preset: Match Source - High Bitrate → change to MP4 → 1080×1920`,

    section15_colorGrading: {
      style: "warm",
      exposure: "Slightly bright — +0.2 to +0.3 stops. Food looks better slightly exposed.",
      contrast: "Medium-high — +10 to +15. Makes colors pop and food look more 3D.",
      highlights: "Slightly recovered — -10. Prevents white surface from blowing out.",
      shadows: "Slightly lifted — +10. Opens up dark areas without losing contrast.",
      saturation: "Boosted — +10 to +15. Food needs more saturation to look appetizing on screen.",
      temperature: "Warm — +15 to +20 on the warmth slider. This is the signature food content look.",
      tint: "Slight green removal — -5. Removes the green tint that some kitchen lights add.",
      skinTones: "If hands appear, keep them natural — don't over-warm or they'll look orange.",
      beginnerSettings: "In CapCut: Adjust → Brightness +5 → Contrast +10 → Warmth +15 → Saturation +10 → Clarity +15. Apply to ALL clips at once.",
    },

    section16_textGraphics: `## TEXT AND GRAPHICS GUIDE

### Text Style Analysis
The Reel uses: **Bold white sans-serif text**, high contrast against dark food, minimal and large.

### Font Recommendations
- **CapCut:** Bold template → "Alfa Slab One" or "Impact" style
- **Premiere Pro:** Neue Haas Grotesk Bold or Bebas Neue
- **Canva:** Montserrat Black or Anton

### Text Hierarchy
1. **Hook text** (largest): "Wait... this is just pasta 🤯"
   - Size: Large (fills ~40% of screen width)
   - Position: Bottom third, centered
   - Animation: Fade in over 0.3s
   - Duration: Stays for entire first scene (3 seconds)

2. **Ingredient labels** (medium):
   - Size: Medium
   - Position: Below each ingredient
   - Animation: Pop in (scale 0→1 quickly)
   - Duration: 1.5 seconds each

3. **Step numbers** (small):
   - Size: Small
   - Position: Top-left corner
   - Animation: Simple appear
   - Color: White with slight shadow

### Original Text for Your Version
If recreating with a different food, here are hook text templates:
- "POV: I made chips from ___"
- "___ → chips. Yes, really 🤯"
- "You'll never guess what this is..."
- "The snack hack nobody's talking about 🍕"

### CTA Text
"Save this before you forget 🔖" or "Tag someone who needs this hack"`,

    section17_transitions: [
      {
        type: "Hard Cut",
        timestamp: "0:03, 0:05, 0:07, 0:12",
        purpose: "Standard beat-synced cut — keeps energy high",
        howItWorks: "One clip ends, next begins immediately with no transition effect",
        howToRecreate: "In CapCut: no transition needed — just cut on the beat. That's it.",
        beginnerAlternative: "This IS the beginner alternative — hard cuts are always the right choice for fast-paced food content",
      },
      {
        type: "Whip Pan",
        timestamp: "0:05",
        purpose: "Energetic scene change between ingredients",
        howItWorks: "A fast horizontal camera movement blur that connects two shots",
        howToRecreate: "Film: end shot 1 by quickly panning right. Start shot 2 with a quick pan right, then settle. In CapCut: add Transition → Whip. OR: quickly pan phone between shots.",
        beginnerAlternative: "Simple jump cut on the beat — achieves the same energy with no extra effort",
      },
      {
        type: "Speed Ramp Transition",
        timestamp: "0:18",
        purpose: "Creates dramatic emphasis on the crunch reveal",
        howItWorks: "The clip slows to 30% right as the crunch happens, then returns to normal speed",
        howToRecreate: "CapCut: Speed → Curve → Hero. Premiere: keyframe the speed value.",
        beginnerAlternative: "Just use the slow-motion clip on its own without the ramp",
      },
      {
        type: "Fade to Black",
        timestamp: "0:25",
        purpose: "Clean, professional ending — signals the video is over",
        howItWorks: "Opacity fades from 100% to 0% black over 1 second",
        howToRecreate: "CapCut: select last clip → Ending → Fade to Black. Premiere: Effects → Cross Dissolve to black solid.",
        beginnerAlternative: "Just let the last clip end — the platform loop handles it",
      },
    ],

    section18_exportSettings: [
      {
        platform: "Instagram Reels",
        resolution: "1080×1920",
        aspectRatio: "9:16",
        frameRate: "30fps",
        codec: "H.264",
        bitrate: "10-15 Mbps",
        audioFormat: "AAC 320kbps",
      },
      {
        platform: "TikTok",
        resolution: "1080×1920",
        aspectRatio: "9:16",
        frameRate: "30fps",
        codec: "H.264",
        bitrate: "10-15 Mbps",
        audioFormat: "AAC 320kbps",
      },
      {
        platform: "YouTube Shorts",
        resolution: "1080×1920",
        aspectRatio: "9:16",
        frameRate: "30fps",
        codec: "H.264",
        bitrate: "15-20 Mbps",
        audioFormat: "AAC 320kbps",
      },
    ],

    section19_publishing: {
      captionStructure: "Start with the hook (recreate curiosity). Add 2-3 lines of context/value. End with a direct CTA. Add line break. Add hashtags.",
      cta: "Save this for your next snack craving! 🔖 Tag a friend who needs this hack 👇",
      hashtags: [
        "#foodhack", "#pastatips", "#snackideas", "#easyrecipes", "#reelsfood",
        "#foodreel", "#cookinghacks", "#kitchentips", "#viralrecipe", "#foodtok",
        "#homecooking", "#quickrecipes", "#snacktime", "#foodlovers", "#recipeideas",
      ],
      title: "Turning Pasta Into Crispy Chips 🍝→🍟",
      coverIdea: "Use the overhead shot of the golden chips in the bowl. Add text overlay: 'PASTA → CHIPS' in bold. This image should make people curious without giving away the full recipe.",
      postingChecklist: [
        "✓ Watch final export on your phone before posting",
        "✓ Check audio plays correctly (sometimes export loses audio)",
        "✓ Caption is written and ready",
        "✓ Hashtags are added (15-20 max)",
        "✓ Cover/thumbnail selected",
        "✓ Posted between 6PM-9PM for maximum reach",
        "✓ Reply to first 10 comments within 1 hour",
      ],
    },

    section20_beginnerVersion: `## BEGINNER VERSION — Phone + CapCut Only

Everything you need: **your phone, CapCut, and a window.**

### Simplified Shot Plan
Instead of 9 shots, film just 5:
1. **Overhead** — finished chips in bowl (3 sec)
2. **Overhead** — all 3 ingredients laid out (2 sec)
3. **Side-on** — pasta going into boiling water (2 sec)
4. **Overhead** — draining and seasoning (3 sec)
5. **Close-up** — bite the chip, crunch (3 sec)

### Simplified Editing (CapCut)
1. Import all 5 clips
2. Pick a trending sound from the Reels audio library
3. Trim clips to fit the beat
4. Add text: hook text + step numbers
5. Adjust → Warmth +15, Saturation +10
6. Export 1080p

**Total time: 45-60 minutes**

### Beginner Tips
- Don't worry about perfect angles — clear and well-lit beats perfect
- Record every shot 3 times and use the best take
- Good lighting hides everything — shoot next to a window
- Keep it under 30 seconds — shorter performs better for food`,

    section21_professionalVersion: `## PROFESSIONAL VERSION — Camera + Premiere Pro

### Upgrade the Camera
Use a mirrorless camera (Sony ZV-E10, Sony A6400, Fujifilm X-T30) with a 35mm prime lens:
- ISO 100-400, aperture f/2.8, 1/60s shutter
- Shoot in C-Log or S-Log for maximum color grading latitude
- Record audio via external Rode VideoMicro

### Upgrade the Lighting
- Key light: Aputure AL-MX or similar small LED panel, 3200K
- Fill: Large white reflector or second panel at 50% power
- Background: Proper marble surface or ordered surface vinyl

### Advanced Techniques
- **Rack focus:** Start focused on the surface, pull focus to the chips
- **Micro slider:** 15cm slider movement on the crunch close-up
- **Depth of field:** At f/2.8, the background blurs beautifully
- **True slow motion:** Record at 120fps for genuine 4x slow-mo on the crunch

### Advanced Editing (Premiere)
- Import C-Log footage and apply camera LUT first
- Use Lumetri curves for precise color
- Add Film Grain overlay at 15% opacity for cinematic texture
- Mix audio in Audition for cleaner ASMR
- Color match all shots using the Lumetri Scopes

### Expected Result
The professional version should feel like a published food magazine — same concept, dramatically higher production quality.`,

    section22_timeBudget: `## TIME AND BUDGET ESTIMATE

### Time Investment
| Phase | Beginner | Experienced |
|-------|----------|-------------|
| Prep & setup | 30 min | 15 min |
| Filming | 30-45 min | 20 min |
| Editing | 60-90 min | 30 min |
| Export & publish | 10 min | 5 min |
| **Total** | **2-3 hours** | **1 hour** |

### Budget Options

**₹0 Setup (What you already have)**
- Your smartphone
- Natural window light
- DIY overhead mount (books + phone holder)
- CapCut (free)
- Cost: ₹0

**Low Budget (₹500–₹2000)**
- Overhead phone mount: ₹500–₹800
- Small LED panel (cheapest): ₹800–₹1500
- Marble vinyl sheet (1m): ₹300–₹500
- Total: ~₹1500–₹2800

**Professional Setup (₹15,000–₹50,000)**
- Entry mirrorless camera: ₹25,000–₹45,000
- Prime lens (35mm): ₹8,000–₹15,000
- LED panel (Aputure): ₹6,000–₹15,000
- Micro slider: ₹2,000–₹5,000
- Overhead arm: ₹2,000–₹4,000
- Total: ₹43,000–₹84,000`,

    section23_finalChecklist: [
      "☐ Concept finalized and hook text written",
      "☐ Shot list reviewed and understood",
      "☐ All ingredients prepared and ready",
      "☐ Background/surface is clean and correct color",
      "☐ Phone is fully charged (100%)",
      "☐ Phone storage cleared (at least 5GB free for 4K)",
      "☐ Overhead mount set up and stable",
      "☐ Lighting tested — no harsh shadows on food",
      "☐ Music track selected and downloaded",
      "☐ Test shot filmed and reviewed",
      "☐ All 5-9 shots filmed (minimum 3 takes each)",
      "☐ Crunch shot filmed at 120fps",
      "☐ Footage imported to CapCut",
      "☐ Rough cut arranged in order",
      "☐ Cuts synced to music beat",
      "☐ Speed ramp applied to crunch shot",
      "☐ All text overlays added and timed",
      "☐ ASMR sound layer boosted",
      "☐ Color grade applied (Warmth +15, Saturation +10)",
      "☐ Final review — watched 3 times",
      "☐ Exported at 1080p, 30fps, MP4",
      "☐ Watched export on phone",
      "☐ Caption and hashtags written",
      "☐ Cover/thumbnail selected",
      "☐ Posted at optimal time",
      "☐ Replied to first comments within 1 hour",
    ],
  },
};
