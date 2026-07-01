/**
 * deck-stage.js — oma-slide fixed-stage controller
 *
 * Provides a <deck-stage> Custom Element (Web Component) that:
 *   1. Scales the 1920×1080 stage uniformly to fit the viewport (letterbox / pillarbox).
 *   2. Handles keyboard navigation (←/→, Space/Shift+Space, PgUp/PgDn, Home/End, 0-9 digits).
 *   3. Handles touch swipe (horizontal) and mouse-wheel navigation.
 *   4. Reads speaker notes from <script type="application/json" id="speaker-notes">.
 *   5. Posts { type: "slideIndexChanged", index, total, note } to window.parent for a presenter view.
 *   6. Dispatches a native "slidechange" CustomEvent on <deck-stage>.
 *   7. Manages .active / .visible CSS classes (never display:none).
 *   8. Tags each slide with data-screen-label and data-om-validate on first connect.
 *   9. Provides clean @media print behaviour (removes transform so PDF prints at design size).
 *
 * Usage:
 *   <!-- Wrap .deck-viewport + .deck-stage in <deck-stage> -->
 *   <deck-stage>
 *     <div class="deck-viewport">
 *       <div class="deck-stage">
 *         <section class="slide" id="slide-01">...</section>
 *         <section class="slide" id="slide-02">...</section>
 *       </div>
 *     </div>
 *   </deck-stage>
 *
 *   <!-- Optional speaker notes (JSON object keyed by slide index, 0-based) -->
 *   <script type="application/json" id="speaker-notes">
 *     { "0": "Opening remarks...", "1": "Talk about the problem..." }
 *   </script>
 *
 * No build step required. ES2022 class syntax; no external dependencies.
 *
 * Validator contract:
 *   Each slide receives:
 *     data-screen-label="Slide N / M"
 *     data-om-validate="no_overflowing_text,no_overlapping_text,slide_sized_text"
 *   These attributes are read by `oma slide validate` (puppeteer-core) to locate
 *   slides and know which checks to run on each one.
 */

"use strict";

/* ─── Constants ──────────────────────────────────────────────────────── */
const STAGE_W = 1920;
const STAGE_H = 1080;
const VALIDATOR_CHECKS = "no_overflowing_text,no_overlapping_text,slide_sized_text";

/* ─── Utility: clamp ─────────────────────────────────────────────────── */
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

/* ─── Utility: debounce ──────────────────────────────────────────────── */
function debounce(fn, ms) {
  let timer = null;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

/* ─── Utility: parse speaker notes ──────────────────────────────────── */
function parseSpeakerNotes() {
  const el = document.getElementById("speaker-notes");
  if (!el) return {};
  try {
    const parsed = JSON.parse(el.textContent || "{}");
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      return parsed;
    }
  } catch {
    /* malformed JSON — silently ignore */
  }
  return {};
}

/* ─── <deck-stage> Web Component ────────────────────────────────────── */
class DeckStage extends HTMLElement {
  /* ── Internal state ── */
  #slides = [];
  #currentIndex = 0;
  #notes = {};
  #stageEl = null;
  #resizeObserver = null;
  #touchStartX = 0;
  #touchStartY = 0;
  #touchStartTime = 0;
  #wheelLocked = false;

  /* ── Lifecycle ── */

  connectedCallback() {
    /*
     * The element may be upgraded BEFORE its children are parsed — this
     * happens whenever deck-stage.js is loaded in <head> without defer:
     * customElements.define() runs during parsing and connectedCallback
     * fires at the <deck-stage> start tag, when .deck-stage / .slide do
     * not exist yet. Querying then returns null, init bails, and slide 0
     * never gets .active → blank render (and export PNG/PDF come out blank).
     *
     * Detect that case (no .deck-stage yet AND the document is still
     * parsing) and defer init until DOMContentLoaded, when the full
     * subtree is guaranteed to exist.
     */
    if (!this.querySelector(".deck-stage") && document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", this.#init, { once: true });
      return;
    }
    this.#init();
  }

  #init = () => {
    this.#stageEl = this.querySelector(".deck-stage");
    if (!this.#stageEl) {
      console.warn("[deck-stage] .deck-stage element not found inside <deck-stage>.");
      return;
    }

    this.#slides = Array.from(this.#stageEl.querySelectorAll(".slide"));
    if (this.#slides.length === 0) {
      console.warn("[deck-stage] No .slide elements found.");
      return;
    }

    this.#notes = parseSpeakerNotes();

    this.#annotateSlides();
    this.#bindKeyboard();
    this.#bindTouch();
    this.#bindWheel();
    this.#startResizeObserver();
    this.#scaleStage();
    this.#goTo(0, /* initial */ true);
  };

  disconnectedCallback() {
    this.#resizeObserver?.disconnect();
    document.removeEventListener("DOMContentLoaded", this.#init);
    document.removeEventListener("keydown", this.#onKeyDown);
    window.removeEventListener("beforeprint", this.#onBeforePrint);
    window.removeEventListener("afterprint", this.#onAfterPrint);
  }

  /* ── Public API ── */

  /** Navigate to a specific slide index (0-based). */
  goTo(index) {
    this.#goTo(clamp(index, 0, this.#slides.length - 1));
  }

  /** Navigate one slide forward. */
  next() {
    if (this.#currentIndex < this.#slides.length - 1) {
      this.#goTo(this.#currentIndex + 1);
    }
  }

  /** Navigate one slide backward. */
  prev() {
    if (this.#currentIndex > 0) {
      this.#goTo(this.#currentIndex - 1);
    }
  }

  get currentIndex() {
    return this.#currentIndex;
  }

  get total() {
    return this.#slides.length;
  }

  /* ── Private: slide annotation ── */

  /**
   * Tags each slide with data-screen-label and data-om-validate.
   * Called once on connect. Required by the validator contract.
   */
  #annotateSlides() {
    const total = this.#slides.length;
    this.#slides.forEach((slide, i) => {
      // data-screen-label: human-readable label consumed by the validator
      if (!slide.dataset.screenLabel) {
        slide.dataset.screenLabel = `Slide ${i + 1} / ${total}`;
      }
      // data-om-validate: comma-separated list of checks the validator runs
      if (!slide.dataset.omValidate) {
        slide.dataset.omValidate = VALIDATOR_CHECKS;
      }
    });
  }

  /* ── Private: navigation ── */

  #goTo(index, initial = false) {
    const prev = this.#currentIndex;
    const next = clamp(index, 0, this.#slides.length - 1);

    if (prev === next && !initial) return;

    /* Outgoing: drop .active, briefly keep .visible so it fades out */
    if (!initial && prev !== next) {
      const outgoing = this.#slides[prev];
      outgoing.classList.remove("active");
      outgoing.classList.add("visible");

      /* Remove .visible after the CSS transition (400 ms guard) */
      const outgoingEl = outgoing;
      const clearVisible = () => {
        outgoingEl.classList.remove("visible");
        outgoingEl.removeEventListener("transitionend", clearVisible);
      };
      outgoing.addEventListener("transitionend", clearVisible, { once: true });
      setTimeout(() => outgoing.classList.remove("visible"), 400);
    }

    /* Incoming: make active */
    const incoming = this.#slides[next];
    incoming.classList.remove("visible");
    incoming.classList.add("active");
    this.#currentIndex = next;

    /* Update counter element if present */
    const counter = document.querySelector(".deck-counter");
    if (counter) {
      counter.textContent = `${next + 1} / ${this.#slides.length}`;
    }

    /* Dispatch slidechange event */
    const note = this.#notes[String(next)] ?? this.#notes[next] ?? "";
    const payload = { index: next, total: this.#slides.length, note };

    this.dispatchEvent(
      new CustomEvent("slidechange", { detail: payload, bubbles: true, composed: true })
    );

    /* Post message to parent frame for presenter view */
    if (window.parent && window.parent !== window) {
      window.parent.postMessage(
        { type: "slideIndexChanged", ...payload },
        "*"
      );
    }
  }

  /* ── Private: scaling ── */

  /**
   * Computes scale = Math.min(viewportW / 1920, viewportH / 1080)
   * and applies transform: scale(s) + translate to centre the stage.
   *
   * The stage is positioned so (scaledW, scaledH) is centred in the viewport.
   * We achieve this by:
   *   left: (vw - 1920) / 2   (works for both scale < 1 and scale > 1)
   *   top:  (vh - 1080) / 2
   * and then transform: scale(s) with transform-origin: top left.
   */
  #scaleStage() {
    if (!this.#stageEl) return;
    if (window.matchMedia("print").matches) return;

    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const scale = Math.min(vw / STAGE_W, vh / STAGE_H);

    const scaledW = STAGE_W * scale;
    const scaledH = STAGE_H * scale;

    const offsetLeft = (vw - scaledW) / 2;
    const offsetTop = (vh - scaledH) / 2;

    this.#stageEl.style.transform = `scale(${scale})`;
    this.#stageEl.style.left = `${offsetLeft}px`;
    this.#stageEl.style.top = `${offsetTop}px`;
    this.#stageEl.style.position = "absolute";
  }

  /* ── Private: resize observer ── */

  #startResizeObserver() {
    const debouncedScale = debounce(() => this.#scaleStage(), 50);

    this.#resizeObserver = new ResizeObserver(debouncedScale);
    this.#resizeObserver.observe(document.documentElement);

    /* Print: remove transform before print so the browser lays out at design size */
    window.addEventListener("beforeprint", this.#onBeforePrint);
    window.addEventListener("afterprint", this.#onAfterPrint);
  }

  #onBeforePrint = () => {
    if (!this.#stageEl) return;
    this.#stageEl.style.transform = "none";
    this.#stageEl.style.left = "";
    this.#stageEl.style.top = "";
    this.#stageEl.style.position = "";
  };

  #onAfterPrint = () => {
    this.#scaleStage();
  };

  /* ── Private: keyboard navigation ── */

  #bindKeyboard() {
    document.addEventListener("keydown", this.#onKeyDown);
  }

  #onKeyDown = (e) => {
    /* Skip if focus is in an input/textarea/select to not block typing */
    const tag = document.activeElement?.tagName?.toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;

    switch (e.key) {
      case "ArrowRight":
      case "ArrowDown":
      case " ":
      case "PageDown":
        e.preventDefault();
        if (e.shiftKey && e.key === " ") {
          this.prev();
        } else {
          this.next();
        }
        break;

      case "ArrowLeft":
      case "ArrowUp":
      case "PageUp":
        e.preventDefault();
        this.prev();
        break;

      case "Home":
        e.preventDefault();
        this.#goTo(0);
        break;

      case "End":
        e.preventDefault();
        this.#goTo(this.#slides.length - 1);
        break;

      default:
        /* Number keys 1-9 and 0 (for slide 10) */
        if (e.key >= "0" && e.key <= "9") {
          const n = e.key === "0" ? 10 : parseInt(e.key, 10);
          this.#goTo(n - 1);
        }
        break;
    }
  };

  /* ── Private: touch / swipe ── */

  #bindTouch() {
    this.addEventListener("touchstart", this.#onTouchStart, { passive: true });
    this.addEventListener("touchend", this.#onTouchEnd, { passive: true });
  }

  #onTouchStart = (e) => {
    const touch = e.changedTouches[0];
    this.#touchStartX = touch.clientX;
    this.#touchStartY = touch.clientY;
    this.#touchStartTime = Date.now();
  };

  #onTouchEnd = (e) => {
    const touch = e.changedTouches[0];
    const dx = touch.clientX - this.#touchStartX;
    const dy = touch.clientY - this.#touchStartY;
    const dt = Date.now() - this.#touchStartTime;

    /* Minimum swipe: 40px horizontal, less than 300ms, more horizontal than vertical */
    if (Math.abs(dx) > 40 && Math.abs(dx) > Math.abs(dy) * 1.5 && dt < 300) {
      if (dx < 0) {
        this.next();
      } else {
        this.prev();
      }
    }
  };

  /* ── Private: mouse-wheel navigation ── */

  #bindWheel() {
    this.addEventListener("wheel", this.#onWheel, { passive: false });
  }

  /**
   * Wheel navigation with a short lock to prevent multi-slide jumping on
   * trackpads that fire many small delta events per gesture.
   */
  #onWheel = (e) => {
    e.preventDefault();
    if (this.#wheelLocked) return;

    if (e.deltaY > 0 || e.deltaX > 0) {
      this.next();
    } else if (e.deltaY < 0 || e.deltaX < 0) {
      this.prev();
    }

    /* Lock for 600 ms to absorb momentum scrolling */
    this.#wheelLocked = true;
    setTimeout(() => {
      this.#wheelLocked = false;
    }, 600);
  };
}

/* ─── Registration ───────────────────────────────────────────────────── */
customElements.define("deck-stage", DeckStage);

/* ─── Convenience: auto-init for plain HTML decks ───────────────────── */
/*
 * If the page does NOT use <deck-stage> as a wrapper element but instead
 * has a bare .deck-viewport / .deck-stage structure, auto-wrap it.
 * This allows quick standalone previews without changing the HTML.
 */
document.addEventListener("DOMContentLoaded", () => {
  /* If a <deck-stage> is already in the DOM, do nothing */
  if (document.querySelector("deck-stage")) return;

  const viewport = document.querySelector(".deck-viewport");
  if (!viewport) return;

  const wrapper = document.createElement("deck-stage");
  viewport.parentNode.insertBefore(wrapper, viewport);
  wrapper.appendChild(viewport);
});

/* ─── Presenter-view helper ──────────────────────────────────────────── */
/*
 * When this page is loaded inside an <iframe> for a presenter view,
 * the parent frame can post { type: "navigateTo", index: N } to control it.
 */
window.addEventListener("message", (e) => {
  if (!e.data || typeof e.data !== "object") return;
  if (e.data.type !== "navigateTo") return;

  const deckEl = document.querySelector("deck-stage");
  if (deckEl && typeof deckEl.goTo === "function") {
    deckEl.goTo(Number(e.data.index));
  }
});
