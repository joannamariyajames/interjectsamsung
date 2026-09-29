import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import {
  ArrowRight,
  AudioLines,
  Car,
  Hand,
  LogOut,
  MapPin,
  Mic,
  RotateCcw,
  Route,
  ShieldCheck,
  Sparkles,
  Workflow,
} from "lucide-react";
import { Logo } from "~/components/Logo";
import { useAuth } from "~/lib/auth";
import { cn } from "~/lib/utils";

const go = (hash: string) => {
  window.location.hash = hash;
};

/* ------------------------------------------------------------ hero demo */

type Beat =
  | { who: "user"; text: string; cut?: boolean }
  | { who: "agent"; text: string; cut?: boolean }
  | { who: "event"; text: string; tone: "accent" | "live" | "gold" };

const SCRIPT: Beat[] = [
  { who: "user", text: "Book the 6 pm flight to Mumbai, um..." },
  { who: "agent", text: "Checking flights to Mumbai for this evening, the 6 pm leaves from", cut: true },
  { who: "event", text: "you cut in - speech stopped", tone: "accent" },
  { who: "user", text: "Actually no, make it Pune." },
  { who: "event", text: "Mumbai search marked stale, never booked", tone: "gold" },
  { who: "agent", text: "Switching to Pune. Nothing was booked for Mumbai - here's the 6 pm to Pune." },
  { who: "event", text: "booked once - repeat requests are blocked", tone: "live" },
];

function HeroDemo() {
  const [shown, setShown] = useState(3);
  useEffect(() => {
    const timer = setInterval(() => setShown((n) => (n >= SCRIPT.length + 2 ? 3 : n + 1)), 1500);
    return () => clearInterval(timer);
  }, []);
  const beats = SCRIPT.slice(0, Math.min(shown, SCRIPT.length));

  return (
    <div className="relative">
      <div className="absolute inset-0 -z-10 rounded-[32px] sm:-inset-6 bg-[radial-gradient(closest-side,color-mix(in_oklab,var(--accent)_22%,transparent),transparent)] blur-2xl" />
      <div className="rounded-[var(--radius-panel)] border border-line/80 bg-panel/85 p-5 shadow-[var(--shadow-lift)] backdrop-blur-xl">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-muted">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-live opacity-60" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-live" />
            </span>
            Live session
          </div>
          <span className="eyebrow">full duplex</span>
        </div>
        <div className="flex h-[330px] flex-col justify-end gap-2.5 overflow-hidden">
          <AnimatePresence initial={false}>
            {beats.map((beat, i) =>
              beat.who === "event" ? (
                <motion.div
                  key={i}
                  layout
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className={cn(
                    "mx-auto flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px]",
                    beat.tone === "accent" && "border-accent/40 bg-accent-soft/50 text-accent",
                    beat.tone === "gold" && "border-gold/40 bg-gold-soft/50 text-gold",
                    beat.tone === "live" && "border-live/40 bg-live-soft/50 text-live",
                  )}
                >
                  {beat.tone === "accent" ? <Hand size={11} /> : beat.tone === "gold" ? <RotateCcw size={11} /> : <ShieldCheck size={11} />}
                  {beat.text}
                </motion.div>
              ) : (
                <motion.div
                  key={i}
                  layout
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className={cn(
                    "max-w-[86%] rounded-[var(--radius-card)] px-3.5 py-2.5 text-[13px] leading-relaxed",
                    beat.who === "user"
                      ? "ml-auto rounded-br-[4px] bg-accent text-accent-foreground"
                      : "rounded-tl-[4px] border border-line/80 bg-surface/80",
                    beat.cut && "border-dashed border-accent/50",
                  )}
                >
                  {beat.text}
                  {beat.cut ? <span className="ml-1 font-mono text-[11px] text-accent">//</span> : null}
                </motion.div>
              ),
            )}
          </AnimatePresence>
        </div>
        <div className="mt-4 flex items-center gap-3 rounded-[var(--radius-pill)] border border-line/80 bg-surface/60 px-3 py-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-live text-background">
            <Mic size={15} />
          </span>
          <span className="flex items-end gap-[3px] text-live" aria-hidden>
            {[0, 0.1, 0.2, 0.3, 0.15, 0.25].map((d, i) => (
              <span key={i} className="wave-bar h-4 w-[3px] rounded-full bg-current" style={{ animationDelay: `${d}s` }} />
            ))}
          </span>
          <span className="text-xs text-muted">Listening - talk over it any time</span>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------- sections */

const PILLARS = [
  {
    icon: AudioLines,
    title: "Stays responsive",
    body: "It listens while it talks. Say anything and it stops mid-word - then \"go on\" picks up from exactly what you heard, not from the top.",
  },
  {
    icon: Workflow,
    title: "Works in the background",
    body: "Lookups and tools start while you are still speaking, so the conversation never blocks on a slow API or a long search.",
  },
  {
    icon: ShieldCheck,
    title: "Recovers cleanly",
    body: "Change your mind mid-sentence and the stale intent is dropped before it acts. A booking or an order is never performed twice.",
  },
];

const STEPS = [
  { n: "01", title: "Hear it early", body: "Partial speech streams in as you talk; retrieval and planning start before you finish." },
  { n: "02", title: "Track what changed", body: "Every fact you give is versioned. A correction marks dependent work stale instead of letting it land." },
  { n: "03", title: "Gate every action", body: "State-changing calls pass one guard: current inputs only, and at most once." },
  { n: "04", title: "Resume, don't restart", body: "Cut it off and the unfinished answer is checkpointed. \"Go on\" continues from what you actually heard." },
];

const STATS = [
  { value: "69%", label: "strict pass rate", note: "Full-Duplex-Bench v3, text replay of all 100 recordings" },
  { value: "97%", label: "right tool chosen", note: "same run, official evaluator, exact matching" },
  { value: "12", label: "benchmark tools", note: "travel, finance, housing and shopping" },
  { value: "0", label: "API keys for voice", note: "speech runs in your browser" },
];

export function Landing({ onAuth }: { onAuth: (mode: "login" | "signup") => void }) {
  const { status, user, logout } = useAuth();
  const [scrolled, setScrolled] = useState(false);
  const signedIn = status === "user";
  const openApp = (hash = "app") => (signedIn ? go(hash) : onAuth("signup"));

  return (
    <div
      className="scrollable h-full"
      onScroll={(e) => setScrolled((e.target as HTMLDivElement).scrollTop > 8)}
    >
      {/* ---------------------------------------------------------- nav */}
      <header
        className={cn(
          "sticky top-0 z-40 transition-[background-color,border-color,backdrop-filter]",
          scrolled ? "border-b border-line/70 bg-background/70 backdrop-blur-xl" : "border-b border-transparent",
        )}
      >
        <nav className="mx-auto flex h-16 max-w-6xl items-center gap-6 px-5">
          <a href="#" className="flex items-center gap-2.5" aria-label="Interject home">
            <Logo size={30} />
            <span className="display text-[1.35rem] leading-none">Interject</span>
          </a>
          <div className="hidden items-center gap-6 text-sm text-muted md:flex">
            <a href="#features" onClick={(e) => { e.preventDefault(); document.getElementById("features")?.scrollIntoView({ behavior: "smooth" }); }} className="hover:text-foreground">Features</a>
            <a href="#how" onClick={(e) => { e.preventDefault(); document.getElementById("how")?.scrollIntoView({ behavior: "smooth" }); }} className="hover:text-foreground">How it works</a>
            <a href="#drive-section" onClick={(e) => { e.preventDefault(); document.getElementById("drive-section")?.scrollIntoView({ behavior: "smooth" }); }} className="hover:text-foreground">Drive</a>
          </div>
          <div className="ml-auto flex items-center gap-2">
            {signedIn ? (
              <>
                <span className="hidden text-xs text-muted sm:inline">Hi, {user?.name.split(" ")[0]}</span>
                <button
                  onClick={() => go("app")}
                  className="inline-flex h-9 items-center gap-1.5 rounded-full bg-accent px-4 text-sm font-medium text-accent-foreground hover:brightness-110"
                >
                  Open app <ArrowRight size={14} />
                </button>
                <button
                  onClick={() => void logout()}
                  aria-label="Log out"
                  title="Log out"
                  className="flex h-9 w-9 items-center justify-center rounded-full text-muted hover:bg-subtle hover:text-foreground"
                >
                  <LogOut size={15} />
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={() => onAuth("login")}
                  className="h-9 rounded-full px-4 text-sm font-medium text-muted hover:bg-subtle hover:text-foreground"
                >
                  Log in
                </button>
                <button
                  onClick={() => onAuth("signup")}
                  className="h-9 rounded-full bg-accent px-4 text-sm font-medium text-accent-foreground shadow-[var(--shadow-soft)] hover:brightness-110"
                >
                  Sign up
                </button>
              </>
            )}
          </div>
        </nav>
      </header>

      {/* --------------------------------------------------------- hero */}
      <section className="mx-auto grid max-w-6xl items-center gap-12 px-5 pb-20 pt-12 md:pt-20 lg:grid-cols-[1.05fr_1fr]">
        <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <span className="inline-flex items-center gap-2 rounded-full border border-line/80 bg-surface/60 px-3 py-1 text-xs text-muted">
            <Sparkles size={12} className="text-gold" /> An interruptible, real-time voice agent
          </span>
          <h1 className="display mt-6 text-[3.1rem] leading-[1.02] tracking-tight sm:text-[4.2rem]">
            Talk over it.
            <br />
            <span className="italic text-accent">It keeps up.</span>
          </h1>
          <p className="mt-6 max-w-xl text-[15px] leading-relaxed text-muted sm:text-base">
            Real people hesitate, correct themselves and cut in. Interject listens while it speaks, stops the
            instant you do, drops what you took back - and never does the same thing twice.
          </p>
          <div className="mt-9 flex flex-wrap items-center gap-3">
            <button
              onClick={() => openApp()}
              className="inline-flex h-12 items-center gap-2 rounded-full bg-accent px-6 text-sm font-medium text-accent-foreground shadow-[var(--shadow-lift)] transition hover:brightness-110"
            >
              {signedIn ? "Open your session" : "Get started free"} <ArrowRight size={16} />
            </button>
            {!signedIn ? (
              <button
                onClick={() => onAuth("login")}
                className="inline-flex h-12 items-center rounded-full border border-line/80 px-6 text-sm font-medium hover:bg-subtle"
              >
                I have an account
              </button>
            ) : null}
          </div>
          <p className="mt-4 text-xs text-muted">Email and password - Gmail works. Chrome or Edge for voice.</p>
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.1 }}>
          <HeroDemo />
        </motion.div>
      </section>

      {/* -------------------------------------------------------- stats */}
      <section className="mx-auto max-w-6xl px-5">
        <div className="grid grid-cols-2 gap-px overflow-hidden rounded-[var(--radius-panel)] border border-line/70 bg-line/70 md:grid-cols-4">
          {STATS.map((s) => (
            <div key={s.label} className="bg-panel/90 p-6">
              <div className="display text-[2.6rem] leading-none text-foreground">{s.value}</div>
              <div className="mt-2 text-sm font-medium">{s.label}</div>
              <div className="mt-1 text-[11px] leading-snug text-muted">{s.note}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ----------------------------------------------------- features */}
      <section id="features" className="mx-auto max-w-6xl scroll-mt-20 px-5 py-24">
        <p className="eyebrow">What makes it different</p>
        <h2 className="display mt-3 max-w-2xl text-[2.4rem] leading-tight sm:text-[2.9rem]">
          Built for how people actually talk.
        </h2>
        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {PILLARS.map(({ icon: Icon, title, body }, i) => (
            <motion.div
              key={title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ delay: i * 0.08 }}
              className="group rounded-[var(--radius-panel)] border border-line/70 bg-panel/70 p-6 transition-colors hover:border-accent/40"
            >
              <span className="flex h-11 w-11 items-center justify-center rounded-[var(--radius-item)] bg-accent-soft text-accent">
                <Icon size={20} />
              </span>
              <h3 className="mt-5 text-lg font-medium">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">{body}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* -------------------------------------------------- how it works */}
      <section id="how" className="border-y border-line/60 bg-panel/40 scroll-mt-16">
        <div className="mx-auto max-w-6xl px-5 py-24">
          <p className="eyebrow">Under the hood</p>
          <h2 className="display mt-3 text-[2.4rem] leading-tight sm:text-[2.9rem]">Four steps, every turn.</h2>
          <div className="mt-12 grid gap-8 md:grid-cols-4">
            {STEPS.map((step) => (
              <div key={step.n} className="relative">
                <div className="font-mono text-xs text-gold">{step.n}</div>
                <div className="mt-3 h-px w-full bg-gradient-to-r from-accent/60 via-line to-transparent" />
                <h3 className="mt-4 font-medium">{step.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{step.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------- drive */}
      <section id="drive-section" className="mx-auto grid max-w-6xl scroll-mt-20 items-center gap-12 px-5 py-24 lg:grid-cols-2">
        <div>
          <p className="eyebrow">Beyond the benchmark</p>
          <h2 className="display mt-3 text-[2.4rem] leading-tight sm:text-[2.9rem]">Drive: an in-car assistant for India.</h2>
          <p className="mt-5 text-[15px] leading-relaxed text-muted">
            Plan a route to any of 7,000 cities and towns, change your mind mid-sentence, add a fuel stop, ask how
            long it'll take - hands-free. It starts planning before you finish speaking and starts guidance exactly
            once.
          </p>
          <ul className="mt-6 space-y-2.5 text-sm">
            {["Works offline - no map service, no API key", "Say \"hold on\" to pause, \"go on\" to continue", "Stale routes are discarded, never guided"].map((t) => (
              <li key={t} className="flex items-center gap-2.5">
                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-live-soft text-live">
                  <ShieldCheck size={12} />
                </span>
                {t}
              </li>
            ))}
          </ul>
          <button
            onClick={() => openApp("drive")}
            className="mt-8 inline-flex h-11 items-center gap-2 rounded-full border border-line/80 px-5 text-sm font-medium hover:bg-subtle"
          >
            <Car size={15} /> Try Drive
          </button>
        </div>
        <div className="rounded-[var(--radius-panel)] border border-line/70 bg-panel/80 p-6 shadow-[var(--shadow-soft)]">
          <div className="flex items-center justify-between text-xs text-muted">
            <span className="eyebrow">Trip</span>
            <span className="rounded-full bg-live-soft px-2 py-0.5 text-live">Guidance active</span>
          </div>
          <div className="mt-5 flex items-start gap-3">
            <ol className="relative space-y-6 pl-7 text-sm before:absolute before:bottom-2 before:left-[5px] before:top-2 before:w-px before:bg-line">
              <li className="relative leading-5">
                <span className="absolute -left-7 top-1 h-3 w-3 rounded-full border-2 border-muted bg-panel" />
                Delhi
              </li>
              <li className="relative leading-5 text-muted">
                <span className="absolute -left-7 top-1 h-3 w-3 rounded-full bg-gold" />
                Fuel stop near Alwar
              </li>
              <li className="relative font-medium leading-5">
                <MapPin size={15} className="absolute -left-[29px] top-0.5 bg-panel text-accent" />
                Jaipur, Rajasthan
              </li>
            </ol>
            <div className="ml-auto text-right">
              <div className="display text-[2rem] leading-none">292 km</div>
              <div className="mt-1 text-xs text-muted">about 5 h 10 min</div>
            </div>
          </div>
          <div className="mt-6 flex flex-wrap gap-2 text-[11px]">
            <span className="flex items-center gap-1 rounded-full border border-gold/40 bg-gold-soft/50 px-2.5 py-1 text-gold">
              <RotateCcw size={11} /> Agra dropped - you corrected it
            </span>
            <span className="flex items-center gap-1 rounded-full border border-line/80 px-2.5 py-1 text-muted">
              <Route size={11} /> planned while you spoke
            </span>
          </div>
        </div>
      </section>

      {/* ---------------------------------------------------------- CTA */}
      <section className="mx-auto max-w-6xl px-5 pb-24">
        <div className="relative overflow-hidden rounded-[28px] border border-line/70 bg-panel/80 px-8 py-14 text-center">
          <div className="absolute inset-0 -z-10 bg-[radial-gradient(600px_220px_at_50%_0%,color-mix(in_oklab,var(--accent)_20%,transparent),transparent)]" />
          <h2 className="display text-[2.4rem] leading-tight sm:text-[3rem]">Go on - interrupt it.</h2>
          <p className="mx-auto mt-4 max-w-md text-sm text-muted">
            Create an account, turn on the mic, and try cutting it off mid-sentence.
          </p>
          <button
            onClick={() => openApp()}
            className="mt-8 inline-flex h-12 items-center gap-2 rounded-full bg-accent px-7 text-sm font-medium text-accent-foreground shadow-[var(--shadow-lift)] hover:brightness-110"
          >
            {signedIn ? "Open your session" : "Create your free account"} <ArrowRight size={16} />
          </button>
        </div>
      </section>

      <footer className="border-t border-line/60">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-5 py-8 text-xs text-muted sm:flex-row">
          <div className="flex items-center gap-2">
            <Logo size={20} /> Interject - Samsung Hackathon, Theme 05: Interruptible Real-Time Agents
          </div>
          <div>MIT licensed</div>
        </div>
      </footer>
    </div>
  );
}
