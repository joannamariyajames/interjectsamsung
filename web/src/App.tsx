import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Car, CloudOff, LogOut, Menu, PanelRightClose, PanelRightOpen, Wifi, WifiOff, X } from "lucide-react";
import { Logo } from "~/components/Logo";
import { Badge, Button, Panel } from "~/components/ui/primitives";
import { Composer } from "~/components/Composer";
import { MindRail } from "~/components/MindRail";
import { Sidebar } from "~/components/Sidebar";
import { Transcript } from "~/components/Transcript";
import { DriveView } from "~/drive/DriveView";
import { AuthDialog } from "~/landing/AuthDialog";
import { Landing } from "~/landing/Landing";
import { useAuth } from "~/lib/auth";
import { useSession } from "~/store/session";

type Route = "landing" | "login" | "signup" | "app" | "drive";

const routeFromHash = (): Route => {
  const hash = window.location.hash.replace(/^#\/?/, "");
  return hash === "login" || hash === "signup" || hash === "app" || hash === "drive" ? hash : "landing";
};
const go = (hash: string) => {
  window.location.hash = hash;
};

function UserMenu() {
  const { user, logout } = useAuth();
  if (!user) return null;
  return (
    <div className="flex items-center gap-1.5">
      <span
        title={user.email}
        className="flex h-8 w-8 items-center justify-center rounded-full bg-accent-soft text-xs font-semibold text-accent"
      >
        {user.name.trim().charAt(0).toUpperCase() || "?"}
      </span>
      <Button variant="ghost" size="icon" aria-label="Log out" title="Log out" onClick={() => void logout()}>
        <LogOut size={15} />
      </Button>
    </div>
  );
}

function ConnectionBadge() {
  const status = useSession((s) => s.status);
  if (status === "open") {
    return (
      <Badge tone="live">
        <Wifi size={11} /> connected
      </Badge>
    );
  }
  return (
    <Badge tone={status === "connecting" ? "warn" : "danger"}>
      <WifiOff size={11} /> {status === "connecting" ? "connecting" : "reconnecting"}
    </Badge>
  );
}

/** The backend runs without a model key: say what that means before anyone asks. */
function OfflineNotice() {
  const provider = useSession((s) => s.provider);
  if (provider !== "local-deterministic") return null;
  return (
    <div
      role="status"
      className="flex items-start gap-2 border-b border-line/70 bg-warn-soft/40 px-5 py-2 text-xs leading-snug text-warn"
    >
      <CloudOff size={13} className="mt-px shrink-0" />
      <p>
        <span className="font-semibold">Offline demo mode</span> - answers come only from a fictional demo
        agency&rsquo;s knowledge base.
        <span className="hidden sm:inline">
          {" "}No AI model is connected; for general questions, start the backend with a Groq key (
          <code className="font-mono">scripts/start-backend-groq.ps1</code>).
        </span>
      </p>
    </div>
  );
}

/**
 * Routes: the public landing page at "/", with log-in and sign-up dialogs
 * (#login, #signup); the assistant (#app) and the in-car Drive extension
 * (#drive) need a session - without one the server refuses their sockets too.
 */
export default function App() {
  const { status, refresh } = useAuth();
  const [route, setRoute] = useState<Route>(routeFromHash);
  const [next, setNext] = useState<Route>("app");

  useEffect(() => {
    void refresh();
    const onHash = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, [refresh]);

  useEffect(() => {
    if (status === "guest" && (route === "app" || route === "drive")) {
      setNext(route); // come back here after logging in
      go("login");
    } else if (status === "user" && (route === "login" || route === "signup")) {
      go(next);
    }
  }, [status, route, next]);

  if (status === "loading") {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="animate-pulse">
          <Logo size={44} />
        </div>
      </div>
    );
  }
  if (status === "user" && route === "app") return <AssistantApp />;
  if (status === "user" && route === "drive") return <DriveView onExit={() => go("app")} />;

  return (
    <>
      <Landing onAuth={(mode) => go(mode)} />
      <AnimatePresence>
        {(route === "login" || route === "signup") && status === "guest" ? (
          <AuthDialog mode={route} onMode={(mode) => go(mode)} onClose={() => go("")} onDone={() => go(next)} />
        ) : null}
      </AnimatePresence>
    </>
  );
}

function AssistantApp() {
  const connect = useSession((s) => s.connect);
  const disconnect = useSession((s) => s.disconnect);
  const stage = useSession((s) => s.stage);
  const [railOpen, setRailOpen] = useState(() => window.innerWidth >= 1280);
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  // The drawer only exists below md; if the window grows past it, drop the
  // overlay rather than leaving two sidebars on screen.
  useEffect(() => {
    const query = window.matchMedia("(min-width: 768px)");
    const onChange = () => query.matches && setNavOpen(false);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  return (
    <div className="flex h-full gap-3 p-3">
      <div className="hidden md:flex">
        <Sidebar />
      </div>

      {/* Narrow windows reach the same controls through a slide-over. Escape is
          reserved for barging in, so this closes on the backdrop or the X. */}
      <AnimatePresence>
        {navOpen ? (
          <motion.div
            className="fixed inset-0 z-50 flex md:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <button
              aria-label="Close menu"
              className="absolute inset-0 bg-black/55 backdrop-blur-[2px]"
              onClick={() => setNavOpen(false)}
            />
            <motion.div
              className="relative z-10 h-full max-w-[86vw] p-3"
              initial={{ x: -24 }}
              animate={{ x: 0 }}
              exit={{ x: -24 }}
              transition={{ type: "spring", stiffness: 380, damping: 34 }}
            >
              <Sidebar onAction={() => setNavOpen(false)} />
              <Button
                variant="outline"
                size="icon"
                aria-label="Close menu"
                onClick={() => setNavOpen(false)}
                className="absolute -right-1 top-5"
              >
                <X size={15} />
              </Button>
            </motion.div>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <Panel className="min-w-0 flex-1 overflow-hidden">
        <header className="flex shrink-0 items-center gap-3 border-b border-line/70 px-5 py-4">
          <Button
            variant="ghost"
            size="icon"
            aria-label="Open menu"
            onClick={() => setNavOpen(true)}
            className="-ml-2 md:hidden"
          >
            <Menu size={17} />
          </Button>

          <div className="min-w-0">
            <h2 className="display truncate text-base leading-none">Live session</h2>
            <p className="eyebrow mt-1 truncate">
              Full duplex &middot; it stops the moment you cut in
            </p>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => (window.location.hash = "drive")} aria-label="Open drive mode">
              <Car size={13} /> Drive
            </Button>
            <ConnectionBadge />
            {stage === "interrupted" ? <Badge tone="accent">interrupted</Badge> : null}
            <UserMenu />
            <Button
              variant="ghost"
              size="icon"
              aria-label={railOpen ? "Hide agent mind panel" : "Show agent mind panel"}
              onClick={() => setRailOpen((open) => !open)}
              className="hidden lg:inline-flex"
            >
              {railOpen ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
            </Button>
          </div>
        </header>

        <OfflineNotice />
        <Transcript />
        <Composer />
      </Panel>

      {railOpen ? (
        <div className="hidden lg:flex">
          <MindRail />
        </div>
      ) : null}
    </div>
  );
}
