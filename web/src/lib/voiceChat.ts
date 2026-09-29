import { create } from "zustand";
import { subscribeFrames, useSession } from "~/store/session";
import {
  isEcho,
  isHold,
  isResume,
  Listener,
  Speaker,
  voiceInputSupported,
  voiceOutputSupported,
} from "./speech";
import type { ServerFrame } from "./types";

/**
 * Hands-free voice for the assistant: the microphone streams partial speech
 * (so speculation starts mid-sentence), replies are read aloud as they stream,
 * and talking over the agent stops it at once. "Hold on" only pauses; "go on"
 * picks up from what was actually heard, not from what had merely streamed.
 */

interface VoiceState {
  supported: boolean;
  listening: boolean;
  speaking: boolean;
  voiceOut: boolean;
  interim: string;
  error: string | null;
  toggleListening: () => void;
  setVoiceOut: (on: boolean) => void;
  bargeIn: () => void;
}

const speaker = new Speaker();
// Citation markers ("[flights#3]") and stray markdown are for the eye, not the ear.
speaker.toSpeech = (text) =>
  text
    .replace(/\[[^\]\s]{1,40}\]/g, "")
    .replace(/[*_#`>|]+/g, " ")
    .replace(/(^|\s)[-•]\s+/g, "$1") // list bullets
    .replace(/\s+/g, " ")
    .trim();

let listener: Listener | null = null;
const replyText: Record<string, string> = {};
const enqueuedUpTo: Record<string, number> = {};
const muted = new Set<string>(); // replies the user cut off: never read out again unasked
let paused: { turnId: string; heard: number } | null = null;
let recentlySpoken = { text: "", until: 0 }; // echo guard just after the agent stops
let localSeq = 0;

const active = () => useVoice.getState().listening && useVoice.getState().voiceOut;

/** Queue every sentence of a reply that has fully arrived; `final` flushes the rest. */
function speakArrived(turnId: string, final: boolean) {
  const text = replyText[turnId] ?? "";
  let from = enqueuedUpTo[turnId] ?? 0;
  const boundary = /[.!?](\s+|$)/g;
  boundary.lastIndex = from;
  let match: RegExpExecArray | null;
  while ((match = boundary.exec(text))) {
    const end = match.index + match[0].length;
    if (!final && end === text.length && !/\s$/.test(match[0])) break; // may still be mid-number ("5.")
    speaker.enqueue(turnId, text.slice(from, end), from);
    from = end;
  }
  if (final && from < text.length) {
    speaker.enqueue(turnId, text.slice(from), from);
    from = text.length;
  }
  enqueuedUpTo[turnId] = from;
}

function serverBusy() {
  const { messages, streaming } = useSession.getState();
  return streaming !== null || messages[messages.length - 1]?.role === "user";
}

function onFrame(frame: ServerFrame) {
  switch (frame.t) {
    case "token":
      replyText[frame.turn_id] = (replyText[frame.turn_id] ?? "") + frame.text;
      if (active() && !muted.has(frame.turn_id)) speakArrived(frame.turn_id, false);
      break;
    case "message":
      if (frame.role !== "agent") break;
      if (!replyText[frame.turn_id]) replyText[frame.turn_id] = frame.content; // an answer that was not streamed
      if (active() && !muted.has(frame.turn_id) && frame.status !== "interrupted") speakArrived(frame.turn_id, true);
      break;
    case "filler":
      // "One moment..." while the slow part runs - only into silence
      if (active() && !speaker.speaking) speaker.enqueue(`filler:${frame.turn_id}`, frame.text, 0);
      break;
  }
}

function addLocalUserMessage(text: string) {
  useSession.setState((s) => ({
    draft: "",
    messages: [
      ...s.messages,
      { turnId: `local-${localSeq++}`, role: "user", content: text, status: "complete", modality: "voice", at: Date.now() },
    ],
  }));
}

export const useVoice = create<VoiceState>((set, get) => {
  speaker.onSpeaking = (speaking) => {
    if (!speaking) recentlySpoken = { text: speaker.lastText, until: Date.now() + 1500 };
    set({ speaking });
  };

  const echo = (text: string) =>
    isEcho(text, speaker.currentText) || (Date.now() < recentlySpoken.until && isEcho(text, recentlySpoken.text));

  const onPartial = (text: string) => {
    if (echo(text)) return;
    if (speaker.speaking) get().bargeIn(); // the user is talking over the agent
    set({ interim: text });
    if (!isHold(text)) useSession.getState().setDraft(text); // retrieval starts on partial speech
  };

  const onFinal = (text: string) => {
    set({ interim: "" });
    if (echo(text)) return;
    const session = useSession.getState();
    const wasSpeaking = speaker.speaking;
    if (wasSpeaking) get().bargeIn(); // also stops generation
    if (isHold(text)) {
      // A pause, not a question: stay quiet and keep the rest for "go on".
      if (!wasSpeaking && serverBusy()) session.interrupt("barge_in");
      useSession.setState({ draft: "" });
      return;
    }
    if (isResume(text) && paused) {
      const { turnId, heard } = paused;
      paused = null;
      muted.delete(turnId);
      enqueuedUpTo[turnId] = heard; // read on from exactly where the listener stopped hearing
      speakArrived(turnId, true);
      if (session.checkpoint) session.submit(text, "voice"); // the server also finishes what it had not generated
      else addLocalUserMessage(text);
      return;
    }
    paused = null;
    session.submit(text, "voice");
  };

  return {
    supported: voiceInputSupported(),
    listening: false,
    speaking: false,
    voiceOut: voiceOutputSupported(),
    interim: "",
    error: voiceInputSupported() ? null : "This browser has no speech recognition - use Chrome or Edge.",

    toggleListening: () => {
      if (!listener) {
        listener = new Listener({
          onPartial,
          onFinal,
          onState: (listening, error) => set({ listening, error: error ?? null }),
        });
      }
      if (listener.listening) {
        listener.stop();
        speaker.cancel();
        paused = null;
        set({ interim: "" });
      } else {
        listener.start();
      }
    },

    setVoiceOut: (on) => {
      speaker.setEnabled(on);
      set({ voiceOut: on && voiceOutputSupported() });
    },

    bargeIn: () => {
      const stopped = speaker.cancel();
      recentlySpoken = { text: "", until: 0 };
      if (stopped && !stopped.turnId.startsWith("filler:")) {
        muted.add(stopped.turnId);
        paused = { turnId: stopped.turnId, heard: stopped.heard };
      }
      if (serverBusy()) useSession.getState().interrupt("barge_in"); // stop generating, not just speaking
    },
  };
});

subscribeFrames(onFrame);
