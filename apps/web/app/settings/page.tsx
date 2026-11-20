"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useSettingsStore, type AppSettings } from "@/stores/settingsStore";
import { cn } from "@/lib/utils";

// ─── Brand SVG Icons ──────────────────────────────────────────────────────────






function ServerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" xmlns="http://www.w3.org/2000/svg">
      <path d="M4 6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2zm0 8a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z"/>
      <path d="M7 8h.01M7 16h.01"/>
    </svg>
  );
}

function SlidersIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" xmlns="http://www.w3.org/2000/svg">
      <path d="M10.5 6h9.75M10.5 6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75"/>
    </svg>
  );
}




function KeyIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4" />
    </svg>
  );
}

// ─── Connection test ───────────────────────────────────────────────────────────

type TestStatus = "idle" | "testing" | "ok" | "fail";

function useConnectionTest() {
  const [status, setStatus] = useState<TestStatus>("idle");
  const [latency, setLatency] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const test = useCallback(async (apiUrl: string) => {
    setStatus("testing");
    setError(null);
    const start = Date.now();
    try {
      const res = await fetch(`${apiUrl.replace(/\/$/, "")}/health`, {
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setLatency(Date.now() - start);
      setStatus("ok");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("fail");
    }
  }, []);

  return { status, latency, error, test };
}

const MODEL_OPTIONS = [
  { value: "gpt-4o-mini",                 label: "GPT-4o mini",           provider: "OpenAI" },
  { value: "gpt-4o",                      label: "GPT-4o",                provider: "OpenAI" },
  { value: "o3-mini",                     label: "o3-mini",               provider: "OpenAI" },
  { value: "deepseek-chat",               label: "DeepSeek-V3",           provider: "DeepSeek" },
  { value: "deepseek-reasoner",           label: "DeepSeek-R1",           provider: "DeepSeek" },
  { value: "claude-3-5-sonnet-20241022",  label: "Claude 3.5 Sonnet",     provider: "Anthropic" },
  { value: "claude-3-opus-20240229",      label: "Claude 3 Opus",         provider: "Anthropic" },
  { value: "qwen-plus",                   label: "Qwen Plus",             provider: "Alibaba" },
  { value: "qwen-max",                    label: "Qwen Max",              provider: "Alibaba" },
];

// ─── Tabs ────────────────────────────────────────────────────────────────────

type TabId = "connection" | "providers" | "defaults";

type TabDef = {
  id: TabId;
  label: string;
  Icon: (props: { className?: string }) => JSX.Element;
};

const TABS: TabDef[] = [
  { id: "connection", label: "Connection", Icon: ServerIcon },
  { id: "providers", label: "Credentials", Icon: KeyIcon },
  { id: "defaults", label: "Defaults", Icon: SlidersIcon },
];

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function SettingsPage() {
  const { settings, saveSettings } = useSettingsStore();
  const [draft, setDraft] = useState<AppSettings>({ ...settings });
  const [saved, setSaved] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>("connection");
  const conn = useConnectionTest();

  function updateDraft<K extends keyof AppSettings>(key: K, value: AppSettings[K]) {
    setDraft((prev) => ({ ...prev, [key]: value }));
    setSaved(false);
  }


  function handleSave() {
    saveSettings(draft);
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  }


  return (
    <div className="max-w-2xl space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
        <a href="/tasks" className="hover:text-foreground transition-colors">Tasks</a>
        <span>/</span>
        <span className="text-foreground font-medium">Settings</span>
      </div>
      {/* ── Page header ────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Configure your backend and model providers.
          </p>
        </div>

      </div>

      {/* ── Tabs ───────────────────────────────────────────────────────────── */}
      <div className="flex gap-1 rounded-xl bg-muted/50 p-1 border border-border/60">
        {TABS.map(({ id, label, Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={cn(
              "flex flex-1 items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-150",
              activeTab === id
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </div>

      {/* ── Tab: Connection ────────────────────────────────────────────────── */}
      {activeTab === "connection" && (
        <Card className="border-border/80">
          <CardHeader className="pb-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-50 dark:bg-blue-950/30">
                <ServerIcon className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <CardTitle className="text-base">API Connection</CardTitle>
                <CardDescription className="text-xs mt-0.5">
                  Your R U Socrates FastAPI backend URL.
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="space-y-2">
              <label className="text-sm font-medium">Backend URL</label>
              <div className="flex gap-2">
                <Input
                  type="url"
                  value={draft.apiUrl}
                  onChange={(e) => updateDraft("apiUrl", e.target.value)}
                  placeholder="http://localhost:8000"
                  className="flex-1 font-mono text-sm"
                />
                <Button
                  variant="outline"
                  size="sm"
                  disabled={conn.status === "testing"}
                  onClick={() => conn.test(draft.apiUrl)}
                  className="shrink-0 gap-1.5"
                >
                  {conn.status === "testing" ? (
                    <>
                      <svg className="h-3.5 w-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                      </svg>
                      Testing
                    </>
                  ) : "Test Connection"}
                </Button>
              </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">API Token</label>
              <p className="text-xs text-muted-foreground">
                Only needed when the backend is started with{" "}
                <code className="rounded bg-muted px-1 py-0.5">RUS_API_TOKEN</code>.
                Held in memory for this session, never saved.
              </p>
              <Input
                type="password"
                value={draft.apiToken}
                onChange={(e) => updateDraft("apiToken", e.target.value)}
                placeholder="leave empty if the backend has no token"
                className="font-mono text-sm"
                autoComplete="off"
              />
            </div>

              {/* Connection status */}
              {conn.status !== "idle" && (
                <div className={cn(
                  "flex items-center gap-2 rounded-lg border px-3 py-2 text-xs font-medium",
                  conn.status === "ok"
                    ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/50 dark:bg-emerald-950/20 dark:text-emerald-400"
                    : conn.status === "fail"
                    ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900/50 dark:bg-red-950/20 dark:text-red-400"
                    : "border-border bg-muted/50 text-muted-foreground"
                )}>
                  {conn.status === "ok" && (
                    <>
                      <svg className="h-3.5 w-3.5 text-emerald-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <path d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"/>
                      </svg>
                      Connected — {conn.latency}ms
                    </>
                  )}
                  {conn.status === "fail" && (
                    <>
                      <svg className="h-3.5 w-3.5 text-red-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <path d="m9.75 9.75 4.5 4.5m0-4.5-4.5 4.5M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"/>
                      </svg>
                      Failed — {conn.error}
                    </>
                  )}
                </div>
              )}
            </div>

            {/* Health endpoint hint */}
            <div className="rounded-lg border border-dashed border-border/60 bg-muted/20 px-4 py-3 text-xs text-muted-foreground">
              <span className="font-medium text-foreground/60">Endpoint tested:</span>{" "}
              <code className="font-mono">{draft.apiUrl.replace(/\/$/, "")}/health</code>
              <p className="mt-1">
                Your FastAPI server should return{" "}
                <code className="font-mono bg-muted px-1 rounded">{"{ status: \"ok\" }"}</code>
                {" "}at this path.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Tab: API Keys ──────────────────────────────────────────────────── */}
      {activeTab === "providers" && (
        <Card className="border-border/80">
          <CardHeader className="pb-4">
            <CardTitle className="text-base">Provider Credentials</CardTitle>
            <CardDescription className="text-xs mt-0.5">
              Keys belong to the backend, not to the browser.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            <p className="text-muted-foreground">
              Every LLM call is made by the Python backend, so the keys are read
              from its environment. Put them in{" "}
              <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
                services/api/.env
              </code>
              :
            </p>
            <pre className="rounded-lg border bg-muted/40 p-3 text-xs overflow-x-auto">
              OPENAI_API_KEY=sk-...{"\n"}
              ANTHROPIC_API_KEY=sk-ant-...{"\n"}
              DASHSCOPE_API_KEY=sk-...
            </pre>
            <p className="text-muted-foreground">
              A key typed into a web form ends up in browser storage, where any
              script on the page can read it. Keeping it on the server also means
              the client never sees it at all.
            </p>
          </CardContent>
        </Card>
      )}

      {/* ── Tab: Defaults ─────────────────────────────────────────────────── */}
      {activeTab === "defaults" && (
        <Card className="border-border/80">
          <CardHeader className="pb-4">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-purple-50 dark:bg-purple-950/30">
                <SlidersIcon className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <CardTitle className="text-base">Task Defaults</CardTitle>
                <CardDescription className="text-xs mt-0.5">
                  Applied to every new research task.
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Default model */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Default Model</label>
              <p className="text-xs text-muted-foreground">
                The LLM forwarded to LiteLLM for each task.
              </p>
              <div className="grid grid-cols-1 gap-2">
                {MODEL_OPTIONS.map((m) => (
                  <label
                    key={m.value}
                    className={cn(
                      "flex cursor-pointer items-center gap-3 rounded-lg border px-4 py-3 transition-all duration-150",
                      draft.defaultModel === m.value
                        ? "border-primary bg-primary/5"
                        : "border-border hover:border-muted-foreground/40 hover:bg-muted/30"
                    )}
                  >
                    <input
                      type="radio"
                      name="defaultModel"
                      value={m.value}
                      checked={draft.defaultModel === m.value}
                      onChange={() => updateDraft("defaultModel", m.value)}
                      className="h-4 w-4 accent-primary"
                    />
                    <div className="flex-1 min-w-0">
                      <span className="text-sm font-medium">{m.label}</span>
                      <span className="ml-2 text-xs text-muted-foreground">{m.provider}</span>
                    </div>
                    <code className="text-[10px] font-mono text-muted-foreground truncate">
                      {m.value}
                    </code>
                  </label>
                ))}
              </div>
            </div>

            {/* Max iterations */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Max Iterations per Task</label>
              <p className="text-xs text-muted-foreground">
                Higher values = deeper research, longer runtimes. Range: 1–100.
              </p>
              <div className="flex items-center gap-4">
                <Input
                  type="number"
                  value={draft.maxIterations}
                  onChange={(e) =>
                    updateDraft("maxIterations", Math.max(1, Math.min(100, parseInt(e.target.value) || 10)))
                  }
                  min={1}
                  max={100}
                  className="w-24 text-center font-mono"
                />
                <div className="flex-1">
                  <div className="flex justify-between text-xs text-muted-foreground mb-1">
                    <span>Quick (1–5)</span>
                    <span>Deep (50–100)</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-blue-400 to-purple-500 transition-all duration-300"
                      style={{ width: `${(draft.maxIterations / 100) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ── Save bar ───────────────────────────────────────────────────────── */}
      <div className="sticky bottom-0 flex items-center justify-end gap-3 rounded-xl border border-border/80 bg-background/95 backdrop-blur px-4 py-3 shadow-sm">
        <p className="flex-1 text-xs text-muted-foreground">
          Changes are saved to your browser and apply immediately.
        </p>
        {saved && (
          <div className="flex items-center gap-1.5 text-sm font-medium text-emerald-600 dark:text-emerald-400">
            <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M9 12.75 11.25 15 15 9.75M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"/>
            </svg>
            Saved!
          </div>
        )}
        <Button onClick={handleSave} className="gap-2">
          <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 1-2-2V9m0 0h18"/>
          </svg>
          Save Settings
        </Button>
      </div>
    </div>
  );
}
