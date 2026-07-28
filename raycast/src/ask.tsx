import { useEffect, useRef, useState } from "react";
import { Action, ActionPanel, Icon, List, getPreferenceValues } from "@raycast/api";

interface Source {
  number: string;
  year: string;
  type: string;
  pdf_url: string;
  title: string;
}

function baseUrl() {
  const { brainUrl } = getPreferenceValues<{ brainUrl: string }>();
  return (brainUrl || "http://127.0.0.1:8787").replace(/\/$/, "");
}

// Canonical Raycast chat pattern: the List's own search bar IS the text input
// (filtering=false + onSearchTextChange). Enter fires the item's action = send.
// The answer streams into the item's detail pane.
export default function Command() {
  const [input, setInput] = useState(""); // live search-bar text
  const [question, setQuestion] = useState(""); // last submitted question
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [loading, setLoading] = useState(false);
  const runId = useRef(0);

  function ask() {
    const q = input.trim();
    if (!q) return;
    setQuestion(q);
    setInput("");
  }

  useEffect(() => {
    if (!question) return;
    const myRun = ++runId.current;
    const controller = new AbortController();
    setAnswer("");
    setSources([]);
    setLoading(true);

    (async () => {
      try {
        const res = await fetch(`${baseUrl()}/ask?q=${encodeURIComponent(question)}`, {
          signal: controller.signal,
          headers: { Accept: "text/event-stream" },
        });
        if (!res.ok || !res.body) throw new Error(`server ${res.status}`);
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        let acc = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          if (runId.current !== myRun) return;
          buf += decoder.decode(value, { stream: true });
          const frames = buf.split("\n\n");
          buf = frames.pop() ?? "";
          for (const frame of frames) {
            const line = frame.trim();
            if (!line.startsWith("data:")) continue;
            const ev = JSON.parse(line.slice(5).trim());
            if (ev.t) {
              acc += ev.t;
              setAnswer(acc);
            } else if (ev.error) {
              acc += `\n\n> ${ev.error}`;
              setAnswer(acc);
            } else if (ev.sources) {
              setSources(ev.sources);
            }
          }
        }
      } catch (e) {
        if (!(e instanceof Error && e.name === "AbortError")) {
          setAnswer((a) => a + "\n\nΔεν υπάρχει σύνδεση με τον server (server.py).");
        }
      } finally {
        if (runId.current === myRun) setLoading(false);
      }
    })();

    return () => controller.abort();
  }, [question]);

  const cites =
    sources.length > 0
      ? "\n\n---\n\nΠηγές — " + sources.map((s) => `[${s.number}/${s.year}](${s.pdf_url})`).join("  ·  ")
      : "";

  const detailMarkdown = !question
    ? "Γράψε μια ερώτηση στα ελληνικά στο πεδίο πάνω και πάτα **Enter**.\n\nπ.χ. *Τι ισχύει για την εγγυητική επιστολή συμμετοχής;*"
    : `## ${question}\n\n` + (answer || (sources.length ? "…" : "Αναζήτηση…")) + cites;

  return (
    <List
      filtering={false}
      isShowingDetail
      searchText={input}
      onSearchTextChange={setInput}
      searchBarPlaceholder="Ρώτησε για αποφάσεις ΕΑΔΗΣΥ και πάτα Enter…"
      isLoading={loading}
    >
      <List.Item
        icon={Icon.QuestionMark}
        title={question || input || "Νέα ερώτηση"}
        detail={<List.Item.Detail markdown={detailMarkdown} />}
        actions={
          <ActionPanel>
            <Action title="Ρώτησε" icon={Icon.ArrowRight} onAction={ask} />
            {sources.map((s) => (
              <Action.OpenInBrowser
                key={`${s.number}-${s.year}`}
                title={`Άνοιγμα ${s.number}/${s.year}`}
                url={s.pdf_url}
              />
            ))}
          </ActionPanel>
        }
      />
    </List>
  );
}
