import { useEffect, useState } from "react";
import {
  Action,
  ActionPanel,
  Icon,
  List,
  getPreferenceValues,
  open,
} from "@raycast/api";
import { useFetch } from "@raycast/utils";

interface Result {
  number: string;
  year: string;
  type: string;
  pdf_url: string;
  score: number;
  snippet: string;
  title: string;
}

export default function Command() {
  const { brainUrl } = getPreferenceValues<{ brainUrl: string }>();
  const base = (brainUrl || "http://127.0.0.1:8787").replace(/\/$/, "");

  const [text, setText] = useState("");
  const [query, setQuery] = useState("");

  // debounce keystrokes so we don't embed on every character
  useEffect(() => {
    const t = setTimeout(() => setQuery(text.trim()), 350);
    return () => clearTimeout(t);
  }, [text]);

  const { data, isLoading, error } = useFetch<Result[]>(
    `${base}/search?q=${encodeURIComponent(query)}&n=12`,
    {
      execute: query.length > 1,
      keepPreviousData: true,
      parseResponse: async (res) => {
        const json = (await res.json()) as { results?: Result[] };
        return json.results ?? [];
      },
    },
  );

  const results = data ?? [];

  return (
    <List
      isLoading={isLoading}
      searchText={text}
      onSearchTextChange={setText}
      searchBarPlaceholder="Περιγράψτε την υπόθεση ή γράψτε αριθμό απόφασης…"
      throttle
    >
      {error ? (
        <List.EmptyView
          icon={Icon.WifiDisabled}
          title="Δεν βρέθηκε ο server"
          description={`Τρέξε το brain: python server.py  (${base})`}
        />
      ) : results.length === 0 ? (
        <List.EmptyView
          icon={Icon.MagnifyingGlass}
          title={query.length > 1 ? "Καμία απόφαση" : "Γράψε τι ψάχνεις"}
          description="π.χ. «διαγωνισμός καθαριότητας νοσοκομείου» ή «1614/2025»"
        />
      ) : (
        results.map((r, i) => (
          <List.Item
            key={`${r.number}-${r.year}-${i}`}
            icon={Icon.Document}
            title={r.title}
            subtitle={r.snippet}
            accessories={[
              { tag: r.year },
              { text: r.score.toFixed(3) },
            ]}
            actions={
              <ActionPanel>
                <Action title="Άνοιγμα PDF" icon={Icon.Globe} onAction={() => open(r.pdf_url)} />
                <Action.CopyToClipboard title="Αντιγραφή Link" content={r.pdf_url} />
                <Action.CopyToClipboard
                  title="Αντιγραφή Αριθμού"
                  content={`${r.number}/${r.year}`}
                  shortcut={{ modifiers: ["cmd"], key: "." }}
                />
              </ActionPanel>
            }
          />
        ))
      )}
    </List>
  );
}
