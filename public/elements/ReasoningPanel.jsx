export default function ReasoningPanel() {
  const thinking = props.thinking || "";
  const stats = props.stats || {};
  const isThinking = props.isThinking || false;

  const hasStats =
    stats.tok_s !== undefined ||
    stats.prompt_tokens !== undefined ||
    stats.completion_tokens !== undefined ||
    stats.total_time_s !== undefined ||
    stats.load_time_s !== undefined;

  if (!thinking && !isThinking && !hasStats) {
    return null;
  }

  return (
    <details open={isThinking} className="mt-2 mb-3 rounded-lg border border-border bg-muted/40 px-3 py-2 text-sm">
      <summary className="cursor-pointer select-none font-medium text-muted-foreground">
        {isThinking ? "Thinking…" : "Thinking"}
        {hasStats && stats.tok_s !== undefined ? (
          <span className="ml-2 text-xs text-muted-foreground">
            {Number(stats.tok_s).toFixed(2)} tok/s
          </span>
        ) : null}
      </summary>

      {thinking ? (
        <div className="mt-2 whitespace-pre-wrap text-muted-foreground">
          {thinking}
        </div>
      ) : (
        <div className="mt-2 italic text-muted-foreground">
          Waiting for reasoning…
        </div>
      )}

      {hasStats ? (
        <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
          {stats.tok_s !== undefined && stats.tok_s !== null ? (
            <span>Speed: {Number(stats.tok_s).toFixed(2)} tok/s</span>
          ) : null}

          {stats.prompt_tokens !== undefined && stats.prompt_tokens !== null ? (
            <span>Prompt: {stats.prompt_tokens}</span>
          ) : null}

          {stats.completion_tokens !== undefined && stats.completion_tokens !== null ? (
            <span>Output: {stats.completion_tokens}</span>
          ) : null}

          {stats.total_time_s !== undefined && stats.total_time_s !== null ? (
            <span>Total: {Number(stats.total_time_s).toFixed(2)}s</span>
          ) : null}

          {stats.load_time_s !== undefined && stats.load_time_s !== null ? (
            <span>Load: {Number(stats.load_time_s).toFixed(2)}s</span>
          ) : null}
        </div>
      ) : null}
    </details>
  );
}
