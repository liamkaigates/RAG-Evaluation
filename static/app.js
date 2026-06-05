async function getJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Request failed: ${path}`);
  return response.json();
}

function pct(value) {
  return `${Math.round(value * 100)}%`;
}

function setStatus(text) {
  document.querySelector("#status").textContent = text;
}

async function loadSummary() {
  const [summary, evaluation] = await Promise.all([getJson("/api/summary"), getJson("/api/evaluate?top_k=4")]);
  document.querySelector("#docCount").textContent = summary.documents;
  document.querySelector("#chunkCount").textContent = summary.chunks;
  document.querySelector("#questionCount").textContent = summary.questions;
  document.querySelector("#mrr").textContent = evaluation.summary.mrr.toFixed(2);
  renderEvalRows(evaluation.questions);
}

function renderEvalRows(rows) {
  const container = document.querySelector("#evalRows");
  container.innerHTML = "";
  rows.forEach((row) => {
    const el = document.createElement("div");
    el.className = "eval-row";
    const question = document.createElement("div");
    const recall = document.createElement("div");
    const mrr = document.createElement("div");
    const coverage = document.createElement("div");
    question.textContent = row.question;
    recall.textContent = `Recall ${pct(row.retrieval.recall_at_k)}`;
    mrr.textContent = `MRR ${row.retrieval.mrr.toFixed(2)}`;
    coverage.textContent = `Cover ${pct(row.answer_metrics.keyword_coverage)}`;
    el.append(question, recall, mrr, coverage);
    container.appendChild(el);
  });
}

function renderResults(results) {
  const container = document.querySelector("#results");
  container.innerHTML = "";
  results.forEach((result) => {
    const el = document.createElement("div");
    el.className = "result";
    const header = document.createElement("div");
    header.className = "result-header";
    const title = document.createElement("strong");
    const score = document.createElement("span");
    const text = document.createElement("p");
    const id = document.createElement("small");
    title.textContent = `${result.rank}. ${result.title}`;
    score.className = "score";
    score.textContent = `Hybrid ${result.score.toFixed(3)} | Sparse ${result.sparse_score.toFixed(3)} | Dense ${result.dense_score.toFixed(3)}`;
    text.textContent = result.text;
    id.textContent = [result.citation, result.source_path, result.id].filter(Boolean).join(" | ");
    header.append(title, score);
    el.append(header, text, id);
    container.appendChild(el);
  });
}

function renderSources(sources) {
  const container = document.querySelector("#sources");
  container.innerHTML = "";
  sources.forEach((source) => {
    const el = document.createElement("div");
    el.className = "source";
    el.textContent = `[${source.label}] ${source.citation || source.title} ${source.source_path ? `(${source.source_path})` : ""}`;
    container.appendChild(el);
  });
}

async function runSearch() {
  const question = document.querySelector("#question").value;
  const payload = await getJson(`/api/search?question=${encodeURIComponent(question)}&top_k=4`);
  document.querySelector("#answer").textContent = payload.answer;
  document.querySelector("#citations").textContent = `Cited chunks: ${payload.citations.join(", ")}`;
  renderSources(payload.sources || []);
  renderResults(payload.results);
  setStatus("Query ready");
}

async function init() {
  try {
    await loadSummary();
    document.querySelector("#search").addEventListener("click", runSearch);
    await runSearch();
  } catch (error) {
    console.error(error);
    setStatus("Dashboard error");
  }
}

init();
