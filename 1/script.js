const LEXICON = {
  "南京": { freq: 70, pos: "ns" },
  "南京市": { freq: 96, pos: "ns" },
  "市长": { freq: 65, pos: "n" },
  "长江": { freq: 85, pos: "ns" },
  "长江大桥": { freq: 92, pos: "n" },
  "大桥": { freq: 54, pos: "n" },
  "研究": { freq: 90, pos: "v" },
  "研究生": { freq: 84, pos: "n" },
  "生命": { freq: 88, pos: "n" },
  "起源": { freq: 72, pos: "n" },
  "乒乓球": { freq: 80, pos: "n" },
  "乒乓球拍": { freq: 76, pos: "n" },
  "球拍": { freq: 52, pos: "n" },
  "拍卖": { freq: 83, pos: "v" },
  "卖完": { freq: 74, pos: "v" },
  "完了": { freq: 70, pos: "u" },
  "规范化": { freq: 62, pos: "vn" },
  "结果": { freq: 66, pos: "n" },
  "分词": { freq: 80, pos: "vn" },
  "后": { freq: 40, pos: "f" },
  "再": { freq: 38, pos: "d" },
  "请": { freq: 35, pos: "v" },
  "把": { freq: 30, pos: "p" },
  "的": { freq: 98, pos: "u" },
  "了": { freq: 95, pos: "u" },
  "ai": { freq: 60, pos: "nx" },
  "nlp": { freq: 60, pos: "nx" }
};

const POS_LABELS = {
  n: "普通名词",
  ns: "地名/专名",
  v: "动词",
  vn: "动名词",
  u: "助词",
  f: "方位词",
  d: "副词",
  p: "介词",
  nx: "外文专名",
  x: "未知"
};

const ALGORITHMS = [
  {
    key: "fmm",
    label: "正向最大匹配",
    tag: "FMM",
    description: "从左到右优先切最长词，直观、速度快，但容易被早期决策锁住。"
  },
  {
    key: "rmm",
    label: "逆向最大匹配",
    tag: "RMM",
    description: "从右到左切最长词，常用来和 FMM 对照，能缓解部分尾部歧义。"
  },
  {
    key: "dp",
    label: "频率驱动动态规划",
    tag: "DP",
    description: "综合词频选择整句最优路径，更接近现代分词器的统计思想。"
  }
];

const exampleButtons = document.querySelectorAll(".example-chip");
const sentenceInput = document.getElementById("sentence-input");
const runButton = document.getElementById("run-button");
const rawOutput = document.getElementById("raw-output");
const normalizedOutput = document.getElementById("normalized-output");
const normalizationList = document.getElementById("normalization-list");
const segmentationResults = document.getElementById("segmentation-results");
const statsSummary = document.getElementById("stats-summary");
const insights = document.getElementById("insights");
const chartCanvas = document.getElementById("stats-chart");

const toggleIds = [
  "normalize-width",
  "normalize-case",
  "normalize-punctuation",
  "normalize-space"
];

exampleButtons.forEach((button) => {
  button.addEventListener("click", () => {
    sentenceInput.value = button.dataset.example || "";
    analyze();
  });
});

toggleIds.forEach((id) => {
  document.getElementById(id).addEventListener("change", analyze);
});

runButton.addEventListener("click", analyze);
sentenceInput.addEventListener("input", analyze);

function toHalfWidth(text) {
  return Array.from(text).map((char) => {
    const code = char.charCodeAt(0);
    if (code === 12288) return " ";
    if (code >= 65281 && code <= 65374) return String.fromCharCode(code - 65248);
    return char;
  }).join("");
}

function normalizeText(text) {
  const operations = [];
  let current = text;

  if (document.getElementById("normalize-width").checked) {
    const next = toHalfWidth(current);
    if (next !== current) {
      operations.push({
        title: "全角转半角",
        before: current,
        after: next,
        reason: "统一字符宽度，避免 `ＡＩ` 与 `AI` 被当成不同字符串。"
      });
      current = next;
    }
  }

  if (document.getElementById("normalize-case").checked) {
    const next = current.replace(/[A-Z]+/g, (match) => match.toLowerCase());
    if (next !== current) {
      operations.push({
        title: "英文转小写",
        before: current,
        after: next,
        reason: "把 `NLP`、`Ai` 这类英文形式统一，方便词典匹配。"
      });
      current = next;
    }
  }

  if (document.getElementById("normalize-punctuation").checked) {
    const next = current
      .replace(/[，、]/g, "，")
      .replace(/[！!]/g, "！")
      .replace(/[？?]/g, "？")
      .replace(/[；;]/g, "；");
    if (next !== current) {
      operations.push({
        title: "标点统一",
        before: current,
        after: next,
        reason: "统一句内标点，减少规则匹配时的额外分支。"
      });
      current = next;
    }
  }

  if (document.getElementById("normalize-space").checked) {
    const next = current.replace(/\s+/g, " ").trim();
    if (next !== current) {
      operations.push({
        title: "空白压缩",
        before: current,
        after: next,
        reason: "去除多余空格，让中文与英文混写句子更规整。"
      });
      current = next;
    }
  }

  if (operations.length === 0) {
    operations.push({
      title: "当前输入已较规范",
      before: current,
      after: current,
      reason: "这说明预处理并不是每次都改字形，但它始终在守住输入质量。"
    });
  }

  return { text: current, operations };
}

function scanCandidates(text, start) {
  const maxLength = 6;
  const items = [];
  for (let len = 1; len <= maxLength && start + len <= text.length; len += 1) {
    const piece = text.slice(start, start + len);
    if (LEXICON[piece]) {
      items.push(piece);
    }
  }
  return items;
}

function scanCandidatesBackward(text, end) {
  const maxLength = 6;
  const items = [];
  for (let len = 1; len <= maxLength && end - len >= 0; len += 1) {
    const piece = text.slice(end - len, end);
    if (LEXICON[piece]) {
      items.push(piece);
    }
  }
  return items;
}

function forwardMaximumMatching(text) {
  const tokens = [];
  let index = 0;
  while (index < text.length) {
    const candidates = scanCandidates(text, index);
    if (candidates.length > 0) {
      const token = candidates.sort((a, b) => b.length - a.length)[0];
      tokens.push(token);
      index += token.length;
    } else {
      tokens.push(text[index]);
      index += 1;
    }
  }
  return tokens;
}

function reverseMaximumMatching(text) {
  const tokens = [];
  let index = text.length;
  while (index > 0) {
    const candidates = scanCandidatesBackward(text, index);
    if (candidates.length > 0) {
      const token = candidates.sort((a, b) => b.length - a.length)[0];
      tokens.unshift(token);
      index -= token.length;
    } else {
      tokens.unshift(text[index - 1]);
      index -= 1;
    }
  }
  return tokens;
}

function dynamicProgrammingSegment(text) {
  const best = new Array(text.length + 1).fill(null);
  best[text.length] = { score: 0, tokens: [] };

  for (let i = text.length - 1; i >= 0; i -= 1) {
    const candidates = scanCandidates(text, i);
    const options = [];

    candidates.forEach((candidate) => {
      const next = best[i + candidate.length];
      if (!next) return;
      const weight = Math.log((LEXICON[candidate]?.freq || 1) + 1) + candidate.length * 1.2;
      options.push({
        score: weight + next.score,
        tokens: [candidate, ...next.tokens]
      });
    });

    const fallbackNext = best[i + 1];
    options.push({
      score: -0.8 + (fallbackNext ? fallbackNext.score : 0),
      tokens: [text[i], ...(fallbackNext ? fallbackNext.tokens : [])]
    });

    best[i] = options.sort((a, b) => b.score - a.score)[0];
  }

  return best[0].tokens;
}

function tagTokens(tokens) {
  return tokens.map((token) => {
    const entry = LEXICON[token];
    return {
      token,
      pos: entry?.pos || "x",
      label: POS_LABELS[entry?.pos || "x"]
    };
  });
}

function collectMetrics(tokens, baselineTokens) {
  const tokenLengths = tokens.map((token) => token.length);
  const averageLength = tokenLengths.reduce((sum, value) => sum + value, 0) / tokenLengths.length;
  const baselineSet = new Set(baselineTokens);
  const overlap = tokens.filter((token) => baselineSet.has(token)).length / Math.max(tokens.length, baselineTokens.length);
  return {
    tokenCount: tokens.length,
    averageLength,
    overlap
  };
}

function renderNormalization(operations) {
  normalizationList.innerHTML = operations.map((item) => `
    <article class="normalization-item">
      <strong>${item.title}</strong>
      <p><strong>前：</strong>${escapeHtml(item.before)}</p>
      <p><strong>后：</strong>${escapeHtml(item.after)}</p>
      <p>${item.reason}</p>
    </article>
  `).join("");
}

function renderAlgorithms(results) {
  segmentationResults.innerHTML = results.map((result) => `
    <article class="algorithm-card">
      <div class="algorithm-head">
        <div>
          <h3>${result.label}</h3>
          <p>${result.description}</p>
        </div>
        <span class="algorithm-tag">${result.tag}</span>
      </div>
      <div>
        <strong>分词结果</strong>
        <div class="token-row">
          ${result.tagged.map((item) => `<div class="token-pill">${escapeHtml(item.token)}<small>${item.pos} · ${item.label}</small></div>`).join("")}
        </div>
      </div>
      <div>
        <strong>统计</strong>
        <p>词数：${result.metrics.tokenCount}，平均词长：${result.metrics.averageLength.toFixed(2)}，与 DP 结果重合度：${(result.metrics.overlap * 100).toFixed(0)}%</p>
      </div>
    </article>
  `).join("");
}

function renderInsights(normalized, results) {
  const fmm = results[0].tokens.join(" / ");
  const rmm = results[1].tokens.join(" / ");
  const dp = results[2].tokens.join(" / ");
  const insightItems = [
    `规范化后的文本是“${normalized}”。预处理的目标不是“改写句子”，而是把输入变成更适合算法处理的统一形式。`,
    `FMM 结果为“${fmm}”，它会优先相信左侧最长词，因此在歧义句里容易较早做出承诺。`,
    `RMM 结果为“${rmm}”，如果句尾存在高频长词，它往往会和 FMM 得出不同的切分。`,
    `DP 结果为“${dp}”，它不是只看局部最长，而是比较整句路径分数，所以更适合解释“为什么现代方法常引入统计信息”。`,
    "词性标注依附于分词结果存在。同一个字串一旦被切成不同词，后面得到的词性序列也会跟着变化。"
  ];

  insights.innerHTML = insightItems.map((text) => `
    <article class="insight-item">
      <p>${escapeHtml(text)}</p>
    </article>
  `).join("");
}

function drawChart(results) {
  const ctx = chartCanvas.getContext("2d");
  const width = chartCanvas.width;
  const height = chartCanvas.height;
  ctx.clearRect(0, 0, width, height);

  const bars = results.map((result) => ({
    label: result.tag,
    value: result.metrics.tokenCount,
    alt: result.metrics.averageLength
  }));
  const maxValue = Math.max(...bars.map((item) => item.value), 1);

  ctx.fillStyle = "#f9f3e6";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(24, 33, 38, 0.1)";
  for (let i = 0; i < 5; i += 1) {
    const y = 36 + i * 42;
    ctx.beginPath();
    ctx.moveTo(34, y);
    ctx.lineTo(width - 18, y);
    ctx.stroke();
  }

  bars.forEach((bar, index) => {
    const x = 60 + index * 118;
    const barHeight = (bar.value / maxValue) * 120;
    const y = 198 - barHeight;
    const gradient = ctx.createLinearGradient(x, y, x, 198);
    gradient.addColorStop(0, index === 2 ? "#d79c2f" : "#0e7c66");
    gradient.addColorStop(1, index === 2 ? "#e57d68" : "#62b6a3");
    ctx.fillStyle = gradient;
    ctx.fillRect(x, y, 58, barHeight);

    ctx.fillStyle = "#182126";
    ctx.font = "600 14px Outfit";
    ctx.fillText(bar.label, x + 10, 220);
    ctx.fillText(String(bar.value), x + 20, y - 8);
    ctx.fillStyle = "#5f6b70";
    ctx.font = "12px Outfit";
    ctx.fillText(`均长 ${bar.alt.toFixed(1)}`, x - 4, 238);
  });
}

function renderStats(results) {
  drawChart(results);
  const bestCompact = [...results].sort((a, b) => a.metrics.tokenCount - b.metrics.tokenCount)[0];
  const bestBalanced = [...results].sort((a, b) => b.metrics.overlap - a.metrics.overlap)[0];
  statsSummary.innerHTML = `
    当前句子里，<strong>${bestCompact.label}</strong> 切出的词数最少，说明它更倾向于把字串并成较长词。
    <strong>${bestBalanced.label}</strong> 与 DP 参考路径的重合度最高，更适合作为课堂里的“相对稳妥”切分示例。
  `;
}

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function analyze() {
  const raw = sentenceInput.value || "";
  const normalization = normalizeText(raw);
  rawOutput.textContent = raw || "请先输入一句中文文本。";
  normalizedOutput.textContent = normalization.text || "规范化结果会显示在这里。";
  renderNormalization(normalization.operations);

  const base = normalization.text || raw;
  const segmented = [
    { ...ALGORITHMS[0], tokens: forwardMaximumMatching(base) },
    { ...ALGORITHMS[1], tokens: reverseMaximumMatching(base) },
    { ...ALGORITHMS[2], tokens: dynamicProgrammingSegment(base) }
  ];

  const baseline = segmented[2].tokens;
  const results = segmented.map((result) => ({
    ...result,
    tagged: tagTokens(result.tokens),
    metrics: collectMetrics(result.tokens, baseline)
  }));

  renderAlgorithms(results);
  renderStats(results);
  renderInsights(base, results);
}

analyze();
