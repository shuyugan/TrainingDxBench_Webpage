(() => {
  "use strict";

  const root = document.documentElement;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  function setTheme(theme) {
    root.dataset.theme = theme;
    try {
      localStorage.setItem("trainingdx-theme", theme);
    } catch {
      // The theme still applies for this page view when storage is unavailable.
    }
    const themeColor = document.querySelector('meta[name="theme-color"]');
    if (themeColor) {
      themeColor.content = theme === "dark" ? "#080c16" : "#f6f8fc";
    }
    window.dispatchEvent(new CustomEvent("trainingdx:theme"));
  }

  const themeToggle = document.querySelector("#theme-toggle");
  themeToggle?.addEventListener("click", () => {
    setTheme(root.dataset.theme === "dark" ? "light" : "dark");
  });

  const menuToggle = document.querySelector("#menu-toggle");
  const mobileNav = document.querySelector("#mobile-nav");

  function closeMobileNav() {
    if (!menuToggle || !mobileNav) return;
    menuToggle.setAttribute("aria-expanded", "false");
    menuToggle.setAttribute("aria-label", "Open navigation");
    mobileNav.hidden = true;
  }

  menuToggle?.addEventListener("click", () => {
    const open = menuToggle.getAttribute("aria-expanded") === "true";
    menuToggle.setAttribute("aria-expanded", String(!open));
    menuToggle.setAttribute("aria-label", open ? "Open navigation" : "Close navigation");
    if (mobileNav) mobileNav.hidden = open;
  });

  mobileNav?.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", closeMobileNav);
  });

  const desktopNavigation = window.matchMedia("(min-width: 1081px)");
  desktopNavigation.addEventListener?.("change", (event) => {
    if (event.matches) closeMobileNav();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeMobileNav();
  });

  document.addEventListener("click", (event) => {
    if (
      mobileNav &&
      menuToggle &&
      !mobileNav.hidden &&
      event.target instanceof Node &&
      !mobileNav.contains(event.target) &&
      !menuToggle.contains(event.target)
    ) {
      closeMobileNav();
    }
  });

  const revealElements = document.querySelectorAll(".reveal");
  if (reducedMotion.matches || !("IntersectionObserver" in window)) {
    revealElements.forEach((element) => element.classList.add("is-visible"));
  } else {
    const revealObserver = new IntersectionObserver(
      (entries, observer) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.12 },
    );
    revealElements.forEach((element) => revealObserver.observe(element));
  }

  const navLinks = [...document.querySelectorAll("#desktop-nav a")];
  const navTargets = navLinks
    .map((link) => link.getAttribute("href"))
    .filter((href) => href?.startsWith("#"))
    .map((href) => document.querySelector(href))
    .filter(Boolean);

  if ("IntersectionObserver" in window) {
    const navObserver = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
        if (!visible) return;
        navLinks.forEach((link) => {
          link.classList.toggle(
            "is-active",
            link.getAttribute("href") === `#${visible.target.id}`,
          );
        });
      },
      { rootMargin: "-30% 0px -58% 0px", threshold: [0.05, 0.25, 0.5] },
    );
    navTargets.forEach((target) => navObserver.observe(target));
  }

  const escapeHtml = (value) =>
    value
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");

  const pythonKeywords = new Set([
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "False",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "None",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "True",
    "try",
    "while",
    "with",
    "yield",
  ]);

  function highlightPython(line) {
    let output = "";
    let index = 0;

    while (index < line.length) {
      const char = line[index];

      if (char === "#") {
        output += `<span class="token-comment">${escapeHtml(line.slice(index))}</span>`;
        break;
      }

      if (char === "'" || char === '"') {
        const quote = char;
        let end = index + 1;
        while (end < line.length) {
          if (line[end] === "\\") {
            end += 2;
            continue;
          }
          end += 1;
          if (line[end - 1] === quote) break;
        }
        output += `<span class="token-string">${escapeHtml(line.slice(index, end))}</span>`;
        index = end;
        continue;
      }

      if (/[A-Za-z_]/.test(char)) {
        let end = index + 1;
        while (end < line.length && /[A-Za-z0-9_]/.test(line[end])) end += 1;
        const word = line.slice(index, end);
        let className = "";
        if (pythonKeywords.has(word)) {
          className = "token-keyword";
        } else if (/^\s*\(/.test(line.slice(end))) {
          className = "token-function";
        }
        output += className
          ? `<span class="${className}">${word}</span>`
          : word;
        index = end;
        continue;
      }

      if (/\d/.test(char)) {
        let end = index + 1;
        while (end < line.length && /[\d._]/.test(line[end])) end += 1;
        output += `<span class="token-number">${escapeHtml(line.slice(index, end))}</span>`;
        index = end;
        continue;
      }

      if ("=+-*/%<>!|&:@".includes(char)) {
        output += `<span class="token-operator">${escapeHtml(char)}</span>`;
      } else {
        output += escapeHtml(char);
      }
      index += 1;
    }
    return output;
  }

  function highlightJson(line) {
    let output = "";
    let index = 0;

    while (index < line.length) {
      const char = line[index];
      if (char === '"') {
        let end = index + 1;
        while (end < line.length) {
          if (line[end] === "\\") {
            end += 2;
            continue;
          }
          end += 1;
          if (line[end - 1] === '"') break;
        }
        const token = line.slice(index, end);
        const isKey = /^\s*:/.test(line.slice(end));
        output += `<span class="${isKey ? "token-key" : "token-string"}">${escapeHtml(token)}</span>`;
        index = end;
        continue;
      }

      const primitive = line.slice(index).match(/^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|^(?:true|false|null)\b/);
      if (primitive) {
        const token = primitive[0];
        const className = /^(true|false|null)$/.test(token)
          ? "token-boolean"
          : "token-number";
        output += `<span class="${className}">${token}</span>`;
        index += token.length;
        continue;
      }

      output += escapeHtml(char);
      index += 1;
    }
    return output;
  }

  function highlightMarkdown(line) {
    const heading = line.match(/^(#{1,6})(\s+)(.*)$/);
    if (heading) {
      return `<span class="token-heading">${escapeHtml(heading[1] + heading[2])}${highlightMarkdownInline(heading[3])}</span>`;
    }
    return highlightMarkdownInline(line);
  }

  function highlightMarkdownInline(line) {
    let output = "";
    let index = 0;
    while (index < line.length) {
      if (line[index] === "`") {
        const end = line.indexOf("`", index + 1);
        if (end !== -1) {
          output += `<span class="token-inline-code">${escapeHtml(line.slice(index, end + 1))}</span>`;
          index = end + 1;
          continue;
        }
      }
      if (line[index] === "[" || line[index] === "*" || line[index] === "-") {
        output += `<span class="token-operator">${escapeHtml(line[index])}</span>`;
      } else {
        output += escapeHtml(line[index]);
      }
      index += 1;
    }
    return output;
  }

  function highlightLine(line, language) {
    if (language === "python") return highlightPython(line);
    if (language === "json") return highlightJson(line);
    if (language === "markdown") return highlightMarkdown(line);
    return escapeHtml(line);
  }

  function renderCode(source, language) {
    return source
      .replaceAll("\r\n", "\n")
      .split("\n")
      .map(
        (line, index) =>
          `<span class="code-line"><span class="line-number" aria-hidden="true">${index + 1}</span><span class="line-content">${highlightLine(line, language) || " "}</span></span>`,
      )
      .join("");
  }

  const workspaceSourceFiles = [
    "collator.py",
    "common.py",
    "data.py",
    "evaluate.py",
    "model.py",
    "settings.py",
    "train.py",
    "training.py",
  ];

  const workspaceModelFiles = [
    ["base/chat_template.jinja", "jinja"],
    ["base/config.json", "json"],
    ["base/generation_config.json", "json"],
    ["base/tokenizer_config.json", "json"],
    ["adapter/adapter_config.json", "json"],
  ];

  const fileItem = (label, path, language, extra = {}) => ({
    type: "file",
    label,
    path,
    language,
    ...extra,
  });

  const assetItem = (label, size) => ({
    type: "asset",
    label,
    size,
  });

  const folderItem = (label, children, extra = {}) => ({
    type: "folder",
    label,
    children,
    ...extra,
  });

  function workspaceTree(arm) {
    const prefix = `./example/${arm}/workspace`;
    const corrected = arm === "corrected";
    const path = (relative) => `${prefix}/${relative}`;
    return folderItem(
      `${arm}/workspace`,
      [
        fileItem("README.md", path("README.md"), "markdown"),
        fileItem("launch.sh", path("launch.sh"), "text"),
        fileItem("requirements.txt", path("requirements.txt"), "text"),
        folderItem(
          "source",
          workspaceSourceFiles.map((name) =>
            fileItem(name, path(`source/${name}`), "python"),
          ),
          { open: true, badge: "8 files" },
        ),
        folderItem(
          "data",
          [
            assetItem("train.jsonl", "4.3 MB"),
            assetItem("dev.jsonl", "523 KB"),
            assetItem("validation.jsonl", "1.1 MB"),
          ],
          { badge: "5.7 MB" },
        ),
        folderItem(
          "model",
          [
            folderItem(
              "base",
              [
                ...workspaceModelFiles
                  .filter(([name]) => name.startsWith("base/"))
                  .map(([name, language]) =>
                    fileItem(name.split("/").at(-1), path(`model/${name}`), language),
                  ),
                assetItem("tokenizer.json", "11.4 MB"),
                assetItem("model.safetensors", "988 MB"),
              ],
              { badge: "base model" },
            ),
            folderItem(
              "adapter",
              [
                fileItem(
                  "adapter_config.json",
                  path("model/adapter/adapter_config.json"),
                  "json",
                ),
                assetItem("adapter.safetensors", "4.3 MB"),
              ],
              { badge: "LoRA" },
            ),
          ],
          { badge: "958 MB" },
        ),
        folderItem(
          "training",
          [
            folderItem("base_validation", [
              fileItem(
                "summary.json",
                path("training/base_validation/summary.json"),
                "json",
              ),
              assetItem("predictions.jsonl", "3.2 MB"),
            ]),
            folderItem("checkpoint", [
              fileItem(
                "trainer_state.json",
                path("training/checkpoint/trainer_state.json"),
                "json",
              ),
              assetItem("optimizer.pt", "8.7 MB"),
              assetItem("rng_rank_0..3.pt", "4 files"),
              assetItem("scheduler.pt", "1 KB"),
            ]),
            folderItem("dev_validation", [
              fileItem(
                "summary.json",
                path("training/dev_validation/summary.json"),
                "json",
              ),
              assetItem(
                "predictions.jsonl",
                corrected ? "1.58 MB" : "1.57 MB",
              ),
            ]),
            folderItem("final_validation", [
              fileItem(
                "summary.json",
                path("training/final_validation/summary.json"),
                "json",
              ),
              assetItem(
                "predictions.jsonl",
                corrected ? "3.23 MB" : "3.21 MB",
              ),
            ]),
            fileItem(
              "training_trace.jsonl",
              path("training/training_trace.jsonl"),
              "json",
            ),
          ],
          { badge: "17 MB" },
        ),
      ],
      { open: arm === "flawed", badge: corrected ? "corrected" : "affected" },
    );
  }

  const treeDefinitions = {
    task: [
      fileItem(
        "Task.md",
        "./example/Task.md",
        "markdown",
        {
          active: true,
          labelPath: "tasks/padding-free-chat-packing/Task.md",
        },
      ),
      workspaceTree("flawed"),
      workspaceTree("corrected"),
      folderItem(
        "private",
        [
          fileItem(
            "oracle.json",
            "./example/private/oracle.json",
            "json",
          ),
          folderItem(
            "verifier",
            [
              fileItem(
                "contract.json",
                "./example/private/verifier/contract.json",
                "json",
              ),
              fileItem(
                "score.py",
                "./example/private/verifier/score.py",
                "python",
              ),
            ],
            { open: true },
          ),
        ],
        { badge: "reference" },
      ),
    ],
    evaluation: [
      folderItem(
        "copilot--claude-sonnet-5",
        [
          fileItem(
            "trajectory.jsonl",
            "./example/evaluation/trajectory.jsonl",
            "json",
            { size: "sanitized" },
          ),
          fileItem(
            "answer.txt",
            "./example/evaluation/answer.txt",
            "text",
            {
              active: true,
              labelPath:
                "evaluation/results/padding-free-chat-packing/copilot--claude-sonnet-5/answer.txt",
            },
          ),
          fileItem(
            "submission.json",
            "./example/evaluation/submission.json",
            "json",
          ),
          fileItem(
            "deleted_paths.json",
            "./example/evaluation/deleted_paths.json",
            "json",
          ),
          folderItem(
            "repair/source",
            [
              fileItem(
                "collator.py",
                "./example/evaluation/repair/source/collator.py",
                "python",
              ),
            ],
            { open: true },
          ),
          fileItem(
            "judge-trajectory.jsonl",
            "./example/evaluation/judge-trajectory.jsonl",
            "json",
            { size: "sanitized" },
          ),
          fileItem(
            "judge.json",
            "./example/evaluation/judge.json",
            "json",
          ),
          fileItem(
            "result.json",
            "./example/evaluation/result.json",
            "json",
          ),
        ],
        { open: true, badge: "failed" },
      ),
    ],
  };

  const folderIcon =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h6l2 2h10v11H3z"></path></svg>';
  const fileIcon =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h9l3 3v15H6zM14 3v4h4"></path></svg>';
  const downloadIcon =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12M7 10l5 5 5-5M5 21h14"></path></svg>';

  function renderTreeItem(item, depth = 0) {
    const badge = item.badge
      ? `<span class="tree-badge">${escapeHtml(item.badge)}</span>`
      : "";
    if (item.type === "folder") {
      const children = item.children
        .map((child) => renderTreeItem(child, depth + 1))
        .join("");
      return `<details class="tree-directory" ${item.open ? "open" : ""}>
        <summary style="--tree-depth:${depth}">${folderIcon}<span>${escapeHtml(item.label)}</span>${badge}</summary>
        <div class="tree-children">${children}</div>
      </details>`;
    }
    if (item.type === "asset") {
      return `<div class="tree-asset" style="--tree-depth:${depth}" title="Included in the complete task package; not expanded in the web preview">
        ${fileIcon}<span>${escapeHtml(item.label)}</span><span class="tree-badge">${escapeHtml(item.size)}</span>
      </div>`;
    }
    if (item.type === "download") {
      return `<a class="tree-download" style="--tree-depth:${depth}" href="${escapeHtml(item.path)}" download>
        ${downloadIcon}<span>${escapeHtml(item.label)}</span><span class="tree-badge">${escapeHtml(item.size)}</span>
      </a>`;
    }
    const labelPath = item.labelPath || item.path.replace("./example/", "");
    const size = item.size
      ? `<span class="tree-badge">${escapeHtml(item.size)}</span>`
      : "";
    return `<button class="file-button ${item.active ? "is-active" : ""}" style="--tree-depth:${depth}" type="button"
      data-code-file="${escapeHtml(item.path)}"
      data-code-label="${escapeHtml(labelPath)}"
      data-language="${escapeHtml(item.language)}">
      ${fileIcon}<span>${escapeHtml(item.label)}</span>${size}
    </button>`;
  }

  document.querySelectorAll("[data-generated-tree]").forEach((tree) => {
    const definition = treeDefinitions[tree.dataset.generatedTree] || [];
    tree.innerHTML = definition.map((item) => renderTreeItem(item)).join("");
  });

  const contentCache = new Map();

  async function fetchText(path) {
    if (contentCache.has(path)) return contentCache.get(path);
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`Could not load ${path} (${response.status})`);
    }
    const content = await response.text();
    contentCache.set(path, content);
    return content;
  }

  document.querySelectorAll("[data-code-browser]").forEach((browser) => {
    const buttons = [...browser.querySelectorAll("[data-code-file]")];
    const output = browser.querySelector("[data-code-output]");
    const loading = browser.querySelector("[data-code-loading]");
    const pathLabel = browser.querySelector("[data-browser-path]");
    const wrapButton = browser.querySelector("[data-wrap-toggle]");
    const copyButton = browser.querySelector("[data-copy-code]");
    let requestId = 0;

    async function activate(button) {
      if (!button || !output || !loading) return;
      const currentRequest = ++requestId;
      const file = button.dataset.codeFile;
      const label = button.dataset.codeLabel || file;
      const language = button.dataset.language || "text";

      buttons.forEach((candidate) => {
        const active = candidate === button;
        candidate.classList.toggle("is-active", active);
        if (candidate.getAttribute("role") === "tab") {
          candidate.setAttribute("aria-selected", String(active));
          candidate.tabIndex = active ? 0 : -1;
          if (active && candidate.id) {
            browser
              .querySelector('[role="tabpanel"]')
              ?.setAttribute("aria-labelledby", candidate.id);
          }
        } else {
          candidate.removeAttribute("aria-selected");
        }
        if (active) {
          candidate.setAttribute("aria-current", "true");
        } else {
          candidate.removeAttribute("aria-current");
        }
      });

      loading.hidden = false;
      loading.textContent = `Loading ${label.split("/").at(-1)}…`;
      if (pathLabel) pathLabel.textContent = label;

      try {
        const content = await fetchText(file);
        if (currentRequest !== requestId) return;
        output.dataset.rawCode = content;
        output.innerHTML = renderCode(content, language);
        output.scrollTop = 0;
        output.parentElement.scrollTop = 0;
        output.parentElement.scrollLeft = 0;
      } catch (error) {
        if (currentRequest !== requestId) return;
        output.dataset.rawCode = "";
        output.innerHTML = `<span class="code-line"><span class="line-number">!</span><span class="line-content token-comment">${escapeHtml(error.message)}</span></span>`;
      } finally {
        if (currentRequest === requestId) loading.hidden = true;
      }
    }

    buttons.forEach((button) => {
      button.addEventListener("click", () => activate(button));
    });

    const tabs = buttons.filter((button) => button.getAttribute("role") === "tab");
    tabs.forEach((tab, index) => {
      tab.addEventListener("keydown", (event) => {
        let nextIndex = null;
        if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
        if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = tabs.length - 1;
        if (nextIndex === null) return;
        event.preventDefault();
        tabs[nextIndex].focus();
        activate(tabs[nextIndex]);
      });
    });

    wrapButton?.addEventListener("click", () => {
      const wrapped = output.classList.toggle("is-wrapped");
      wrapButton.setAttribute("aria-pressed", String(wrapped));
    });

    copyButton?.addEventListener("click", async () => {
      const label = copyButton.querySelector("span");
      const original = label?.textContent || "Copy";
      try {
        await navigator.clipboard.writeText(output?.dataset.rawCode || "");
        if (label) label.textContent = "Copied";
      } catch {
        if (label) label.textContent = "Unavailable";
      }
      window.setTimeout(() => {
        if (label) label.textContent = original;
      }, 1400);
    });

    activate(buttons.find((button) => button.classList.contains("is-active")) || buttons[0]);
  });

  const imageDialog = document.querySelector("#image-dialog");
  const dialogImage = imageDialog?.querySelector("img");

  document.querySelectorAll(".diagram-expand").forEach((button) => {
    button.addEventListener("click", () => {
      if (!imageDialog || !dialogImage) return;
      dialogImage.src = button.dataset.image;
      dialogImage.alt = button.dataset.alt || "";
      imageDialog.showModal();
    });
  });

  imageDialog?.querySelector(".dialog-close")?.addEventListener("click", () => {
    imageDialog.close();
  });

  imageDialog?.addEventListener("click", (event) => {
    if (event.target === imageDialog) imageDialog.close();
  });

  const year = document.querySelector("#current-year");
  if (year) year.textContent = String(new Date().getFullYear());

  function initializeNeuralCanvas() {
    const canvas = document.querySelector("#neural-canvas");
    const hero = document.querySelector(".hero");
    if (!(canvas instanceof HTMLCanvasElement) || !hero) return;

    const context = canvas.getContext("2d");
    if (!context) return;

    let width = 0;
    let height = 0;
    let nodes = [];
    let edges = [];
    let colors = {};
    let animationFrame = null;
    let heroVisible = true;
    let pageVisible = !document.hidden;
    const pointer = { x: 0, y: 0, active: false };

    function readColors() {
      const styles = getComputedStyle(root);
      colors = {
        line: styles.getPropertyValue("--network-line").trim(),
        node: styles.getPropertyValue("--network-node").trim(),
        hot: styles.getPropertyValue("--network-hot").trim(),
      };
    }

    function buildGraph() {
      const layerCounts = width < 700 ? [4, 6, 7, 6, 4] : [5, 8, 10, 10, 8, 5];
      nodes = [];
      edges = [];
      const horizontalMargin = width < 700 ? 12 : 35;
      const verticalMargin = height * 0.12;

      layerCounts.forEach((count, layer) => {
        const x =
          horizontalMargin +
          ((width - horizontalMargin * 2) * layer) / (layerCounts.length - 1);
        for (let index = 0; index < count; index += 1) {
          const y =
            verticalMargin +
            ((height - verticalMargin * 2) * (index + 0.5)) / count +
            (Math.random() - 0.5) * 28;
          nodes.push({
            layer,
            index,
            x,
            y,
            baseY: y,
            radius: Math.random() > 0.86 ? 3.2 : 2.1,
            phase: Math.random() * Math.PI * 2,
            speed: 0.3 + Math.random() * 0.4,
            hot: Math.random() > 0.92,
          });
        }
      });

      for (let layer = 0; layer < layerCounts.length - 1; layer += 1) {
        const current = nodes.filter((node) => node.layer === layer);
        const next = nodes.filter((node) => node.layer === layer + 1);
        current.forEach((node) => {
          const nearest = [...next]
            .sort((left, right) => Math.abs(left.y - node.y) - Math.abs(right.y - node.y))
            .slice(0, Math.random() > 0.55 ? 3 : 2);
          nearest.forEach((target) => {
            edges.push({
              source: node,
              target,
              phase: Math.random(),
              pulse: Math.random() > 0.72,
              speed: 0.00005 + Math.random() * 0.00008,
            });
          });
        });
      }
    }

    function resize() {
      const bounds = hero.getBoundingClientRect();
      width = Math.max(1, bounds.width);
      height = Math.max(1, bounds.height);
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      readColors();
      buildGraph();
      draw(performance.now());
    }

    function nodePosition(node, time) {
      const drift = reducedMotion.matches
        ? 0
        : Math.sin(time * 0.00035 * node.speed + node.phase) * 9;
      return { x: node.x, y: node.baseY + drift };
    }

    function draw(time) {
      context.clearRect(0, 0, width, height);

      edges.forEach((edge) => {
        const source = nodePosition(edge.source, time);
        const target = nodePosition(edge.target, time);
        const middleX = (source.x + target.x) / 2;
        context.beginPath();
        context.moveTo(source.x, source.y);
        context.bezierCurveTo(middleX, source.y, middleX, target.y, target.x, target.y);
        context.strokeStyle = `rgba(${colors.line}, 0.12)`;
        context.lineWidth = 0.8;
        context.stroke();

        if (edge.pulse && !reducedMotion.matches) {
          const progress = (edge.phase + time * edge.speed) % 1;
          const inverse = 1 - progress;
          const pulseX =
            inverse ** 3 * source.x +
            3 * inverse ** 2 * progress * middleX +
            3 * inverse * progress ** 2 * middleX +
            progress ** 3 * target.x;
          const pulseY =
            inverse ** 3 * source.y +
            3 * inverse ** 2 * progress * source.y +
            3 * inverse * progress ** 2 * target.y +
            progress ** 3 * target.y;
          const glow = context.createRadialGradient(pulseX, pulseY, 0, pulseX, pulseY, 8);
          glow.addColorStop(0, `rgba(${colors.hot}, 0.75)`);
          glow.addColorStop(1, `rgba(${colors.hot}, 0)`);
          context.fillStyle = glow;
          context.beginPath();
          context.arc(pulseX, pulseY, 8, 0, Math.PI * 2);
          context.fill();
        }
      });

      nodes.forEach((node) => {
        const position = nodePosition(node, time);
        const pointerDistance = pointer.active
          ? Math.hypot(pointer.x - position.x, pointer.y - position.y)
          : Number.POSITIVE_INFINITY;
        const highlighted = pointerDistance < 120;
        const radius = node.radius + (highlighted ? (120 - pointerDistance) / 60 : 0);
        context.beginPath();
        context.arc(position.x, position.y, radius, 0, Math.PI * 2);
        context.fillStyle = node.hot
          ? `rgba(${colors.hot}, 0.8)`
          : `rgba(${colors.node}, ${highlighted ? 0.75 : 0.42})`;
        context.fill();

        if (highlighted || node.hot) {
          context.beginPath();
          context.arc(position.x, position.y, radius + 5, 0, Math.PI * 2);
          context.strokeStyle = node.hot
            ? `rgba(${colors.hot}, 0.16)`
            : `rgba(${colors.node}, 0.14)`;
          context.lineWidth = 1;
          context.stroke();
        }
      });
    }

    function animate(time) {
      draw(time);
      animationFrame = window.requestAnimationFrame(animate);
    }

    function startAnimation() {
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
      animationFrame = null;
      if (reducedMotion.matches || !heroVisible || !pageVisible) {
        draw(performance.now());
      } else {
        animationFrame = window.requestAnimationFrame(animate);
      }
    }

    hero.addEventListener("pointermove", (event) => {
      const bounds = hero.getBoundingClientRect();
      pointer.x = event.clientX - bounds.left;
      pointer.y = event.clientY - bounds.top;
      pointer.active = true;
    });
    hero.addEventListener("pointerleave", () => {
      pointer.active = false;
    });
    window.addEventListener("resize", resize, { passive: true });
    window.addEventListener("trainingdx:theme", () => {
      readColors();
      draw(performance.now());
    });
    reducedMotion.addEventListener?.("change", startAnimation);
    document.addEventListener("visibilitychange", () => {
      pageVisible = !document.hidden;
      startAnimation();
    });
    if ("IntersectionObserver" in window) {
      const heroObserver = new IntersectionObserver((entries) => {
        heroVisible = entries[0]?.isIntersecting ?? true;
        startAnimation();
      });
      heroObserver.observe(hero);
    }

    resize();
    startAnimation();
  }

  initializeNeuralCanvas();
})();
