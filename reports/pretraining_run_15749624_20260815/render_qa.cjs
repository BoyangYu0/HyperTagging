const { chromium } = require("C:/Users/Karl_/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");

(async () => {
  const report = "file:///C:/Users/Karl_/Documents/Codex/2026-08-12/c/pretraining_run_15749624_report_20260815/report.html";
  const output = "C:/Users/Karl_/Documents/Codex/2026-08-12/c/pretraining_run_15749624_report_20260815/report_full.png";
  const browser = await chromium.launch({
    executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto(report, { waitUntil: "networkidle" });
  await page.waitForTimeout(3000);
  const stats = await page.evaluate(() => ({
    title: document.title,
    scrollHeight: document.documentElement.scrollHeight,
    svgCount: document.querySelectorAll("svg").length,
    canvasCount: document.querySelectorAll("canvas").length,
    tableCount: document.querySelectorAll("table").length,
    bodyTextLength: document.body.innerText.length,
    scatterSeries: Array.from(document.querySelectorAll(".recharts-scatter")).map((node) => ({
      points: node.querySelectorAll(".recharts-scatter-symbol").length,
      fill: node.getAttribute("fill"),
      symbolFill: node.querySelector("path")?.getAttribute("fill") ?? null,
      symbolTransform: node.querySelector(".recharts-scatter-symbol")?.getAttribute("transform") ?? null,
      bbox: (() => { const box = node.getBoundingClientRect(); return { x: box.x, y: box.y, width: box.width, height: box.height }; })(),
      firstSymbol: node.querySelector(".recharts-scatter-symbol")?.outerHTML.slice(0, 500) ?? null,
      html: node.outerHTML.slice(0, 180),
    })),
  }));
  const chartTitle = page.getByText("Step-3000 validation embeddings in the joint PCA basis", { exact: true }).last();
  const box = await chartTitle.boundingBox();
  if (box) {
    await chartTitle.scrollIntoViewIfNeeded();
    await page.waitForTimeout(500);
    await page.screenshot({ path: "C:/Users/Karl_/Documents/Codex/2026-08-12/c/pretraining_run_15749624_report_20260815/embedding_compare_qa.png" });
  }
  await page.screenshot({ path: output, fullPage: true });
  console.log(JSON.stringify({ ...stats, consoleErrors: errors, screenshot: output }));
  await browser.close();
})();
