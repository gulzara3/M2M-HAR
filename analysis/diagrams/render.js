const { chromium } = require('playwright');
(async () => {
  const [,, html, png, pdf] = process.argv;
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1640, height: 800 }, deviceScaleFactor: 3 });
  await p.goto('file://' + html); await p.waitForTimeout(300);
  const el = await p.$('body'); const bb = await el.boundingBox();
  await p.screenshot({ path: png, clip: { x: 0, y: 0, width: Math.ceil(bb.width), height: Math.ceil(bb.height) } });
  if (pdf) await p.pdf({ path: pdf, width: Math.ceil(bb.width) + 'px', height: Math.ceil(bb.height) + 2 + 'px', printBackground: true, pageRanges: '1' });
  await b.close();
})();
