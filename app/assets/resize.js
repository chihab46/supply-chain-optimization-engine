// A Plotly figure first mounted in a hidden Dash tab can retain its 700×450
// fallback canvas. Fit its actual canvas to the measured, explicitly sized host.
(() => {
  const tracked = new WeakSet();
  const pending = new Set();
  let frame = null;
  const fit = root => {
    pending.add(root);
    if (frame !== null) return;
    frame = requestAnimationFrame(() => {
      frame = null;
      const roots = [...pending];
      pending.clear();
      roots.forEach(host => {
        const graph = host.matches('.js-plotly-plot') ? host : host.querySelector('.js-plotly-plot');
        const box = host.getBoundingClientRect();
        if (!graph?._fullLayout || !window.Plotly || box.width < 1 || box.height < 1) return;
        const width = Math.floor(box.width), height = Math.floor(box.height);
        if (Math.abs(graph._fullLayout.width - width) > 1 ||
            Math.abs(graph._fullLayout.height - height) > 1) {
          window.Plotly.relayout(graph, {width, height, autosize: false});
        }
      });
    });
  };
  const resize = new ResizeObserver(entries => entries.forEach(entry => fit(entry.target)));
  const attach = node => {
    if (!(node instanceof Element)) return;
    const hosts = node.matches('.dash-graph') ? [node] : [...node.querySelectorAll('.dash-graph')];
    hosts.forEach(host => {
      if (!tracked.has(host)) {tracked.add(host); resize.observe(host);}
      fit(host);
    });
  };
  new MutationObserver(records => records.forEach(record => {
    const host = record.target instanceof Element && record.target.closest('.dash-graph');
    if (host) fit(host);
    record.addedNodes.forEach(attach);
  })).observe(document.documentElement, {childList: true, subtree: true});
  attach(document.documentElement);
})();
