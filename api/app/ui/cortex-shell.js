// Cortex shell — shared auth and header chrome for every cortex page.
//
// Loaded by the unified header partial (_nav.html). Pages get:
//   - window.cortexAuth: { getKey, setKey, clearKey, validate, onChange }
//   - A header pill that shows key status (none / unchecked / valid / invalid)
//   - A connection drawer to paste/replace the key, visible from any page
//   - localStorage migration from legacy per-page keys to one canonical key
//
// Pages should:
//   - Use window.cortexAuth.getKey() instead of reading localStorage directly
//   - Use window.cortexAuth.onChange(callback) to reload when the key changes
//   - Show window.cortexAuth.showError(msg) on auth failures

(function () {
  const KEY = 'cortexKey';
  const LEGACY_KEYS = ['apiKey', 'hiveApiKey', 'hiveWriterKey', 'hiveAdminKey', 'adminKey'];
  const STATUS = { NONE: 'none', UNCHECKED: 'unchecked', VALID: 'valid', INVALID: 'invalid' };
  const subscribers = [];
  let currentStatus = STATUS.NONE;
  let currentIdentity = null;

  function migrate() {
    if (localStorage.getItem(KEY)) return;
    for (const k of LEGACY_KEYS) {
      const v = localStorage.getItem(k);
      if (v && v.length > 8) {
        localStorage.setItem(KEY, v);
        console.log(`[cortex] migrated key from legacy "${k}"`);
        return;
      }
    }
  }

  function getKey() { return localStorage.getItem(KEY) || ''; }
  function setKey(v) {
    if (v) localStorage.setItem(KEY, v);
    else localStorage.removeItem(KEY);
    notify();
  }
  function clearKey() { setKey(''); }
  function notify() { subscribers.forEach(fn => { try { fn(); } catch (e) {} }); }

  async function validate() {
    const k = getKey();
    if (!k) { setStatus(STATUS.NONE, null); return false; }
    setStatus(STATUS.UNCHECKED, null);
    try {
      const r = await fetch('/whoami', { headers: { 'X-API-Key': k } });
      if (r.ok) {
        currentIdentity = await r.json();
        setStatus(STATUS.VALID, currentIdentity);
        return true;
      }
      setStatus(STATUS.INVALID, null);
      return false;
    } catch (e) {
      setStatus(STATUS.INVALID, null);
      return false;
    }
  }

  function setStatus(s, identity) {
    currentStatus = s;
    currentIdentity = identity;
    renderPill();
    notify();
  }

  function renderPill() {
    const pill = document.getElementById('cortex-conn-pill');
    if (!pill) return;
    const k = getKey();
    let label = 'No key';
    let cls = 'none';
    if (currentStatus === STATUS.UNCHECKED) { label = 'Checking…'; cls = 'unchecked'; }
    else if (currentStatus === STATUS.VALID) {
      const role = currentIdentity?.role || '?';
      label = `${role} · ${k.slice(0, 8)}…`;
      cls = 'valid';
    }
    else if (currentStatus === STATUS.INVALID) { label = `Invalid · ${k.slice(0, 8)}…`; cls = 'invalid'; }
    else if (k) { label = `${k.slice(0, 8)}…`; cls = 'unchecked'; }
    pill.className = `cortex-pill ${cls}`;
    pill.textContent = label;
  }

  function openDrawer() {
    const d = document.getElementById('cortex-conn-drawer');
    const input = document.getElementById('cortex-conn-input');
    if (!d || !input) return;
    input.value = getKey();
    d.classList.add('open');
    setTimeout(() => input.focus(), 50);
  }
  function closeDrawer() {
    const d = document.getElementById('cortex-conn-drawer');
    if (d) d.classList.remove('open');
  }

  function showError(msg) {
    const bar = document.getElementById('cortex-error-bar');
    if (!bar) {
      console.warn('[cortex] error bar not in DOM:', msg);
      return;
    }
    bar.textContent = msg;
    bar.style.display = 'block';
    clearTimeout(showError._t);
    showError._t = setTimeout(() => { bar.style.display = 'none'; }, 5000);
  }

  function onChange(fn) { subscribers.push(fn); }

  // Wire up DOM listeners after parse
  function wire() {
    const pill = document.getElementById('cortex-conn-pill');
    if (pill) pill.addEventListener('click', openDrawer);

    const closeBtn = document.getElementById('cortex-conn-close');
    if (closeBtn) closeBtn.addEventListener('click', closeDrawer);

    const saveBtn = document.getElementById('cortex-conn-save');
    if (saveBtn) saveBtn.addEventListener('click', async () => {
      const input = document.getElementById('cortex-conn-input');
      const newKey = input.value.trim();
      const prevKey = getKey();
      setKey(newKey);
      const ok = await validate();
      if (ok) {
        closeDrawer();
        // If key actually changed, hard reload so every page re-fetches under
        // the new identity. Avoids the messiness of trying to reactively
        // re-bind Alpine pages from outside their data scope.
        if (newKey !== prevKey) {
          setTimeout(() => location.reload(), 300);
        }
      }
    });

    const clearBtn = document.getElementById('cortex-conn-clear');
    if (clearBtn) clearBtn.addEventListener('click', () => { clearKey(); validate(); });

    // ESC closes drawer
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });

    renderPill();
    validate(); // fire-and-forget on load
  }

  // Public API
  window.cortexAuth = {
    getKey, setKey, clearKey, validate, onChange, showError,
    KEY, STATUS,
    get status() { return currentStatus; },
    get identity() { return currentIdentity; },
  };

  // Wrap fetch helper: throws on 401 so pages can catch + showError
  window.cortexFetch = async function (path, opts = {}) {
    const headers = Object.assign({}, opts.headers || {}, { 'X-API-Key': getKey() });
    const r = await fetch(path, Object.assign({}, opts, { headers }));
    if (r.status === 401) {
      showError('Auth failed — set or replace your API key (header pill).');
      setStatus(STATUS.INVALID, null);
      const err = new Error('unauthorized');
      err.status = 401;
      throw err;
    }
    return r;
  };

  // Run migration immediately so getKey() works before wire()
  migrate();

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wire);
  } else {
    wire();
  }
})();
